#!/usr/bin/env python
"""Fine-tune Marble splats against the real posed frames with gsplat (runs on the GPU box).

Input dir from finetune_export.py: splats.npz, cameras.npz, frames/, masks/, depth.npz.
Splats visible in any training view are optimised (means, quats, scales, opacities, SH0 colour);
the rest are frozen. Loss = masked L1 + lambda_ssim * (1 - SSIM) (person = ignore) + a quadratic
tether of means / log-scales to their Marble values, with a hard clamp on the displacement.
Every 25th frame is held out for PSNR.

runs 1-4 (defaults): no densification, no pruning, one tether weight for every visible splat.
run 5 (opt-in flags): an "observed" mask (projects inside the image and within --observed-depth-tol
of the wvp depth in >= --observed-min-views training views, computed once from the Marble init);
the tether is scaled by --reg-observed-scale on observed splats so they can move to the real surface;
--prune drops observed splats that went transparent, and after --prune-depth-start also observed
splats sitting more than --prune-depth-cm in FRONT of the measured surface (the veil); --densify
runs 3DGS split/clone restricted to observed splats up to --max-splats; --lambda-opacity-entropy
pushes observed opacities to 0/1.

    python finetune_gsplat.py --data ~/w/ft-corridor --out ~/w/ft-corridor/run1 --iters 3000
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization

SH_C0 = 0.28209479177387814
CM_PER_UNIT = 260.8  # 1 native unit = 2.608 m (1.7 m / body height 0.6518811)


def load_images(d: Path, n: int, dev):
    imgs = np.stack([np.asarray(Image.open(d / "frames" / f"{i:04d}.jpg")) for i in range(n)])
    masks = np.stack([np.asarray(Image.open(d / "masks" / f"{i:04d}.png")) for i in range(n)])
    imgs = torch.from_numpy(imgs).to(dev)  # uint8 (F,H,W,3)
    valid = torch.from_numpy(masks < 128).to(dev)  # True = use in loss
    return imgs, valid


def gaussian_window(size=11, sigma=1.5, dev="cuda"):
    g = torch.arange(size, dtype=torch.float32, device=dev) - size // 2
    g = torch.exp(-g * g / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim_map(x, y, win):
    """x, y: (1,3,H,W) in [0,1]. Returns per-pixel SSIM (1,3,H,W)."""
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.conv2d(x, win, padding=5, groups=3)
    mu_y = F.conv2d(y, win, padding=5, groups=3)
    sxx = F.conv2d(x * x, win, padding=5, groups=3) - mu_x**2
    syy = F.conv2d(y * y, win, padding=5, groups=3) - mu_y**2
    sxy = F.conv2d(x * y, win, padding=5, groups=3) - mu_x * mu_y
    return ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sxx + syy + C2))


def quat_to_rotmat(q):
    """wxyz (n,4) -> (n,3,3)."""
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        1,
    ).reshape(-1, 3, 3)


class Model:
    KEYS = ("means", "quats", "log_scales", "logit_op", "f_dc")

    def __init__(self, s, dev):
        self.p = {
            "means": torch.from_numpy(s["means"]).to(dev),
            "quats": torch.from_numpy(s["quats"]).to(dev),
            "log_scales": torch.from_numpy(s["log_scales"]).to(dev),
            "logit_op": torch.from_numpy(s["logit_opacities"]).to(dev),
            "f_dc": torch.from_numpy(s["f_dc"]).to(dev),
        }
        self.init = {k: v.clone() for k, v in self.p.items()}

    @property
    def n(self):
        return len(self.p["means"])

    means = property(lambda self: self.p["means"])
    quats = property(lambda self: self.p["quats"])
    log_scales = property(lambda self: self.p["log_scales"])
    logit_op = property(lambda self: self.p["logit_op"])
    f_dc = property(lambda self: self.p["f_dc"])

    def params(self):
        return self.p

    def render(self, viewmat, K, W, H, depth=False):
        colors = 0.5 + SH_C0 * self.p["f_dc"]
        out, alpha, info = rasterization(
            means=self.p["means"],
            quats=self.p["quats"],
            scales=torch.exp(self.p["log_scales"]),
            opacities=torch.sigmoid(self.p["logit_op"]),
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=W,
            height=H,
            near_plane=0.02,
            far_plane=200.0,
            eps2d=0.3,
            packed=True,
            render_mode="RGB+ED" if depth else "RGB",
            rasterize_mode="classic",
        )
        if depth:
            return out[0, ..., :3], alpha[0], out[0, ..., 3], info
        return out[0], alpha[0], None, info


def project(means, v, K, W, H, margin=0.05):
    """-> (inside frustum mask, z, u_int, v_int) for one view."""
    P = means @ v[:3, :3].T + v[:3, 3]
    z = P[:, 2]
    zc = z.clamp(min=1e-3)
    u = K[0, 0] * P[:, 0] / zc + K[0, 2]
    vv = K[1, 1] * P[:, 1] / zc + K[1, 2]
    ok = (
        (z > 0.02)
        & (u > -margin * W)
        & (u < (1 + margin) * W)
        & (vv > -margin * H)
        & (vv < (1 + margin) * H)
    )
    ui = u.clamp(0, W - 1).long()
    vi = vv.clamp(0, H - 1).long()
    return ok, z, ui, vi


def visible_mask(model, viewmats, K, W, H, dev, margin=0.05):
    """Splats that project inside any training frustum in front of the camera."""
    vis = torch.zeros(model.n, dtype=torch.bool, device=dev)
    for v in viewmats:
        ok, _, _, _ = project(model.means, v, K, W, H, margin)
        vis |= ok
    return vis


def observed_mask(means, viewmats, K, W, H, depths, valids, tol, min_views):
    """Observed = the training views carry evidence about this splat: it projects inside the image on a
    non-person pixel with a supported wvp depth, and passes the depth test there (z <= depth * (1+tol),
    i.e. it is the front-most surface along that ray, not hidden behind one), in >= min_views views."""
    cnt = torch.zeros(len(means), dtype=torch.int16, device=means.device)
    for v, gd, sup, val in zip(viewmats, *depths, valids):
        ok, z, ui, vi = project(means, v, K, W, H, margin=0.0)
        g = gd[vi, ui]
        hit = ok & sup[vi, ui] & val[vi, ui] & (z < g * (1 + tol))
        cnt += hit.to(torch.int16)
    return cnt >= min_views, cnt


def depth_offset(means, viewmats, K, W, H, depths, valids):
    """Median signed (splat z - wvp depth) over the sampled views that see each splat; NaN if none."""
    n = len(means)
    err = torch.full((n, len(viewmats)), float("nan"), device=means.device)
    for j, (v, gd, sup, val) in enumerate(zip(viewmats, *depths, valids)):
        ok, z, ui, vi = project(means, v, K, W, H, margin=0.0)
        g = gd[vi, ui]
        sel = ok & sup[vi, ui] & val[vi, ui]
        err[:, j] = torch.where(sel, z - g, torch.full_like(z, float("nan")))
    return torch.nanmedian(err, dim=1).values


def psnr(a, b, valid):
    d = ((a - b) ** 2)[
        valid.expand_as(a) if valid.dim() == a.dim() else valid[..., None].expand_as(a)
    ]
    return float(10 * torch.log10(1.0 / d.mean().clamp(min=1e-10)))


def to_u8(t):
    return (t.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


def surgery(model, opt, keep, extra=None, side=None):
    """Drop rows where ~keep, append `extra` rows; carries Adam state and the side tensors along."""
    for g in opt.param_groups:
        k = g["name"]
        p = g["params"][0]
        st = opt.state.pop(p, None)
        new = p.detach()[keep]
        if extra is not None:
            new = torch.cat([new, extra[k]], 0)
        new = new.contiguous().requires_grad_(True)
        g["params"] = [new]
        model.p[k] = new
        if st is not None:
            ea, eas = st["exp_avg"][keep], st["exp_avg_sq"][keep]
            if extra is not None:
                z = torch.zeros((len(extra[k]),) + ea.shape[1:], device=ea.device, dtype=ea.dtype)
                ea = torch.cat([ea, z], 0)
                eas = torch.cat([eas, z], 0)
            opt.state[new] = {
                "step": st["step"],
                "exp_avg": ea.contiguous(),
                "exp_avg_sq": eas.contiguous(),
            }
    for k in model.init:
        model.init[k] = (
            torch.cat([model.init[k][keep], extra[k]], 0)
            if extra is not None
            else model.init[k][keep]
        )
    out = []
    for t, fill in side or []:
        t = t[keep]
        if extra is not None:
            t = torch.cat(
                [t, torch.full((len(extra["means"]),), fill, dtype=t.dtype, device=t.device)], 0
            )
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--holdout", type=int, default=25)
    ap.add_argument(
        "--train-stride", type=int, default=2, help="use every Nth non-held-out frame for training"
    )
    ap.add_argument("--lambda-ssim", type=float, default=0.2)
    ap.add_argument(
        "--sigma-pos",
        type=float,
        default=0.02,
        help="tether scale for means, native units (0.02 = ~5 cm)",
    )
    ap.add_argument(
        "--clamp-pos",
        type=float,
        default=0.05,
        help="hard clamp on displacement, native units (~13 cm)",
    )
    ap.add_argument("--sigma-scale", type=float, default=0.5, help="tether on log-scale change")
    ap.add_argument(
        "--lambda-reg",
        type=float,
        default=1e-3,
        help="weight of the tether (sum over visible splats / n_visible * lambda * 1e3)",
    )
    ap.add_argument("--lr-means", type=float, default=1.6e-4)
    ap.add_argument("--lr-quats", type=float, default=1e-3)
    ap.add_argument("--lr-scales", type=float, default=5e-3)
    ap.add_argument("--lr-op", type=float, default=2.5e-2)
    ap.add_argument("--lr-color", type=float, default=2.5e-3)
    ap.add_argument(
        "--lambda-depth",
        type=float,
        default=0.0,
        help="weight of |log(rendered depth) - log(wvp depth)| on supported, non-person texels",
    )
    ap.add_argument(
        "--freeze-geometry", action="store_true", help="optimise colour and opacity only"
    )
    # run5: observed mask / prune / densify / opacity entropy
    ap.add_argument(
        "--observed-min-views",
        type=int,
        default=5,
        help="views a splat must pass the depth test in to count as observed",
    )
    ap.add_argument(
        "--observed-depth-tol",
        type=float,
        default=0.05,
        help="relative slack on the depth test (z <= wvp depth * (1+tol))",
    )
    ap.add_argument(
        "--reg-observed-scale",
        type=float,
        default=1.0,
        help="tether weight multiplier on observed splats (run5: 0.05)",
    )
    ap.add_argument(
        "--reg-unobserved-scale",
        type=float,
        default=1.0,
        help="tether weight multiplier on visible-but-never-front-most splats (run7: 0.3); these carry the "
        "off-path smear, so the depth-TV prior needs some slack to reshape them",
    )
    ap.add_argument(
        "--lambda-depth-tv",
        type=float,
        default=0.0,
        help="edge-aware total variation on the rendered log depth of random off-path views (run7). No "
        "photometric term there: the footage says nothing about those pixels, only that a corridor "
        "surface should be piecewise smooth where the render has no colour edge.",
    )
    ap.add_argument(
        "--tv-every",
        type=int,
        default=2,
        help="apply the off-path depth-TV term every Nth iteration",
    )
    ap.add_argument(
        "--tv-max-m", type=float, default=1.5, help="max lateral offset of the TV views, metres"
    )
    ap.add_argument(
        "--tv-edge",
        type=float,
        default=10.0,
        help="edge weight exp(-tv_edge * |grad rendered RGB|)",
    )
    ap.add_argument(
        "--prune",
        action="store_true",
        help="prune observed splats (transparent, then veil) during training",
    )
    ap.add_argument("--prune-every", type=int, default=200)
    ap.add_argument("--prune-start", type=int, default=400)
    ap.add_argument("--prune-opacity", type=float, default=0.05)
    ap.add_argument("--prune-depth-start", type=int, default=1500)
    ap.add_argument(
        "--prune-depth-cm",
        type=float,
        default=8.0,
        help="prune observed splats whose median depth sits this far in front of the wvp surface",
    )
    ap.add_argument(
        "--depth-views",
        type=int,
        default=48,
        help="training views sampled for the per-splat depth statistic",
    )
    ap.add_argument(
        "--densify", action="store_true", help="3DGS split/clone restricted to observed splats"
    )
    ap.add_argument("--densify-start", type=int, default=500)
    ap.add_argument("--densify-stop-frac", type=float, default=0.75)
    ap.add_argument("--grow-grad2d", type=float, default=2e-4)
    ap.add_argument(
        "--grow-scale3d",
        type=float,
        default=0.01,
        help="native units; below this a selected splat is cloned, above it is split",
    )
    ap.add_argument("--max-splats", type=int, default=2_400_000)
    ap.add_argument(
        "--lambda-opacity-entropy", type=float, default=0.0, help="push observed opacities to 0/1"
    )
    ap.add_argument(
        "--pseudo-dir",
        type=Path,
        default=None,
        help="extra pseudo-GT views (npz with viewmats + PNGs) mixed in at --pseudo-weight",
    )
    ap.add_argument("--pseudo-weight", type=float, default=0.3)
    ap.add_argument(
        "--init-npz",
        type=Path,
        default=None,
        help="start from a finetuned.npz instead of the Marble splats.npz",
    )
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poses", default="0,275,549", help="frame indices rendered for the figures")
    ap.add_argument("--offpath-frame", type=int, default=275)
    ap.add_argument(
        "--m-per-unit", type=float, default=1.7 / 0.6518811, help="metres per native unit"
    )
    a = ap.parse_args()
    globals()["CM_PER_UNIT"] = a.m_per_unit * 100
    fig_poses = [int(x) for x in a.poses.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda")
    t0 = time.time()

    s = np.load(a.data / "splats.npz")
    c = np.load(a.data / "cameras.npz")
    model = Model(np.load(a.init_npz) if a.init_npz else s, dev)
    viewmats = torch.from_numpy(c["viewmats"]).to(dev)
    K = torch.from_numpy(c["K"]).to(dev)
    W, H = int(c["width"]), int(c["height"])
    nF = len(viewmats)
    imgs, valid = load_images(a.data, nF, dev)
    dz = np.load(a.data / "depth.npz")
    depth_layers = torch.from_numpy(dz["depth"]).to(dev)
    depth_sup = torch.from_numpy(dz["supported"]).to(dev)
    depth_index = dz["depth_index"]

    def gt_depth(i):
        L = int(depth_index[i])
        d = F.interpolate(depth_layers[L][None, None], size=(H, W), mode="nearest")[0, 0]
        sup = (
            F.interpolate(depth_sup[L][None, None].float(), size=(H, W), mode="nearest")[0, 0] > 0.5
        )
        return d, sup

    test_idx = list(range(0, nF, a.holdout))
    train_idx = [i for i in range(nF) if i % a.holdout != 0][:: a.train_stride]
    print(
        f"{model.n} splats, {nF} frames ({len(train_idx)} train / {len(test_idx)} test) at {W}x{H}; load {time.time() - t0:.0f}s",
        flush=True,
    )

    vis = visible_mask(model, viewmats[train_idx], K, W, H, dev)
    n_vis = int(vis.sum())
    print(f"visible in >=1 training view: {n_vis} ({100 * n_vis / model.n:.1f}%)", flush=True)

    use_obs = a.prune or a.densify or a.reg_observed_scale != 1.0 or a.lambda_opacity_entropy > 0
    if use_obs:
        with torch.no_grad():
            gds = [gt_depth(i) for i in train_idx]
            depths = ([g[0] for g in gds], [g[1] for g in gds])
            obs, hit_cnt = observed_mask(
                model.means,
                viewmats[train_idx],
                K,
                W,
                H,
                depths,
                [valid[i] for i in train_idx],
                a.observed_depth_tol,
                a.observed_min_views,
            )
            del gds, depths
            torch.cuda.empty_cache()
        n_obs = int(obs.sum())
        print(
            f"observed (depth test in >= {a.observed_min_views} of {len(train_idx)} views): {n_obs} "
            f"({100 * n_obs / model.n:.1f}%); median hits among visible {float(hit_cnt[vis].float().median()):.0f}",
            flush=True,
        )
        dv_idx = train_idx[:: max(1, len(train_idx) // a.depth_views)][: a.depth_views]
        with torch.no_grad():
            gds = [gt_depth(i) for i in dv_idx]
            dv_depths = ([g[0] for g in gds], [g[1] for g in gds])
            dv_valid = [valid[i] for i in dv_idx]
            dv_views = viewmats[dv_idx]
        del hit_cnt
    else:
        obs = torch.zeros(model.n, dtype=torch.bool, device=dev)

    orig_idx = torch.arange(model.n, dtype=torch.int32, device=dev)
    bbox_lo = model.init["means"].min(0).values
    bbox_hi = model.init["means"].max(0).values

    def evaluate(tag):
        model_psnr = []
        derr = []
        with torch.no_grad():
            for i in test_idx:
                img, al, dep, _ = model.render(viewmats[i], K, W, H, depth=True)
                gt = imgs[i].float() / 255
                model_psnr.append(psnr(img, gt, valid[i]))
                gd, sup = gt_depth(i)
                sel = sup & valid[i] & (al[..., 0] > 0.5)
                derr.append(
                    float((torch.log(dep.clamp(min=0.05)) - torch.log(gd))[sel].abs().median())
                )
        m = float(np.mean(model_psnr))
        dr = float(np.mean(derr))
        print(
            f"[{tag}] held-out masked PSNR: mean {m:.2f} dB (min {min(model_psnr):.2f}, max {max(model_psnr):.2f}); median |log depth ratio| vs wvp {dr:.3f}",
            flush=True,
        )
        return m, model_psnr, dr

    m_per_unit = a.m_per_unit

    def offpath_view(frame, d_m, axis=0, sign=1.0):
        c2w = torch.linalg.inv(viewmats[frame]).clone()
        c2w[:3, 3] += c2w[:3, axis] * (sign * d_m / m_per_unit)
        return torch.linalg.inv(c2w)

    def render_set(tag):
        with torch.no_grad():
            for i in fig_poses:
                img, _, _, _ = model.render(viewmats[i], K, W, H)
                Image.fromarray(to_u8(img)).save(a.out / f"{tag}-pose-{i}.png")
            for d in (0.5, 1.5):
                img, _, _, _ = model.render(offpath_view(a.offpath_frame, d), K, W, H)
                Image.fromarray(to_u8(img)).save(a.out / f"{tag}-offpath-{d}m.png")

    before, before_list, dr_before = evaluate("before")
    render_set("before")
    arr = np.asarray(Image.open(a.out / "before-pose-0.png"))
    print(f"before-pose-0 mean intensity {arr.mean():.1f} (black render would be ~0)", flush=True)
    if arr.mean() < 2:
        raise SystemExit("render is black: check camera convention / near plane")

    # optional pseudo-GT views (Difix3D+ style): pseudo.npz with viewmats (N,4,4) + pseudo/%04d.png
    pseudo = None
    if a.pseudo_dir:
        pz = np.load(a.pseudo_dir / "pseudo.npz")
        pv = torch.from_numpy(pz["viewmats"]).to(dev)
        pi = torch.from_numpy(
            np.stack(
                [
                    np.asarray(Image.open(a.pseudo_dir / f"{i:04d}.png").convert("RGB"))
                    for i in range(len(pv))
                ]
            )
        ).to(dev)
        pseudo = (pv, pi)
        print(f"pseudo-GT views: {len(pv)} at weight {a.pseudo_weight}", flush=True)

    for p in model.params().values():
        p.requires_grad_(True)
    opt = torch.optim.Adam(
        [
            {"params": [model.p["means"]], "lr": a.lr_means, "name": "means"},
            {"params": [model.p["quats"]], "lr": a.lr_quats, "name": "quats"},
            {"params": [model.p["log_scales"]], "lr": a.lr_scales, "name": "log_scales"},
            {"params": [model.p["logit_op"]], "lr": a.lr_op, "name": "logit_op"},
            {"params": [model.p["f_dc"]], "lr": a.lr_color, "name": "f_dc"},
        ],
        eps=1e-15,
    )
    win = gaussian_window(dev=dev)
    log = []
    n_pruned_op = n_pruned_depth = n_cloned = n_split = 0
    grad_acc = torch.zeros(model.n, device=dev)
    grad_cnt = torch.zeros(model.n, device=dev)

    def reg_weights():
        return vis.float() * torch.where(
            obs,
            torch.full((), a.reg_observed_scale, device=dev),
            torch.full((), a.reg_unobserved_scale, device=dev),
        )

    w_reg = reg_weights()
    order = np.random.permutation(train_idx)
    ptr = 0
    for it in range(1, a.iters + 1):
        use_pseudo = pseudo is not None and (it % 3 == 0)
        if use_pseudo:
            j = int(np.random.randint(len(pseudo[0])))
            vm = pseudo[0][j]
            gt = pseudo[1][j].float() / 255
            m = torch.ones((H, W, 1), device=dev)
            wgt = a.pseudo_weight
        else:
            if ptr >= len(order):
                order = np.random.permutation(train_idx)
                ptr = 0
            i = int(order[ptr])
            ptr += 1
            vm = viewmats[i]
            gt = imgs[i].float() / 255
            m = valid[i].float()[..., None]
            wgt = 1.0
        img, alpha, dep, info = model.render(vm, K, W, H, depth=True)
        if a.densify and it >= a.densify_start:
            info["means2d"].retain_grad()
        l1 = ((img - gt).abs() * m).sum() / (m.sum() * 3)
        sm = ssim_map(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], win)
        ss = (sm * m.permute(2, 0, 1)[None]).sum() / (m.sum() * 3)
        loss = wgt * ((1 - a.lambda_ssim) * l1 + a.lambda_ssim * (1 - ss))
        if a.lambda_depth > 0 and not use_pseudo:
            gd, sup = gt_depth(i)
            md = (sup & valid[i]).float()
            dl = (
                (torch.log(dep.clamp(min=0.05)) - torch.log(gd)).abs() * md
            ).sum() / md.sum().clamp(min=1)
            loss = loss + a.lambda_depth * dl
        else:
            dl = torch.zeros(())
        tv = torch.zeros((), device=dev)
        if a.lambda_depth_tv > 0 and it % a.tv_every == 0:
            fr = int(np.random.choice(train_idx))
            off_m = float(np.random.uniform(0.25, a.tv_max_m)) * (
                1.0 if np.random.rand() < 0.5 else -1.0
            )
            im2, al2, dp2, _ = model.render(offpath_view(fr, off_m), K, W, H, depth=True)
            ld = torch.log(dp2.clamp(min=0.05))
            mk = (al2[..., 0] > 0.5).float()
            ix = (im2[:, 1:] - im2[:, :-1]).abs().mean(-1)
            iy = (im2[1:] - im2[:-1]).abs().mean(-1)
            wx = torch.exp(-a.tv_edge * ix.detach()) * mk[:, 1:] * mk[:, :-1]
            wy = torch.exp(-a.tv_edge * iy.detach()) * mk[1:] * mk[:-1]
            tv = ((ld[:, 1:] - ld[:, :-1]).abs() * wx).sum() / wx.sum().clamp(min=1) + (
                (ld[1:] - ld[:-1]).abs() * wy
            ).sum() / wy.sum().clamp(min=1)
            loss = loss + a.lambda_depth_tv * tv
        dpos = ((model.means - model.init["means"]) / a.sigma_pos) ** 2
        dsc = ((model.log_scales - model.init["log_scales"]) / a.sigma_scale) ** 2
        reg = a.lambda_reg * ((dpos.sum(1) + dsc.sum(1)) * w_reg).sum() / n_vis * 1e3
        if a.lambda_opacity_entropy > 0 and int(obs.sum()) > 0:
            o = torch.sigmoid(model.logit_op[obs]).clamp(1e-6, 1 - 1e-6)
            reg = (
                reg
                + a.lambda_opacity_entropy
                * (-(o * torch.log(o) + (1 - o) * torch.log(1 - o))).mean()
            )
        (loss + reg).backward()
        gate = vis.float()
        for k, p in model.params().items():
            if p.grad is None:
                continue
            g = (
                torch.zeros_like(gate)
                if (a.freeze_geometry and k in ("means", "quats", "log_scales"))
                else gate
            )
            p.grad.mul_(g[:, None] if p.grad.dim() == 2 else g)
        if a.densify and it >= a.densify_start and info["means2d"].grad is not None:
            with torch.no_grad():
                g2 = info["means2d"].grad.detach().clone()
                g2[..., 0] *= W / 2.0
                g2[..., 1] *= H / 2.0
                gn = g2.norm(dim=-1).reshape(-1)
                gid = info["gaussian_ids"].reshape(-1)
                grad_acc.index_add_(0, gid, gn)
                grad_cnt.index_add_(0, gid, torch.ones_like(gn))
        # means lr decays 100x over the run, like gsplat's simple_trainer
        opt.param_groups[0]["lr"] = a.lr_means * (0.01 ** (it / a.iters))
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            d = model.means - model.init["means"]
            nrm = d.norm(dim=1, keepdim=True)
            model.means.copy_(model.init["means"] + d * (a.clamp_pos / nrm.clamp(min=a.clamp_pos)))
            model.quats.copy_(model.quats / model.quats.norm(dim=1, keepdim=True).clamp(min=1e-8))

        do_prune = a.prune and it >= a.prune_start
        do_grow = a.densify and a.densify_start <= it <= a.densify_stop_frac * a.iters
        if (do_prune or do_grow) and it % a.prune_every == 0:
            with torch.no_grad():
                # --- grow (observed only) ---
                extra = None
                keep = torch.ones(model.n, dtype=torch.bool, device=dev)
                if do_grow:
                    avg = grad_acc / grad_cnt.clamp(min=1)
                    sel = obs & (grad_cnt > 0) & (avg > a.grow_grad2d)
                    room = a.max_splats - model.n
                    if int(sel.sum()) > 0 and room > 0:
                        if int(sel.sum()) > room:  # keep the highest-gradient candidates
                            idx = torch.nonzero(sel).squeeze(1)
                            top = idx[torch.argsort(avg[idx], descending=True)[:room]]
                            sel = torch.zeros_like(sel)
                            sel[top] = True
                        big = torch.exp(model.log_scales).max(dim=1).values > a.grow_scale3d
                        sp, cl = sel & big, sel & ~big
                        n_split += int(sp.sum())
                        n_cloned += int(cl.sum())
                        rot = quat_to_rotmat(model.quats[sp])
                        sc = torch.exp(model.log_scales[sp])
                        # 2-sigma cap: Marble has a few metre-wide background blobs, and an unclipped
                        # normal sample throws their children right out of the scene
                        off = torch.einsum(
                            "nij,nj->ni", rot, torch.randn_like(sc).clamp(-2, 2) * sc
                        )
                        extra = {
                            "means": torch.cat(
                                [model.means[sp] + off, model.means[sp] - off, model.means[cl]], 0
                            ),
                            "quats": torch.cat(
                                [model.quats[sp], model.quats[sp], model.quats[cl]], 0
                            ),
                            "log_scales": torch.cat(
                                [
                                    model.log_scales[sp] - math.log(1.6),
                                    model.log_scales[sp] - math.log(1.6),
                                    model.log_scales[cl],
                                ],
                                0,
                            ),
                            "logit_op": torch.cat(
                                [model.logit_op[sp], model.logit_op[sp], model.logit_op[cl]], 0
                            ),
                            "f_dc": torch.cat([model.f_dc[sp], model.f_dc[sp], model.f_dc[cl]], 0),
                        }
                        # split children of the few metre-wide Marble blobs can chain outwards over
                        # successive refines; keep every new splat inside the original scene box
                        extra["means"] = torch.max(torch.min(extra["means"], bbox_hi), bbox_lo)
                        keep &= ~sp
                # --- prune (observed only) ---
                if do_prune:
                    dead = obs & (torch.sigmoid(model.logit_op) < a.prune_opacity)
                    n_pruned_op += int((dead & keep).sum())
                    keep &= ~dead
                    if it >= a.prune_depth_start:
                        off_d = depth_offset(model.means, dv_views, K, W, H, dv_depths, dv_valid)
                        veil = obs & torch.nan_to_num(off_d, nan=0.0).lt(
                            -a.prune_depth_cm / CM_PER_UNIT
                        )
                        n_pruned_depth += int((veil & keep).sum())
                        keep &= ~veil
                if extra is not None or not bool(keep.all()):
                    (vis, obs, orig_idx) = surgery(
                        model, opt, keep, extra, side=[(vis, True), (obs, True), (orig_idx, -1)]
                    )
                    n_vis = max(1, int(vis.sum()))
                    w_reg = reg_weights()
                    grad_acc = torch.zeros(model.n, device=dev)
                    grad_cnt = torch.zeros(model.n, device=dev)
                    print(
                        f"  it {it}: refine -> {model.n} splats (pruned op {n_pruned_op}, depth {n_pruned_depth}; "
                        f"cloned {n_cloned}, split {n_split})",
                        flush=True,
                    )
                else:
                    grad_acc.zero_()
                    grad_cnt.zero_()

        if it % 50 == 0 or it == 1:
            rec = {
                "it": it,
                "l1": float(l1.detach()),
                "ssim": float(ss.detach()),
                "depth": float(dl.detach()),
                "tv": float(tv.detach()),
                "reg": float(reg.detach()),
                "n": model.n,
                "psnr_train": psnr(img.detach(), gt, m[..., 0] > 0.5),
                "t": time.time() - t0,
            }
            log.append(rec)
            print(
                f"it {it}: l1 {rec['l1']:.4f} ssim {rec['ssim']:.3f} depth {rec['depth']:.4f} tv {rec['tv']:.4f} reg {rec['reg']:.4f} "
                f"train psnr {rec['psnr_train']:.2f} n {model.n} | {rec['t']:.0f}s VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
                flush=True,
            )
        if it % a.eval_every == 0 and it < a.iters:
            evaluate(f"it {it}")
            render_set(f"it{it}")

    after, after_list, dr_after = evaluate("after")
    render_set("after")
    with torch.no_grad():
        alive = orig_idx >= 0
        d = (model.means - model.init["means"]).norm(dim=1)[vis]
        ds = (model.log_scales - model.init["log_scales"]).abs().max(1).values[vis]
        op0 = torch.sigmoid(model.init["logit_op"])[vis]
        op1 = torch.sigmoid(model.logit_op)[vis]
        stats = {
            "psnr_before": before,
            "psnr_after": after,
            "psnr_before_list": before_list,
            "psnr_after_list": after_list,
            "depth_ratio_before": dr_before,
            "depth_ratio_after": dr_after,
            "n_splats": model.n,
            "n_splats_init": int(len(orig_idx)),
            "n_visible": n_vis,
            "n_observed": int(obs.sum()),
            "n_new": int((~alive).sum()),
            "n_pruned_opacity": n_pruned_op,
            "n_pruned_depth": n_pruned_depth,
            "n_cloned": n_cloned,
            "n_split": n_split,
            "disp_cm_unobserved_p90": float(
                (model.means - model.init["means"]).norm(dim=1)[vis & ~obs].quantile(0.9)
            )
            * CM_PER_UNIT
            if int((vis & ~obs).sum())
            else None,
            "disp_native_median": float(d.median()),
            "disp_native_p90": float(d.quantile(0.9)),
            "disp_native_max": float(d.max()),
            "disp_cm_median": float(d.median()) * CM_PER_UNIT,
            "disp_cm_p90": float(d.quantile(0.9)) * CM_PER_UNIT,
            "logscale_change_p90": float(ds.quantile(0.9)),
            "opacity_dropped_below_0.05": int(((op0 > 0.05) & (op1 < 0.05)).sum()),
            "opacity_raised_above_0.5_from_below_0.1": int(((op0 < 0.1) & (op1 > 0.5)).sum()),
            "iters": a.iters,
            "args": vars(a)
            | {k: str(getattr(a, k)) for k in ("data", "out", "pseudo_dir", "init_npz")},
            "log": log,
            "runtime_s": time.time() - t0,
        }
    json.dump(stats, open(a.out / "stats.json", "w"), indent=1)
    np.savez(
        a.out / "finetuned.npz",
        means=model.means.detach().cpu().numpy(),
        quats=model.quats.detach().cpu().numpy(),
        log_scales=model.log_scales.detach().cpu().numpy(),
        logit_opacities=model.logit_op.detach().cpu().numpy(),
        f_dc=model.f_dc.detach().cpu().numpy(),
        visible=vis.cpu().numpy(),
        orig_index=orig_idx.cpu().numpy(),
        scale0=s["scale0"],
        origin_c2w=s["origin_c2w"],
        R_raw_to_native=s["R_raw_to_native"],
        fb=s["fb"],
    )
    print(
        f"PSNR before {before:.2f} -> after {after:.2f} dB; depth ratio {dr_before:.3f} -> {dr_after:.3f}; "
        f"{len(orig_idx)} -> {model.n} splats (pruned {n_pruned_op + n_pruned_depth}, added {stats['n_new']}); "
        f"displacement median {stats['disp_cm_median']:.1f} cm p90 {stats['disp_cm_p90']:.1f} cm; "
        f"opacity killed {stats['opacity_dropped_below_0.05']}, new {stats['opacity_raised_above_0.5_from_below_0.1']}; {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
