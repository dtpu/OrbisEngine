#!/usr/bin/env python
"""Bring a fine-tuned gsplat state (finetune_gsplat.py finetuned.npz, SfM native frame) back into
raw Marble coordinates and write it as .spz (v2, sh degree 0, same layout as the Marble file) and
as a gaussian .ply. Inverse of finetune_export.py:  raw = scale0 * R^T (native - t[origin]).

    python scripts/finetune_import.py --npz run1/finetuned.npz --out public/marble-corridor-finetuned
"""
import argparse
import gzip
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_video_colours import ROOT, SH_C0, SPZ_COLOR_SCALE, read_spz  # noqa: E402
from finetune_export import quat_mul, rotmat_to_quat_xyzw  # noqa: E402


def write_spz(out: Path, xyz, alpha, f_dc, log_scale, quat_xyzw, fb: int, header_template: bytes):
    n = len(xyz)
    magic, ver, _, sh, _, flags, res = struct.unpack('<IIIBBBB', header_template)
    header = struct.pack('<IIIBBBB', magic, ver, n, sh, fb, flags, res)
    fixed = np.clip(np.round(xyz * float(1 << fb)), -(1 << 23), (1 << 23) - 1).astype(np.int32)
    fixed = np.where(fixed < 0, fixed + (1 << 24), fixed).astype(np.uint32)
    pos_b = np.stack([fixed & 0xFF, (fixed >> 8) & 0xFF, (fixed >> 16) & 0xFF], 2).astype(np.uint8)   # (n,3,3)
    alpha_b = np.clip(np.round(alpha * 255), 0, 255).astype(np.uint8)
    col_b = np.clip(np.round(f_dc * (SPZ_COLOR_SCALE * 255.0) + 127.5), 0, 255).astype(np.uint8)
    scale_b = np.clip(np.round((log_scale + 10.0) * 16.0), 0, 255).astype(np.uint8)
    q = quat_xyzw / np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    q = np.where(q[:, 3:4] < 0, -q, q)
    rot_b = np.clip(np.round(q[:, :3] * 127.5 + 127.5), 0, 255).astype(np.uint8)
    body = header + pos_b.tobytes() + alpha_b.tobytes() + col_b.tobytes() + scale_b.tobytes() + rot_b.tobytes()
    with gzip.open(out, 'wb', compresslevel=6) as f:
        f.write(body)


def write_ply(out: Path, xyz, alpha, f_dc, log_scale, quat_xyzw):
    n = len(xyz)
    a = np.clip(alpha, 1e-4, 1 - 1e-4)
    rows = np.zeros((n, 17), np.float32)
    rows[:, 0:3] = xyz; rows[:, 6:9] = f_dc; rows[:, 9] = np.log(a / (1 - a))
    rows[:, 10:13] = log_scale; rows[:, 13] = quat_xyzw[:, 3]; rows[:, 14:17] = quat_xyzw[:, :3]
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    header = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n + ''.join('property float %s\n' % s for s in names) + 'end_header\n'
    with open(out, 'wb') as f:
        f.write(header.encode()); f.write(rows.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--world', default=ROOT / 'public/marble-corridor-recon-f0-v2.spz', type=Path, help='original spz (header template + round-trip check)')
    ap.add_argument('--out', type=Path, required=True, help='output stem')
    a = ap.parse_args()
    s = np.load(a.npz)
    R = s['R_raw_to_native']; t = s['origin_c2w'][:3, 3]; scale0 = float(s['scale0']); fb = int(s['fb'])
    xyz = ((s['means'].astype(np.float64) - t) @ R) * scale0                     # (R^T applied via right-multiplication)
    q_native = np.concatenate([s['quats'][:, 1:4], s['quats'][:, 0:1]], 1).astype(np.float64)   # wxyz -> xyzw
    q_inv = rotmat_to_quat_xyzw(R.T)[None].repeat(len(xyz), 0)
    quat = quat_mul(q_inv, q_native)
    log_scale = s['log_scales'].astype(np.float64) + np.log(scale0)
    alpha = 1 / (1 + np.exp(-s['logit_opacities'].astype(np.float64)))
    f_dc = s['f_dc'].astype(np.float64)

    spz = read_spz(a.world)
    vis = s['visible'] if 'visible' in s else np.zeros(len(xyz), bool)
    # runs that prune/densify carry orig_index (-1 = splat created during training); the round-trip
    # check only makes sense on surviving splats that were never optimised.
    oi = s['orig_index'] if 'orig_index' in s else np.arange(len(xyz))
    frozen = (~vis) & (oi >= 0) & (oi < spz['n']); src = oi[frozen]  # oi indexes --init-npz, not Marble
    print(f'round trip on frozen splats: max |xyz - original| {np.abs(xyz[frozen] - spz["xyz"][src]).max():.2e}, '
          f'max |log_scale - original| {np.abs(log_scale[frozen] - spz["log_scale"][src]).max():.2e}, '
          f'max |quat.quat - 1| {np.abs(np.abs((quat[frozen] * spz["quat"][src]).sum(1)) - 1).max():.2e}', flush=True)
    print(f'{len(xyz)} splats (original {spz["n"]}: {(oi < 0).sum()} created, {spz["n"] - (oi >= 0).sum()} pruned); '
          f'moved {vis.sum()}; bbox raw {xyz.min(0).round(2)} .. {xyz.max(0).round(2)} (original {spz["xyz"].min(0).round(2)} .. {spz["xyz"].max(0).round(2)})', flush=True)
    write_spz(a.out.with_suffix('.spz'), xyz, alpha, f_dc, log_scale, quat, fb, spz['header'])
    write_ply(a.out.with_suffix('.ply'), xyz.astype(np.float32), alpha.astype(np.float32), f_dc.astype(np.float32), log_scale.astype(np.float32), quat.astype(np.float32))
    chk = read_spz(a.out.with_suffix('.spz'))
    print(f'wrote {a.out.with_suffix(".spz")} ({a.out.with_suffix(".spz").stat().st_size / 1e6:.1f} MB, re-read {chk["n"]} splats, max pos err {np.abs(chk["xyz"] - xyz).max():.2e}), '
          f'{a.out.with_suffix(".ply")} ({a.out.with_suffix(".ply").stat().st_size / 1e6:.0f} MB)', flush=True)


if __name__ == '__main__':
    main()
