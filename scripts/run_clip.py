#!/usr/bin/env python3
"""One command from a source clip to a viewer-ready 4D preset.

Runs the stage sequence from share/SECOND-CLIP-STATUS.md and share/OWN-CLIPS-STATUS.md, but
concurrently: everything that does not depend on the rest starts at once. The GPU clean pass, the
Pi3X reconstruction and the LHM avatar chain all run in parallel, and Marble is the critical path
after them. (--marble image inpaints frame 0 on its own instead, which unblocks Marble in ~83 s.)

  WLT_API_KEY=... python3 scripts/run_clip.py --clip public/clips/bedroom.mp4 --name bedroom \
      --fps 12 --dilate 40 --bottom-extra 80 --person-frame 150

ONE Marble world per run (--marble video, the default; 1600 credits each), and it is not submitted
until a human has looked at the cleaned frames: the run stops at the `review` gate and prints their
paths. Pass --gate-pass on the next run to continue, or --no-gate to run unattended.

Before any of that, and before any spend, the clip is checked for CUTS: everything downstream
assumes one continuous shot, and a cut comes back as a camera teleport rather than an error. A
multi-shot clip is not processed as one -- the shots are scored and the best one is named and
trimmed (--shot N picks another, --all-shots runs them all). See scripts/shot_cuts.py.

Stages write into .context/run/<name>/ and record themselves in state.json, so a crashed or
killed run picks up where it stopped: re-run the same command. --force <stage>[,<stage>] redoes
one, --only <stage>[,...] runs a subset (with its finished dependencies read from state).
"""
from __future__ import annotations

import argparse, json, math, os, re, shutil, subprocess, sys, threading, time
from urllib.parse import quote
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# A clean clone uses its own toolchain and ignored output directories.
PY = os.environ.get("WANDER_PYTHON", sys.executable)
MODAL = os.environ.get("WANDER_MODAL", shutil.which("modal") or str(ROOT / "worker/.venv-da3/bin/modal"))
SHARE = Path(os.environ.get("WANDER_SHARE_DIR", ROOT / ".context/share")).expanduser().resolve()
CLIPS = Path(os.environ.get("WANDER_CLIPS_DIR", ROOT / ".context/clips")).expanduser().resolve()
MARBLE_DIR = Path(os.environ.get("WANDER_MARBLE_DIR", ROOT / ".context/marble")).expanduser().resolve()
LOCAL_ENV = {"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
             "LAMA_MODEL": os.path.expanduser("~/.cache/lama/big-lama.pt"),
             "MODAL_PROFILE": os.environ.get("MODAL_PROFILE", "dtpu")}

MARBLE_MODES = ("video", "image", "multi", "both", "none")
# scale_fit gates on each sampled frame's depth-ratio MEDIAN (0.95-1.05) and on its p10/p90: the median
# alone passed hpwide with p10 0.71 / p90 1.37 -- two modes, 0.7 and 1.15, from a 15 deg frame tilt,
# and no mass at 1.0 (share/ADVERSARIAL-LOG.md R11)
SCALE_RATIO_BAND = (0.95, 1.05)
SCALE_SPREAD_BAND = (0.90, 1.10)


def world_half(marble: str):
    """(stages, deps, world_stage) for the scene half.

    ONE generation by default. A marble-1.1 world is 1600 credits, so `video` submits exactly one
    and `both` (two worlds, 3200 credits) has to be asked for by name. `review` is the human gate
    that sits between the clean pass and the spend: the clean pass can leave a removed arm's
    dumbbells hanging in mid-air, and that is only visible to a person looking at the frames.
    """
    stages, deps = [], {}
    if marble in ("image", "both"):
        stages.append("clean_first"); deps["clean_first"] = []
    if marble in ("video", "both", "none"):
        stages.append("clean"); deps["clean"] = []
    if marble == "multi":
        stages.append("clean_multi"); deps["clean_multi"] = []
    if marble == "none":
        return stages, deps, None
    # The prompt is written from the clip BEFORE anything is generated, and it is what
    # disable_recaption then pins. Without it Marble captions the clip itself and generates from the
    # caption (share/HP-MARBLE-VERDICT.md).
    stages.append("world_prompt"); deps["world_prompt"] = []
    stages.append("review"); deps["review"] = list(stages[:-1])
    if marble in ("image", "both"):
        stages.append("marble_image"); deps["marble_image"] = ["clean_first", "review"]
    if marble in ("video", "both"):
        stages.append("marble_video"); deps["marble_video"] = ["clean", "review"]
    if marble == "multi":
        stages.append("marble_multi"); deps["marble_multi"] = ["clean_multi", "review"]
    world = {"multi": "marble_multi", "video": "marble_video", "both": "marble_video"}.get(marble, "marble_image")
    return stages, deps, world


def single_graph(marble: str, objects: bool = True):
    stages, deps, world = world_half(marble)
    stages += ["pi3x", "frame_align", "person_prep", "lhm_frozen", "lhm_motion", "package"]
    deps.update({"pi3x": [], "frame_align": ["pi3x"], "person_prep": [], "lhm_frozen": ["person_prep"],
                 "lhm_motion": ["lhm_frozen", "pi3x"], "package": ["lhm_motion", "frame_align"]})
    if objects:
        stages += ["objects"]
        deps["objects"] = ["package"]
    if world:
        stages += ["scale_fit", "place_fit", "anchors", "finetune", "verify"]
        deps["scale_fit"] = ["pi3x", "frame_align", "package", world]
        deps["place_fit"] = ["scale_fit"]
        deps["anchors"] = ["place_fit"]
        deps["finetune"] = ["scale_fit"]
        deps["verify"] = ["scale_fit"]
    return stages, deps


def multiperson_graph(limit: int, marble: str, objects: bool = True):
    """Stage list and dependencies for up to `limit` people.

    The real track count is only known once `tracks` has run, so the graph is built for `limit`
    slots and the slots no track claims mark themselves skipped. One person found => only slot 00
    runs and `package_people` writes the ordinary single-person layout, so a one-person clip comes
    out of this mode byte-identical to the single-person pipeline.
    """
    stages, deps, world = world_half(marble)
    stages += ["pi3x", "frame_align", "tracks"]
    deps.update({"pi3x": [], "frame_align": ["pi3x"], "tracks": ["pi3x"]})
    motions = []
    for i in range(limit):
        prep, frozen, motion = f"person_prep_{i:02d}", f"lhm_frozen_{i:02d}", f"lhm_motion_{i:02d}"
        stages += [prep, frozen, motion]
        deps[prep] = ["tracks"]
        deps[frozen] = [prep]
        deps[motion] = [frozen, "pi3x"]
        motions.append(motion)
    stages += ["package_people"]
    deps["package_people"] = motions + ["frame_align"]
    if objects:
        stages += ["objects"]
        deps["objects"] = ["package_people", "tracks"]
    if world:
        stages += ["scale_fit", "place_fit", "anchors", "finetune", "verify"]
        deps["scale_fit"] = ["pi3x", "frame_align", "package_people", world]
        deps["place_fit"] = ["scale_fit"]
        deps["anchors"] = ["place_fit"]
        deps["finetune"] = ["scale_fit"]
        deps["verify"] = ["scale_fit"]
    return stages, deps


def parse_object_spec(spec: str) -> dict:
    """`id:motion:prompt[:roi]` -> a dict for the objects stage.

    roi is `joint=TRACK,JOINT,RADIUS` (attached / handoff), `track2d` (free, from the object's own
    2D track) or `box=x0,y0,x1,y1` (worldDynamic). Omitted means the whole frame, which is right
    for something that fills a decent part of it and wrong for anything small -- see
    docs/objects.md §4.
    """
    parts = spec.split(":", 3)
    if len(parts) < 3:
        raise SystemExit(f"--object wants id:motion:prompt[:roi], got {spec!r}")
    oid, motion, prompt = parts[0], parts[1], parts[2]
    if motion not in ("attached", "free", "handoff", "worldDynamic"):
        raise SystemExit(f"--object motion must be attached|free|handoff|worldDynamic, got {motion!r}")
    return dict(id=oid, motion=motion, prompt=prompt, roi=parts[3] if len(parts) > 3 else "")


class SkipStage(Exception):
    """A stage with nothing to do (a person slot no track claimed); not a failure."""


class GateStop(Exception):
    """The clean-review gate is closed: a human has to look before any credits are spent."""

_print_lock = threading.Lock()


def say(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class State:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.data = json.loads(path.read_text()) if path.exists() else {"stages": {}}

    def done(self, stage):
        return self.data["stages"].get(stage, {}).get("status") == "ok"

    def record(self, stage, **kw):
        with self.lock:
            self.data["stages"].setdefault(stage, {}).update(kw)
            self.path.write_text(json.dumps(self.data, indent=2))


_modal_gate = threading.Lock()
_modal_last = [0.0]
MODAL_SPACING = 20.0  # Modal rate-limits app creation, and this pipeline starts several at once


def _space_modal_launch():
    with _modal_gate:
        wait = MODAL_SPACING - (time.time() - _modal_last[0])
        if wait > 0:
            time.sleep(wait)
        _modal_last[0] = time.time()


def run(cmd, log: Path, env_extra=None, cwd=ROOT, attempts=4, append=False):
    """Run a command, logging to `log`. `attempts` retries only on a Modal rate limit.

    A command that spends Marble credits can never be retried here, whatever the caller asks for:
    a `submit` that fails *after* the operation exists has already been paid for, and running it
    again buys a second 1600-credit world. Network retries belong around `poll` (see
    Pipeline.marble_poll), which takes an operation id and cannot create one.
    """
    if attempts > 1 and "submit" in cmd:
        raise RuntimeError(f"refusing to retry a Marble submit: {' '.join(cmd[:3])}")
    env = dict(os.environ, **LOCAL_ENV, **(env_extra or {}))
    log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        if cmd[0] == MODAL:
            _space_modal_launch()
        with log.open("a" if append else "w") as fh:
            fh.write(" ".join("<key>" if v.startswith("wlt") else v for v in cmd) + "\n\n")
            fh.flush()
            p = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if p.returncode == 0:
            return
        if "rate limit" in log.read_text().lower() and attempt < attempts - 1:
            say(f"   rate-limited, retrying {Path(cmd[2] if len(cmd) > 2 else cmd[0]).name} in 45s")
            time.sleep(45)
            continue
        raise RuntimeError(f"{cmd[0]} exited {p.returncode}; see {log}")


def probe(clip: Path) -> dict:
    s = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(clip)],
        check=True, stdout=subprocess.PIPE).stdout)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return dict(width=int(s["width"]), height=int(s["height"]), fps=float(num) / float(den),
                frames=int(s.get("nb_frames") or 0), duration=float(s.get("duration") or 0))


def body_stats(ply: Path) -> tuple[float, float]:
    """(body height, feet y) of frame 0 from the opaque splats, the numbers fourd.html derives."""
    import numpy as np
    from plyfile import PlyData
    v = PlyData.read(str(ply))["vertex"].data
    op = 1 / (1 + np.exp(-v["opacity"]))
    y = v["y"][op > 0.5]
    lo, hi = float(np.percentile(y, 1)), float(np.percentile(y, 99))
    return hi - lo, lo


class Pipeline:
    def __init__(self, a, clip: Path | None = None, name: str | None = None, shot: dict | None = None):
        self.a = a
        self.clip = Path(clip or a.clip).resolve()
        self.name = name or a.name
        self.shot = shot
        self.ctx = ROOT / ".context" / "run" / self.name
        self.ctx.mkdir(parents=True, exist_ok=True)
        for directory in (SHARE, CLIPS, MARBLE_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        self.state = State(self.ctx / "state.json")
        self.info = probe(self.clip)
        public = ROOT / "public"
        if not self.clip.is_relative_to(public):
            # Keep externally supplied source footage available to the shared viewer too.
            dest = public / "clips" / self.name / ("source" + self.clip.suffix.lower())
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.stat().st_size != self.clip.stat().st_size or dest.stat().st_mtime_ns != self.clip.stat().st_mtime_ns:
                shutil.copy2(self.clip, dest)
            self.clip = dest.resolve()
        self.video_url = "/" + quote(self.clip.relative_to(public).as_posix())
        self.clean_mp4 = CLIPS / f"{self.name}-clean.mp4"
        self.first_png = self.ctx / "first" / "f_0000.png"
        self.key = os.environ.get(a.marble_key or "WLT_API_KEY", "")
        self.people_limit = a.people if a.people else (4 if a.all_people else 0)
        self.multi = self.people_limit > 0
        self.marble = "none" if a.skip_marble else a.marble
        self.object_specs = [parse_object_spec(v) for v in (a.object or [])]
        want_objects = bool(self.object_specs) or not a.no_objects
        self.stages, self.deps = (multiperson_graph(self.people_limit, self.marble, want_objects)
                                  if self.multi else single_graph(self.marble, want_objects))
        self.certs = subprocess.run([PY, "-m", "certifi"], check=True, stdout=subprocess.PIPE,
                                    text=True).stdout.strip()
        self.state.record("_run", clip=str(self.clip), name=self.name, fps=a.fps, source=self.info,
                          shot=self.shot, started=time.strftime("%Y-%m-%d %H:%M:%S"))

    def stage_fn(self, stage):
        """`lhm_motion_01` -> self.lhm_motion(1); everything else is a plain method."""
        base, _, tail = stage.rpartition("_")
        if base and tail.isdigit():
            return lambda: getattr(self, base)(int(tail))
        return getattr(self, stage)

    def track_doc(self):
        path = self.ctx / "tracks" / "tracks.json"
        if not path.exists():
            raise RuntimeError("tracking stage has not run")
        return json.loads(path.read_text())

    def vacant(self, stage):
        """True for a person slot no track claimed: the graph is built for --people N slots before
        the track count is known, so the surplus slots must not block what depends on them."""
        base, _, tail = stage.rpartition("_")
        if not (base and tail.isdigit() and base in ("person_prep", "lhm_frozen", "lhm_motion")):
            return False
        try:
            return int(tail) >= self.track_doc()["trackCount"]
        except Exception:
            return False

    def track_or_skip(self, idx):
        doc = self.track_doc()
        if idx >= doc["trackCount"]:
            raise SkipStage(f"only {doc['trackCount']} person track(s) in this clip")
        track = doc["tracks"][idx]
        ident = self.ctx / "identity.json"
        if ident.exists() and track["track"] in (json.loads(ident.read_text()).get("failedTracks") or []):
            raise SkipStage(f"track {track['track']} failed the identity audit (changed person)")
        return track

    # ---- stages -------------------------------------------------------------
    def clean_first(self):
        """One inpainted frame, so Marble's image world can start before the full pass finishes."""
        shutil.rmtree(self.first_png.parent, ignore_errors=True)
        run([MODAL, "run", "worker/modal_clean_video.py", "--clip", str(self.clip),
             "--only", "0", "--frames-out", str(self.first_png.parent),
             "--report", str(self.ctx / "clean-first.json"), *self.clean_flags()],
            self.ctx / "clean_first.log")

    def clean(self):
        run([MODAL, "run", "worker/modal_clean_video.py", "--clip", str(self.clip),
             "--out", str(self.clean_mp4), "--frame0", str(self.ctx / "clean-f0.png"),
             "--masks", str(self.ctx / "masks.npz"), "--report", str(self.ctx / "clean.json"),
             *self.clean_flags()], self.ctx / "clean.log")

    def clean_flags(self):
        a = self.a
        return ["--fps", str(a.fps), "--width", str(self.info["width"]),
                "--height", str(self.info["height"]), "--dilate", str(a.dilate),
                "--bottom-extra", str(a.bottom_extra), "--lama-px", str(a.lama_px)] + \
               (["--moved-mask"] if a.moved_mask else [])

    def review(self):
        """The credit gate. 1600 credits are about to be spent on whatever these frames show."""
        if self.a.reuse_world:
            say("   gate not applicable: --reuse-world generates nothing and spends nothing")
            return
        out = self.ctx / "review"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        shots = []
        if self.clean_mp4.exists():
            info = probe(self.clean_mp4)
            n = max(1, info["frames"] or round(info["duration"] * info["fps"]) or 1)
            for k in range(6):
                i = min(n - 1, round(k * (n - 1) / 5))
                dest = out / f"clean_{i:04d}.png"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(self.clean_mp4),
                                "-vf", f"select=eq(n\\,{i})", "-vframes", "1", str(dest)], check=True)
                shots.append(dest)
        if self.first_png.exists():
            shots.append(self.first_png)
        if not shots:
            raise RuntimeError("nothing to review: the clean stage produced no frames")
        report = self.ctx / "clean.json"
        if report.exists():
            r = json.loads(report.read_text())
            say(f"   clean: {r.get('frames')} frames, person mask {100 * r.get('maskFraction', 0):.1f}% "
                f"of pixels, dilate {r.get('dilate')}/{r.get('bottomExtra')}")
        self.state.record("review", samples=[str(s) for s in shots])
        say("   REVIEW THESE BEFORE SPENDING 1600 CREDITS:")
        for s in shots:
            say(f"     {s}")
        say("   look for: the person fully gone; nothing of his left hanging in mid-air (a bag, a "
            "dumbbell, a plate); no limb smears; no second person at the frame edge.")
        say("   bad? re-clean with a larger --dilate/--bottom-extra (gym needed 90/120).")
        if not self.a.gate_pass:
            raise GateStop("re-run with --gate-pass once you have looked at the frames above "
                           "(or --no-gate to skip this gate on an unattended run)")
        say("   gate open (--gate-pass/--no-gate)")

    def marble_submit(self, script, target, suffix, extra=None) -> str:
        """Submit ONCE. Never called from a retry loop; `run(..., attempts=1)` and the guard in
        run() make a second submit -- a second 1600 credits -- impossible from this file."""
        log = self.ctx / f"marble_{suffix}.log"
        try:
            run([PY, f"scripts/{script}", "submit", *([str(target)] if target else []),
                 "--name", f"{self.name}-{suffix}",
                 "--marble-dir", str(MARBLE_DIR), "--spz", f"public/marble-{self.name}-{suffix}.spz",
                 "--thumb", str(SHARE / f"{self.name}-{suffix}-thumb.png"), *(extra or [])],
                log, {"WLT_API_KEY": self.key, "SSL_CERT_FILE": self.certs}, attempts=1)
            return ""                                   # submit polled to completion itself
        except RuntimeError:
            op = self.operation_id(log)
            if not op:
                raise RuntimeError("Marble submit failed before an operation existed, so no credits "
                                   f"were spent; safe to re-run with --force marble_{suffix}. See {log}")
            say(f"   submit died after operation {op} was created -- credits are already spent, "
                f"recovering by polling (never by resubmitting)")
            return op

    @staticmethod
    def operation_id(log: Path) -> str:
        for line in log.read_text().splitlines() if log.exists() else []:
            if line.startswith("op ") and "submitted" in line:
                return line.split()[1]
        return ""

    def marble_poll(self, script, op_id: str, suffix):
        """Retry the POLL, and only the poll. This takes an operation id: there is no path from
        here to a submit, so a flaky network costs nothing. Four polls died in one night."""
        assert op_id, "marble_poll needs an existing operation id"
        log = self.ctx / f"marble_{suffix}.log"
        last = None
        for attempt in range(self.a.poll_attempts):
            try:
                run([PY, f"scripts/{script}", "poll", op_id, "--name", f"{self.name}-{suffix}",
                     "--marble-dir", str(MARBLE_DIR), "--spz", f"public/marble-{self.name}-{suffix}.spz",
                     "--thumb", str(SHARE / f"{self.name}-{suffix}-thumb.png")],
                    log, {"WLT_API_KEY": self.key, "SSL_CERT_FILE": self.certs}, attempts=1, append=True)
                return
            except RuntimeError as e:
                last = e
                say(f"   poll {op_id} attempt {attempt + 1}/{self.a.poll_attempts} failed; retrying in 60s")
                time.sleep(60)
        raise RuntimeError(f"poll of operation {op_id} failed {self.a.poll_attempts} times. The world "
                           f"is generated and paid for; recover it by hand with "
                           f"`{Path(PY).name} scripts/{script} poll {op_id} --name {self.name}-{suffix} "
                           f"--marble-dir {MARBLE_DIR} --spz public/marble-{self.name}-{suffix}.spz`. "
                           f"Last error: {last}")

    def marble_reuse(self, script, suffix):
        """--reuse-world: adopt an existing world. No generation, no credits."""
        wid = self.a.reuse_world
        wj = MARBLE_DIR / f"{self.name}-{suffix}-world.json"
        spz = ROOT / "public" / f"marble-{self.name}-{suffix}.spz"
        if wj.exists() and spz.exists() and json.loads(wj.read_text()).get("world_id") == wid:
            say(f"   reusing world {wid} already on disk ({spz.name}); no network, no credits")
            return
        if not self.key:
            raise RuntimeError(f"--reuse-world needs ${self.a.marble_key} to fetch the world metadata")
        run([PY, f"scripts/{script}", "fetch", wid, "--name", f"{self.name}-{suffix}",
             "--marble-dir", str(MARBLE_DIR), "--spz", str(spz),
             "--thumb", str(SHARE / f"{self.name}-{suffix}-thumb.png")],
            self.ctx / f"marble_{suffix}.log", {"WLT_API_KEY": self.key, "SSL_CERT_FILE": self.certs},
            attempts=1)

    def marble_world(self, script, target, suffix, extra=None):
        if self.a.reuse_world:
            return self.marble_reuse(script, suffix)
        if not self.key:
            raise RuntimeError(f"no Marble key in ${self.a.marble_key}; export it, or pass "
                               f"--marble none / --reuse-world <id>")
        if (MARBLE_DIR / f"{self.name}-{suffix}-world.json").exists():
            say(f"   !! {self.name}-{suffix} already has a world. This buys a SECOND one for another "
                f"1600 credits. --reuse-world <id> adopts the existing one for nothing.")
        say(f"   submitting ONE marble-1.1 world (1600 credits) from "
            f"{Path(target).name if target else f'{suffix} inputs'}")
        op = self.marble_submit(script, target, suffix, extra)
        oid = op or self.operation_id(self.ctx / f"marble_{suffix}.log")
        if oid:                       # so a later failure never has to be grepped out of a log
            self.state.record("_marble", **{suffix: oid})
        if op:
            self.marble_poll(script, op, suffix)

    def marble_image(self):
        extra = []
        if self.prompt_json.exists():
            extra = ["--prompt", json.loads(self.prompt_json.read_text())["text_prompt"], "--seed", "7"]
        self.marble_world("marble_image_world.py", self.first_png, "image", extra)

    def marble_video(self):
        self.marble_world("marble_video_world.py", self.clean_mp4, "clean")

    # ---- vision-written prompt, measured mode choice, verification ----------
    @property
    def prompt_json(self) -> Path:
        return self.ctx / "prompt.json"

    @property
    def mode_json(self) -> Path:
        return self.ctx / "mode.json"

    def world_prompt(self):
        """Look at the clip and write the text_prompt disable_recaption will pin.

        Marble's video mode captions the clip and generates from its own caption; image and
        multi-image do the same unless disable_recaption is set with a prompt of ours. Every prompt
        before this stage existed was hand-written per clip.
        """
        run([PY, "scripts/world_prompt.py", "--clip", str(self.clip), "--n", "6",
             "--out", str(self.prompt_json)], self.ctx / "world_prompt.log", attempts=2)
        rec = json.loads(self.prompt_json.read_text())
        say(f"   prompt: {rec['text_prompt']}")

    def world_mode(self, decide_only=False):
        """Measured image-vs-multi-image choice (scripts/select_world_mode.py). Uses the Pi3X poses
        when a previous run left them; before any solve exists it falls back to the coarse
        pre-solve's heading sweep, and with neither it says so rather than guessing."""
        # Ordering note: the Pi3X solve runs after the world half in this graph, so on a first pass
        # `--marble multi` wants `--only pi3x` run first (or a coarse admission.json from
        # worker/experiments/clip_admission.py). Image and video modes need neither.
        cam = self.ctx / "pi3x" / "cameras.json"
        pred = self.ctx / "admission.json"
        cmd = [PY, "scripts/select_world_mode.py", "--out", str(self.mode_json)]
        if cam.exists():
            cmd += ["--cameras", str(cam), "--clip", str(self.clip)]
        elif pred.exists():
            cmd += ["--predict-json", str(pred)]
        else:
            say("   no camera poses and no coarse solve: cannot measure the mode, leaving --marble as given")
            return None
        run(cmd, self.ctx / "world_mode.log", attempts=2)
        rec = json.loads(self.mode_json.read_text())
        say(f"   measured mode: {rec['mode']} -- {rec['why']}")
        return rec

    def clean_multi(self):
        """Inpaint only the frames multi-image mode will send, chosen by angle and by structure."""
        rec = self.world_mode()
        if not rec or rec.get("mode") != "multi-image":
            raise SkipStage("no measured multi-image decision for this clip; use --marble image/video")
        # --only indexes the DECODED (post --fps) sequence; the chosen frames are source indices.
        report = self.ctx / "clean.json"
        if report.exists():
            idx = json.loads(report.read_text())["indices"]
            only = ",".join(str(idx.index(f)) for f in rec["frames"])
        else:
            step = self.info["fps"] / self.a.fps
            only = ",".join(str(int(round(f / step))) for f in rec["frames"])
        out = self.ctx / "clean-multi"
        shutil.rmtree(out, ignore_errors=True)
        run([MODAL, "run", "worker/modal_clean_video.py", "--clip", str(self.clip),
             "--only", only, "--frames-out", str(out),
             "--report", str(self.ctx / "clean-multi.json"), *self.clean_flags()],
            self.ctx / "clean_multi.log")

    def marble_multi(self):
        rec = json.loads(self.mode_json.read_text())
        frames = sorted(self.ctx.joinpath("clean-multi").glob("f_*.png"))
        if len(frames) != len(rec["frames"]):
            raise RuntimeError(f"{len(frames)} cleaned frames for {len(rec['frames'])} chosen views")
        images = [f"{p}:{az}" for p, az in zip(frames, rec["azimuth"])]
        self.marble_world("marble_multi_world.py", None, "multi", extra=[
            "--images", *images, "--prompt-file", str(self.prompt_json), "--seed", "7"])

    def verify(self):
        """Render the world we got back from the clip's own poses and score it against the footage."""
        out = self.ctx / "verify"
        scale0 = (self.state.data["stages"].get("_scale") or {}).get("scale0")
        if scale0 is None:
            raise RuntimeError("no fitted scale0 in state.json; the scale_fit stage has not run")
        # verify_world renders with the viewer's own mapping (no re-anchoring), so it needs the
        # packaged, levelled cameras.json, not the raw Pi3X one
        packaged = ROOT / "public" / "worlds" / f"{self.name}-4d" / "cameras.json"
        cmd = [PY, "scripts/verify_world.py", "--world", str(self.world_spz()),
               "--cameras", str(packaged if packaged.exists() else self.cameras()), "--clip", str(self.clip),
               "--scale0", str(scale0), "--tag", self.name, "--out", str(out),
               "--share", str(SHARE)]
        if self.prompt_json.exists():
            cmd += ["--prompt-file", str(self.prompt_json)]
        run(cmd, self.ctx / "verify.log", attempts=2)
        rep = json.loads((out / "report.json").read_text())
        self.state.record("_verify", medianScore=rep.get("medianScore"), passed=rep.get("pass"),
                          regenerate=rep.get("regenerate"), sheet=rep.get("sheet"))
        say(f"   world fidelity {rep.get('medianScore')}/100, pass {rep.get('pass')}; sheet {rep.get('sheet')}")
        for v in rep.get("verdicts", []):
            say(f"     f{v.get('frame')}: {v.get('score')} -- {v.get('worst_error')}")
        if rep.get("regenerate"):
            say(f"   CORRECTED PROMPT: {rep['correction']['text_prompt']}")
            say(f"   {rep['correction']['why']}")

    def pi3x(self):
        shutil.rmtree(self.ctx / "pi3x", ignore_errors=True)
        run([MODAL, "run", "worker/modal_motion.py", "--experiment", "pi3x",
             "--video", str(self.clip), "--out", str(self.ctx / "pi3x")], self.ctx / "pi3x.log")
        self.pose_guard()

    def pose_guard(self):
        """Between the solve and everything that reads it: did the camera teleport?

        A cut too soft for the scene-score detector is indistinguishable from continuous motion in
        the frames and obvious in the poses, where it becomes a jump across the scene in one frame
        interval. So does a solve that simply failed to register, which has bitten this pipeline
        more often than cuts have. Either way the poses are not a camera path and nothing may be
        placed against them, so this raises instead of letting the run continue.
        """
        p = subprocess.run([PY, "scripts/shot_cuts.py", "--poses", str(self.ctx / "pi3x"),
                            "--json", str(self.ctx / "pose-guard.json")],
                           cwd=ROOT, env=dict(os.environ, **LOCAL_ENV),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout.strip().splitlines():
            say(f"   {line}")
        if p.returncode == 3:
            doc = json.loads((self.ctx / "pose-guard.json").read_text())
            raise RuntimeError(doc["message"])
        if p.returncode == 4:
            doc = json.loads((self.ctx / "pose-guard.json").read_text())
            raise RuntimeError(f"the pose guard could not check the solve: {doc.get('reason')}. "
                               f"An unchecked solve is a failed solve here -- hp33 wrote no "
                               f"cameras.json and would otherwise sail through. Re-run pi3x.")
        if p.returncode != 0:
            raise RuntimeError(f"the pose guard itself failed (exit {p.returncode}); the solve is "
                               f"UNCHECKED and nothing may be placed against it")

    def tracks(self):
        """Every person in the clip, linked into identity-stable tracks (one MultiHMR pass)."""
        shutil.rmtree(self.ctx / "tracks", ignore_errors=True)
        run([MODAL, "run", "worker/modal_multiperson.py::main", "--video", str(self.clip),
             "--cameras", str(self.ctx / "pi3x" / "cameras.json"), "--fps", str(self.a.fps),
             "--det-thresh", str(self.a.det_thresh), "--out", str(self.ctx / "tracks")],
            self.ctx / "tracks.log")
        doc = self.track_doc()
        say(f"   {doc['trackCount']} person track(s): " + ", ".join(
            f"{t['track']}=score {t['quality']['score']:.2f} over {t['quality']['samples']} samples"
            for t in doc["tracks"]))
        self.identity_guard()

    def identity_guard(self):
        """Did the tracker keep each person in his own track?

        The old answer (`reproject_multiperson.py`, avatar-in-its-own-box) could not be no: the box
        and the poses come from the same tracker, so a swap moves both and the avatar lands inside
        its own swapped box. scripts/identity_audit.py asks the source pixels instead, against fixed
        frame-0 and frame-N references, and runs a deliberate-swap control so an "ok" means the test
        had the power to say otherwise.
        """
        p = subprocess.run([PY, "scripts/identity_audit.py", "--tracks", str(self.ctx / "tracks"),
                            "--clip", str(self.clip), "--swap-test",
                            "--json-out", str(self.ctx / "identity.json")],
                           cwd=ROOT, env=dict(os.environ, **LOCAL_ENV),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout.strip().splitlines():
            say(f"   {line}")
        if p.returncode == 3:
            doc = json.loads((self.ctx / "identity.json").read_text())
            failed = doc.get("failedTracks") or []
            if failed and len(failed) < len(doc.get("tracks", [])) and not doc.get("reason"):
                # some tracks changed person, some did not: build the ones that held (the swap
                # control passed, so the ok verdicts mean something) and skip the rest
                say(f"   identity: track(s) {failed} changed person and will be skipped; "
                    f"{[t for t in doc['tracks'] if t not in failed]} held")
                return
            raise RuntimeError("the identity audit says a track changed person. Do not build avatars "
                               f"from these tracks; see {self.ctx / 'identity.json'}.")
        if p.returncode == 4:
            raise RuntimeError(f"the identity audit could not run; see {self.ctx / 'identity.json'}")
        if p.returncode != 0:
            raise RuntimeError(f"scripts/identity_audit.py failed (exit {p.returncode})")

    def person_prep(self, idx=None):
        if idx is None:
            frame = self.a.person_frame if self.a.person_frame is not None else self.info["frames"] // 2
            run([PY, "scripts/prepare_lhm_person.py", str(self.clip), "--frame", str(frame),
                 "--method", self.a.person_method, "--out", str(self.ctx / "prepared-person")],
                self.ctx / "person_prep.log")
            return
        self.track_or_skip(idx)
        run([PY, "scripts/prepare_track_person.py", str(self.ctx / "tracks"), "--track", str(idx),
             "--clip", str(self.clip), "--method", self.a.person_method, "--python", PY,
             "--out", str(self.ctx / f"prepared-{idx:02d}")], self.ctx / f"person_prep_{idx:02d}.log")

    def lhm_frozen(self, idx=None):
        prepared = self.ctx / ("prepared-person" if idx is None else f"prepared-{idx:02d}")
        dest = self.ctx / ("lhm-frozen" if idx is None else f"lhm-frozen-{idx:02d}")
        if idx is not None:
            self.track_or_skip(idx)
        shutil.rmtree(dest, ignore_errors=True)
        run([MODAL, "run", "worker/modal_lhm.py", "--prepared", str(prepared), "--out", str(dest)],
            self.ctx / (f"lhm_frozen_{idx:02d}.log" if idx is not None else "lhm_frozen.log"))

    def lhm_motion(self, idx=None):
        if idx is None:
            shutil.rmtree(self.ctx / "lhm-motion", ignore_errors=True)
            run([MODAL, "run", "worker/modal_lhm.py", "--canonical", str(self.ctx / "lhm-frozen"),
                 "--video", str(self.clip), "--cameras", str(self.ctx / "pi3x" / "cameras.json"),
                 "--out", str(self.ctx / "lhm-motion")], self.ctx / "lhm_motion.log")
            return
        track = self.track_or_skip(idx)
        dest = self.ctx / f"lhm-motion-{idx:02d}"
        shutil.rmtree(dest, ignore_errors=True)
        # The depth reference PLY holds every masked person, so restrict the registration to this
        # track's own box in its first sample or both avatars inherit a blended scale.
        first = track["quality"]["firstSample"]
        rec = next(r for r in json.loads((self.ctx / "tracks" / f"track_{idx:02d}" / "motion.json").read_text())["frames"]
                   if r["sample"] == first)
        roi = ",".join(f"{v:.0f}" for v in rec["maskBox"])
        run([MODAL, "run", "worker/modal_multiperson.py::animate", "--video", str(self.clip),
             "--canonical", str(self.ctx / f"lhm-frozen-{idx:02d}"),
             "--cameras", str(self.ctx / "pi3x" / "cameras.json"),
             "--track-dir", str(self.ctx / "tracks" / f"track_{idx:02d}"),
             "--track-id", str(idx), "--depth-roi", roi,
             "--depth-reference", str(self.ctx / "pi3x" / f"frame_{first:03d}.ply"),
             "--out", str(dest)], self.ctx / f"lhm_motion_{idx:02d}.log")

    def package(self):
        out = ROOT / "public" / "worlds" / f"{self.name}-4d"
        shutil.rmtree(out, ignore_errors=True)
        run([PY, "scripts/package_person_sequence.py", str(self.ctx / "lhm-motion"),
             "--cameras", str(self.ctx / "pi3x" / "cameras.json"), "--clip", str(self.clip),
             "--out", str(out)], self.ctx / "package.log")

    def package_people(self):
        out = ROOT / "public" / "worlds" / f"{self.name}-4d"
        shutil.rmtree(out, ignore_errors=True)
        motions = []
        for i in range(self.people_limit):
            folder = self.ctx / f"lhm-motion-{i:02d}"
            if (folder / "sequence.json").exists():
                motions += ["--motion", f"{i}={folder}"]
        if not motions:
            raise RuntimeError("no per-track animation finished")
        run([PY, "scripts/package_multiperson.py", *motions, "--tracks", str(self.ctx / "tracks"),
             "--cameras", str(self.ctx / "pi3x" / "cameras.json"), "--clip", str(self.clip),
             "--out", str(out)], self.ctx / "package_people.log")

    def objects(self):
        """Objects in the clip that are neither a body nor the static scene. Format: docs/objects.md.

        With `--object ID:MOTION:PROMPT[:ROI]` the caller names each object and its motion case and
        the named path below segments, solves and packages exactly those. Without it the stage looks
        for itself: `detect_object_flights.py` finds every span in which something small flies free
        under gravity, and only an accepted flight is lifted, described from its own pixels, given a
        shape and packaged. A clip in which nothing is thrown reports that and passes.
        """
        world = ROOT / "public" / "worlds" / f"{self.name}-4d"
        if not (world / "people.json").exists() and not (world / "person").exists():
            raise RuntimeError("the people package has not run; objects live beside it")
        if not self.object_specs:
            return self.objects_auto(world)
        outs = []
        for spec in self.object_specs:
            oid, motion = spec["id"], spec["motion"]
            odir = self.ctx / f"object-{oid}"
            odir.mkdir(parents=True, exist_ok=True)

            seg = [PY, "scripts/object_segment.py", "--clip", str(self.clip),
                   "--prompt", spec["prompt"], "--out", str(odir / "seg")]
            roi = spec["roi"]
            if roi.startswith("joint="):
                t, j, r = roi[6:].split(",")
                seg += ["--roi-joint", f"{t}:{j}:{r}", "--tracks-dir", str(self.ctx / "tracks")]
            elif roi.startswith("box="):
                seg += ["--roi-box", roi[4:]]
            if not (odir / "seg" / "segment.json").exists():
                run(seg, odir / "segment.log")

            fit = None
            if motion in ("free", "handoff"):
                t2 = odir / "track2d.json"
                if not t2.exists():
                    run([PY, "scripts/track_object_2d.py", "--clip", str(self.clip),
                         "--tracks", str(self.ctx / "tracks" / "tracks.json"),
                         "--out", str(t2), "--patches", str(odir / "patches.npz")],
                        odir / "track2d.log")
                fit = odir / "fit3d.json"
                if not fit.exists():
                    run([PY, "scripts/lift_object_3d.py", "--track2d", str(t2),
                         "--cameras", str(self.cameras()),
                         "--people", str(world / "people.json"),
                         "--tracks-dir", str(self.ctx / "tracks"),
                         "--thrower", "1", "--catcher", "0", "--out", str(fit)],
                        odir / "lift3d.log")
                ori = odir / "orientation.json"
                if not ori.exists():
                    f = json.loads(fit.read_text())["fits"]["handAnchored"]
                    run([PY, "scripts/object_orientation.py", "--track2d", str(t2),
                         "--patches", str(odir / "patches.npz"),
                         "--v0", ",".join(f"{v:.6f}" for v in f["v0"]),
                         "--g", f"0,-{json.loads(fit.read_text())['gravityUnitsPerS2']:.6f},0",
                         "--out", str(ori)], odir / "orientation.log")

            pkg = [PY, "scripts/package_objects.py", "--world", str(world),
                   "--tracks-dir", str(self.ctx / "tracks"), "--cameras", str(self.cameras()),
                   "--id", oid, "--motion", motion, "--prompt", spec["prompt"],
                   "--thrower", "1", "--catcher", "0"]
            if fit:
                pkg += ["--fit", str(fit), "--orientation", str(odir / "orientation.json")]
            model = odir / "object.ply"
            if model.exists():
                pkg += ["--model", str(model)]
            run(pkg, odir / "package.log")
            outs.append(oid)
        self.state.record("_objects", ids=outs, world=str(world / "objects.json"))
        say(f"   packaged {len(outs)} object(s): {', '.join(outs)}")

    @staticmethod
    def scale_file(world: Path):
        """The single-person world file that actually carries a metre scale, if any."""
        for n, keys in (("placement.json", ("metresPerWorldUnit",)), ("collision.json", ("metresPerWorldUnit",)),
                        ("framealign.json", ("metresPerWorldUnitAvatar",))):
            p = world / n
            if p.exists() and any(json.loads(p.read_text()).get(k) for k in keys):
                return p
        return None

    def objects_auto(self, world: Path):
        """Seedless thrown-object lane: detect -> lift -> describe -> shape -> package, each step
        skipped when its output exists, and the chain stops honestly at the first step with nothing
        to pass on. No prompt, no seed frame, no thrower/catcher index comes from the caller."""
        odir = self.ctx / "objects"
        odir.mkdir(parents=True, exist_ok=True)
        multi = (world / "people.json").exists()
        if multi:
            tracks, scale, masks = self.ctx / "tracks" / "tracks.json", world / "people.json", None
        else:
            tracks = world / "person" / "motion.json"
            scale, masks = self.scale_file(world), self.ctx / "masks.npz"
        if scale is None:
            raise SkipStage("no metre scale on disk for this world yet (place_fit has not run), "
                            "and a ballistic fit needs |g| in world units")

        flights = odir / "flights.json"
        if not flights.exists():
            cmd = [PY, "scripts/detect_object_flights.py", "--clip", str(self.clip),
                   "--tracks", str(tracks), "--cameras", str(self.cameras()), "--people", str(scale),
                   "--cache", str(odir / "detect-cache.npz"), "--out", str(flights),
                   "--crops", str(odir / "crops")]
            if masks and masks.exists():
                cmd += ["--masks", str(masks)]
            run(cmd, odir / "detect.log")
        fl = json.loads(flights.read_text())["flights"]
        if not fl:
            self.state.record("_objects", ids=[], flights=0)
            say("   no thrown objects: nothing in this clip flies free under gravity")
            return
        say("   %d free flight(s): %s" % (len(fl), ", ".join(
            f"f{r['frames'][0]}-{r['frames'][-1]} ({r['nObs']} obs, {r['speedMs']:.1f} m/s)" for r in fl)))
        if not multi:
            self.state.record("_objects", ids=[], flights=len(fl),
                              note="single-person world: no floorFit or per-track wrist poses to lift against")
            say("   flights recorded, not lifted: the single-person layout carries no floor fit or "
                "wrist poses. Rerun with --people N to package them")
            return

        fit = odir / "fit3d.json"
        if not fit.exists():
            run([PY, "scripts/lift_object_3d.py", "--flights", str(flights),
                 "--cameras", str(self.cameras()), "--people", str(world / "people.json"),
                 "--tracks-dir", str(self.ctx / "tracks"), "--tracks", str(tracks), "--out", str(fit)],
                odir / "lift3d.log")
        fj = json.loads(fit.read_text())
        if not fj["flights"]:
            self.state.record("_objects", ids=[], flights=len(fl), lifted=0)
            say("   %d flight(s) detected, none reached a tracked wrist at both ends: "
                "%s. Not packaged" % (len(fl), "; ".join(d["rejected"] for d in fj["droppedFlights"])))
            return

        desc = odir / "description.json"
        if not desc.exists():
            cmd = [PY, "scripts/describe_object.py", "--flights", str(flights),
                   "--crops-dir", str(odir / "crops"), "--cameras", str(self.cameras()), "--out", str(desc)]
            if not os.environ.get("OPENAI_API_KEY"):
                cmd += ["--no-vlm"]
            run(cmd, odir / "describe.log")
        dj = json.loads(desc.read_text())
        dd = dj["description"]
        say(f"   it looks like: {dd['label']} (confidence {dd.get('confidence', 0):.2f}), "
            f"~{100 * dj['sizeMetres'][1]:.0f} cm long")

        model = odir / "object.ply"
        shape_note = ""
        if not model.exists() and not self.a.no_shape and os.path.exists(MODAL):
            shape = odir / "shape"
            try:
                run([MODAL, "run", "worker/modal_image_to_3d.py", "--image", str(odir / "best-crop.png"),
                     "--out-dir", str(shape), "--refine-first", "--refine-prompt", dd["refinePrompt"],
                     "--refine-strength", "0.85",
                     "--note", "auto path: crop picked by detect_object_flights.py sharpness, prompt "
                               "written by describe_object.py from the in-flight crops"],
                    odir / "shape.log")
                run([PY, "scripts/object_appearance.py", "--in", str(shape / "object.ply"),
                     "--out", str(model), "--largest-cluster", "--budget", "6000",
                     "--report", str(odir / "appearance.json")], odir / "appearance.log")
            except Exception as e:      # a proxy is the documented floor, not a failure
                shape_note = f"shape generation failed: {e}"
                say("   " + shape_note + "; packaging a proxy")
        oid = dj["id"]
        pkg = [PY, "scripts/package_objects.py", "--world", str(world), "--fit", str(fit),
               "--tracks-dir", str(self.ctx / "tracks"), "--cameras", str(self.cameras()),
               "--id", oid, "--label", dd["label"], "--prompt", dd["segmentWord"],
               "--object-class", "thrown", "--flights-json", str(flights),
               "--size-m", ",".join(f"{v:.4f}" for v in dj["sizeMetres"]),
               "--color-srgb", ",".join(f"{v:.3f}" for v in dd["colourSRGB"]),
               "--colour-provenance",
               f"VLM reading of {len(dj['cropsUsed'])} in-flight crops ({dd.get('model', 'no VLM')}); "
               f"crop-pixel median sRGB {['%.2f' % v for v in dd['colourSRGBFromPixels']]}"]
        if model.exists():
            pkg += ["--model", str(model), "--invented", "--shape-provenance",
                    "INVENTED. Image-to-3D (TRELLIS-image-large) from the sharpest in-flight crop after an "
                    "SDXL img2img refine (strength 0.85) whose prompt was written by a VLM from the crops: "
                    f"'{dd['refinePrompt']}'. Size from bbox px x fitted depth / focal (an upper bound), "
                    "long axis capped by the VLM aspect. Nothing here was typed by a person."]
        else:
            pkg += ["--shape-provenance", "proxy capsule sized from the tracklet's apparent scale at the "
                                         "fitted depth. " + shape_note]
        run(pkg, odir / "package.log")
        self.state.record("_objects", ids=[oid], flights=len(fl), lifted=len(fj["flights"]),
                          world=str(world / "objects.json"), label=dd["label"])
        say(f"   packaged '{oid}' ({dd['label']}): {len(fj['flights'])} flight(s) -> {world / 'objects.json'}")

    # ---- world scale + fine-tune ------------------------------------------
    def world_spz(self) -> Path:
        for suffix in ("clean", "image", "multi"):
            p = ROOT / "public" / f"marble-{self.name}-{suffix}.spz"
            if p.exists():
                return p
        raise RuntimeError(f"no Marble world on disk for {self.name}: expected "
                           f"public/marble-{self.name}-clean.spz")

    def cameras(self) -> Path:
        p = self.ctx / "pi3x" / "cameras.json"
        if not p.exists():
            raise RuntimeError(f"{p} is missing; the pi3x stage has not run")
        return p

    def n_cameras(self) -> int:
        return len(json.loads(self.cameras().read_text())["cameras"])

    def frame_align(self):
        """Gravity in the Pi3X frame (scripts/frame_align.py graph), before anything is packaged.

        The packagers put camera 0 at the origin; without this they also kept its ORIENTATION, so
        the whole clip inherited the phone's pitch at frame 0 while Marble's world is level. Every
        later stage then absorbed that angle with a translation: hpwide fitted its scale on a
        bimodal depth ratio and its feet 36-63 cm under the floor from a 15 deg tilt. The file this
        writes is read by scripts/sfm_frame.py in the packagers, the depth-ratio diagnostic and the
        fine-tune export, so the packaged frame is levelled and nothing downstream needs a flag.
        """
        out = self.ctx / "pi3x" / "framealign.json"
        run([PY, "scripts/frame_align.py", "graph", "--pi3x", str(self.ctx / "pi3x"),
             "--clip", self.name, "--anchor-samples", self.anchor_samples(), "--write"],
            self.ctx / "frame_align.log", attempts=1)
        fa = json.loads(out.read_text())
        self.state.record("_frame_align", tiltFromYDeg=fa["tiltFromYDeg"], adopted=fa["adopted"],
                          source=fa["source"], estimatorSpreadDeg=fa["estimatorSpreadDeg"],
                          cameraHeightUnits=fa["cameraHeightUnits"], file=str(out))
        say(f"   camera 0 is {fa['tiltFromYDeg']:.1f} deg off gravity ({fa['source']}); estimators "
            f"agree to {fa['estimatorSpreadDeg']:.1f} deg; camera {fa['cameraHeightUnits']:.3f} native "
            f"units over the floor")

    def anchor_samples(self) -> str:
        n = self.n_cameras()
        return ",".join(str(round(k * (n - 1) / 7)) for k in range(8))

    def diag(self, scale0: float, probe: int):
        """bake_video_colours --diag at one scale0 -> (5 sampled depth ratios, figure path).

        --anchors is not optional: with --cameras and no point cloud the diagnostic exits with
        `need --scene-points, --anchors or --depth-dir` before computing anything. The ratio is
        parsed out of `(ratio 0.993)`, which is followed by `, p10 ... p90 ...` on the same line --
        a rsplit on "ratio" takes all of that and float() raises.
        """
        log = self.ctx / f"diag-{probe}.log"
        tag = f"{self.name}-diag-s{scale0:.4f}".replace(".", "")
        run([PY, "scripts/bake_video_colours.py", "--diag", "--world", str(self.world_spz()),
             "--cameras", str(self.cameras()), "--clip", str(self.clip),
             "--anchors", str(self.ctx / "pi3x" / "anchors.npz"),
             "--anchor-samples", self.anchor_samples(),
             "--person", str(self.person_ply()), "--scale0", f"{scale0:.6f}",
             "--tag", tag], log, attempts=1)
        rows = [tuple(float(x) for x in m) for m in
                re.findall(r"\(ratio ([0-9.]+)\), p10 ([0-9.]+) p90 ([0-9.]+)", log.read_text())]
        if not rows:
            raise RuntimeError(f"the depth-ratio diagnostic printed no ratios; see {log}")
        return rows, SHARE / f"{tag}-depth-ratio.png"

    def person_ply(self) -> Path:
        p = ROOT / "public" / "worlds" / f"{self.name}-4d" / "person" / "frame_000.ply"
        if not p.exists():
            raise RuntimeError(f"{p} is missing; the package stage has not run")
        return p

    def scale_fit(self):
        """Fit scale0, the SfM-to-Marble scale, against the Pi3X anchor cloud.

        Mandatory before anything ships: Marble's own metric_scale_factor has been wrong by 2.9x
        (gym), 2.1x (atrium) and 2.5x (living). ratio = splat depth / measured depth is ~1/scale0,
        so scale0 <- scale0 * geomean(ratio) converges in two or three probes.
        """
        s, probes = self.a.scale0, []
        if s is None:
            s = self.placement().get("floorMatchedScale") or 1.0
            for i in range(self.a.scale_probes):
                rows, fig = self.diag(s, i)
                ratios = [r for r, _, _ in rows]
                gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
                probes.append((s, gm))
                say(f"   probe {i}: scale0 {s:.4f} -> ratios {[round(r, 3) for r in ratios]} "
                    f"(geomean {gm:.4f})")
                if abs(math.log(gm)) < 0.006:
                    break
                s = s * gm
            else:
                raise RuntimeError(f"scale0 did not converge in {self.a.scale_probes} probes "
                                   f"{[(round(x, 4), round(g, 4)) for x, g in probes]}; pass "
                                   f"--scale0 by hand or raise --scale-probes")
        else:
            rows, fig = self.diag(s, "given")
            ratios = [r for r, _, _ in rows]
            gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        p10, p90 = [lo for _, lo, _ in rows], [hi for _, _, hi in rows]
        spread = [r for r in ratios if not SCALE_RATIO_BAND[0] <= r <= SCALE_RATIO_BAND[1]]
        wide = [(round(lo, 2), round(hi, 2)) for _, lo, hi in rows
                if not (SCALE_SPREAD_BAND[0] <= lo and hi <= SCALE_SPREAD_BAND[1])]
        self.state.record("_scale", scale0=s, geomean=gm, ratios=ratios, p10=p10, p90=p90, probes=probes,
                          gateOk=not spread and not wide, medianOk=not spread, spreadOk=not wide,
                          figure=str(fig))
        say(f"   fitted scale0 {s:.4f}, ratios {[round(r, 3) for r in ratios]}, geomean {gm:.4f}, "
            f"p10 {min(p10):.3f}-{max(p10):.3f} p90 {min(p90):.3f}-{max(p90):.3f}")
        if wide and not self.a.allow_scale_spread:
            raise RuntimeError(
                f"the depth ratio is not ONE mode at 1.0: per-frame p10/p90 {wide} at scale0 {s:.4f} "
                f"(want both inside {SCALE_SPREAD_BAND[0]}-{SCALE_SPREAD_BAND[1]}; medians "
                f"{[round(r, 3) for r in ratios]}). A median near 1 over a spread this wide is two "
                f"populations averaged, not a scale -- hpwide read p10 0.71 / p90 1.37 with medians "
                f"0.98-1.02 while its frame was 15 deg off gravity. Read {fig}, check "
                f"{self.ctx / 'frame_align.log'} and the world's geometry before passing "
                f"--allow-scale-spread.")
        if spread and not self.a.allow_scale_spread:
            raise RuntimeError(
                f"depth ratios {[round(r, 3) for r in ratios]} are not all inside "
                f"{SCALE_RATIO_BAND[0]}-{SCALE_RATIO_BAND[1]} at any single scale (geomean fit {s:.4f} "
                f"gives {gm:.4f}). That is Marble geometry error or SfM drift, not a scale error -- it "
                f"means the avatar reads ~{abs(1 - min(ratios)) * 100:.0f}% wrong at one end of the "
                f"clip. Read share/{self.name}-diag-*-depth-ratio.png, then either re-trim the clip to "
                f"the window that fits or pass --allow-scale-spread to ship the geometric-mean fit "
                f"with that caveat (gym, atrium and selfie did).")
        if spread or wide:
            say(f"   !! --allow-scale-spread: shipping the geomean fit; medians "
                f"{(max(ratios) - 1) * 100:+.0f}% / {(min(ratios) - 1) * 100:+.0f}%, p10/p90 "
                f"{min(p10):.2f}/{max(p90):.2f} across the clip")

    def place_fit(self):
        """Size check + per-sample placement solve (scripts/place_solve.py, share/PLACEMENT.md).

        scale_fit ends the day it always did: one scale0 for the whole clip, fitted as the geometric
        mean of a depth ratio that on most clips never touches 1.0 at any single value. That number
        is the RATIO between the Marble world and the depth map, so it is blind to two things at
        once -- a global size error, which cancels in it, and the per-sample drift, which it averages
        away. This stage measures both against the world's own geometry and writes
        `public/worlds/<clip>-4d/placement.json`, which fourd.html reads under `?place=1`.

        It FAILS the run when the size check fails, rather than shipping a giant: atrium went out
        25 % oversized and gym 73 %, and neither was visible in any number the pipeline printed.
        """
        fit = self.state.data["stages"].get("_scale") or {}
        s = fit.get("scale0")
        if s is None:
            raise RuntimeError("no fitted scale0: run the scale_fit stage first")
        floor = self.placement().get("floor")
        if floor is None:
            raise RuntimeError("no implied floor: the package stage has not produced frame_000.ply")
        wdir = ROOT / "public" / "worlds" / f"{self.name}-4d"
        out = wdir / "placement.json"
        cmd = [PY, "scripts/place_solve.py", "--clip", self.name,
               "--world", str(self.world_spz()), "--cameras", str(wdir / "cameras.json"),
               "--scale", f"{s:.6f}", "--floor", f"{floor:.6f}", "--out", str(out)]
        people = wdir / "people.json"
        cmd += (["--people", str(people)] if people.exists()
                else ["--person", str(wdir / "person" / "sequence.json")])
        if not self.a.allow_bad_size:
            cmd.append("--fail-on-size")
        run(cmd, self.ctx / "place_fit.log", attempts=1)
        doc = json.loads(out.read_text())
        sc = doc["surface"]["sizeCheck"]
        res = doc["residualCm"]
        self.state.record("_placement_solve", out=str(out), samples=doc["samples"],
                          metresPerWorldUnit=doc["metresPerWorldUnit"], sizeCheck=sc,
                          residualCm=res, reprojectionCostPx=doc["reprojectionCostPx"])
        say(f"   1 u = {doc['metresPerWorldUnit']:.3f} m; source camera "
            f"{sc['cameraHeightM']:.2f} m above the floor, room {sc['floorToCeilingM']:.2f} m "
            f"floor to ceiling -> size {'PASS' if sc['pass_'] else 'FAIL'}")
        for pid in res["before"]:
            say(f"   {pid}: contact residual {res['before'][pid]['rms']:.1f} -> "
                f"{res['after'][pid]['rms']:.1f} cm RMS over {doc['samples']} samples")


    def anchors(self):
        """The anchored checks (scripts/anchor_checks.py, share/METHODS-REVIEW.md §4).

        Everything above this line measured the clip against something the clip itself produced.
        This stage is the first that does not: the metre comes from a metric-depth model or a
        VLM-named standard object rather than from `1.70 / stature`, the silhouette is scored
        against Mask R-CNN rather than against the tracker's own box, the camera height is a trace
        rather than a median, and a fit whose residual sits under its own noise floor refuses.

        It FAILS the run. A clip whose metres rest on the avatar and nothing else is reported as
        weakly anchored and stops here, because every metre it goes on to print -- camera-freedom
        cliffs, clamp boxes, foot residuals, jitter -- inherits that assumption silently.
        """
        out = self.ctx / "anchors.json"
        cmd = [PY, "scripts/anchor_checks.py", "--clip", self.name,
               "--tol", str(self.a.ruler_tol), "--json-out", str(out)]
        if self.a.ruler_vlm:
            cmd += ["--ruler-arg=--vlm"]
        if self.a.allow_weak_anchor:
            cmd.append("--no-gate")
        p = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, **LOCAL_ENV),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout.strip().splitlines():
            say(f"   {line}")
        if p.returncode == 3:
            doc = json.loads(out.read_text()) if out.exists() else {}
            raise RuntimeError(f"anchor checks failed: {', '.join(doc.get('failed', ['unknown']))}. "
                               f"See {out}. Pass --allow-weak-anchor to record the failure and go on.")
        if p.returncode != 0:
            raise RuntimeError(f"scripts/anchor_checks.py itself failed (exit {p.returncode})")
        if out.exists():
            self.state.record("_anchors", **json.loads(out.read_text()))

    def finetune(self):
        """The own-clip recipe: fitted scale0 -> finetune_export_bedroom -> run5 flags on the A10G
        -> figures -> import (share/FINETUNE-STATUS.md, share/NEW-CLIPS-STATUS.md).

        finetune_export.py is the corridor route and needs a .wvp container; every own clip since
        bedroom uses finetune_export_bedroom.py (cameras.json + Pi3X anchors). The ship decision is
        a human reading the off-path figures, so this stops after one training run and the figures.
        """
        if not self.a.gpu_box:
            raise RuntimeError("Fine-tuning requires --gpu-box or WANDER_GPU_BOX; use --skip-finetune to omit it")
        s = (self.state.data["stages"].get("_scale") or {}).get("scale0")
        if s is None:
            raise RuntimeError("no fitted scale0: run the scale_fit stage first")
        out = ROOT / "public" / f"marble-{self.name}-finetuned"
        if Path(f"{out}.spz").exists() and "finetune" not in (self.a.force or ""):
            raise RuntimeError(f"{out}.spz already exists; pass --force finetune to redo it")
        export = self.ctx / "ft-export"
        shutil.rmtree(export, ignore_errors=True)
        run([PY, "scripts/finetune_export_bedroom.py", "--world", str(self.world_spz()),
             "--cameras", str(self.cameras()), "--anchors", str(self.ctx / "pi3x" / "anchors.npz"),
             "--anchor-samples", self.anchor_samples(), "--clip", str(self.clip),
             "--scale0", f"{s:.6f}", "--mask-dilate", str(self.a.ft_mask_dilate),
             "--out", str(export)], self.ctx / "finetune_export.log", attempts=1)
        m_per_unit = self.m_per_unit()
        # Native-unit budgets rescaled so the metric budget matches the corridor's: 1 unit = 2.608 m
        # there, clamp 0.52 m, tether 0.26 m, lr 5.2e-3 m/step.
        clamp, sigma, lr = 0.52 / m_per_unit, 0.26 / m_per_unit, 5.2e-3 / m_per_unit
        n = self.n_cameras()
        poses, mid = f"0,{(n - 1) // 2},{n - 1}", (n - 1) // 2
        say(f"   {n} cameras, 1 native unit = {m_per_unit:.3f} m -> clamp {clamp:.3f} sigma "
            f"{sigma:.3f} lr {lr:.2e}, figure poses {poses}")
        box, keyfile = self.a.gpu_box, self.a.gpu_key
        ssh = ["ssh", *(["-i", keyfile] if keyfile else [])]
        remote = f"~/w/ft-{self.name}"
        run(["rsync", "-az", "-e", " ".join(ssh), f"{export}/", f"{box}:{remote}/"],
            self.ctx / "finetune_rsync.log")
        run(["rsync", "-az", "-e", " ".join(ssh), "scripts/finetune_gsplat.py", f"{box}:{remote}/"],
            self.ctx / "finetune_rsync_trainer.log")
        run([*ssh, box, f"cd {remote} && ~/venv/bin/python finetune_gsplat.py "
                        f"--data {remote} --out {remote}/run5 --iters {self.a.ft_iters} "
                        f"--lambda-depth 1.0 --lr-means {lr:.4e} --clamp-pos {clamp:.4f} "
                        f"--sigma-pos {sigma:.4f} --reg-observed-scale 0.05 "
                        f"--prune --prune-every 200 --prune-start 400 --prune-depth-start 1500 "
                        f"--prune-depth-cm 8 --densify --max-splats 2400000 "
                        f"--lambda-opacity-entropy 0.01 "
                        f"--poses {poses} --offpath-frame {mid} --m-per-unit {m_per_unit:.4f}"],
            self.ctx / "finetune_train.log")
        runs = self.ctx / "ft-run5"
        run(["rsync", "-az", "-e", " ".join(ssh), f"{box}:{remote}/run5/", str(runs) + "/"],
            self.ctx / "finetune_fetch.log")
        run([PY, "scripts/finetune_figures.py", "--run", str(runs), "--data", str(export),
             "--tag", f"finetune-{self.name}", "--poses", poses, "--offpath-frame", str(mid),
             "--world-label", f"Marble {self.name}-clean (as is)"], self.ctx / "finetune_figures.log")
        run([PY, "scripts/finetune_import.py", "--npz", str(runs / "finetuned.npz"),
             "--world", str(self.world_spz()), "--out", str(out)], self.ctx / "finetune_import.log")
        stats = runs / "stats.json"
        if stats.exists():
            d = json.loads(stats.read_text())
            say(f"   held-out PSNR {d.get('psnr_before')} -> {d.get('psnr_after')}, "
                f"depth ratio {d.get('depth_ratio_before')} -> {d.get('depth_ratio_after')}")

    def m_per_unit(self) -> float:
        """Metres per native unit, from the exporter's own anchor-cloud floor-to-camera estimate."""
        m = re.search(r"1 native unit = ([0-9.]+) m",
                      (self.ctx / "finetune_export.log").read_text())
        if not m:
            raise RuntimeError("the exporter did not print its metres-per-native-unit estimate; "
                               f"see {self.ctx}/finetune_export.log")
        return float(m.group(1))

    # ---- summary ------------------------------------------------------------
    def placement(self) -> dict:
        """The headline output: what fourd.html needs.

        Marble writes the metric scale at assets.splats.semantics_metadata, NOT at the top level --
        reading the wrong path is why this printed nothing for six consecutive runs.
        """
        out = {}
        for suffix in ("clean", "image", "multi"):
            wj = MARBLE_DIR / f"{self.name}-{suffix}-world.json"
            if not wj.exists():
                continue
            w = json.loads(wj.read_text())
            meta = ((w.get("assets") or {}).get("splats") or {}).get("semantics_metadata") or {}
            msf = meta.get("metric_scale_factor") or w.get("metric_scale_factor")
            gpo = meta.get("ground_plane_offset") or w.get("ground_plane_offset")
            if msf:
                out.update(metricScaleFactor=msf, groundPlaneOffset=gpo,
                           height=1.7 * msf, marbleFloor=-gpo / msf,
                           world=w.get("world_id") or w.get("id"),
                           worldUrl=w.get("world_marble_url"))
            break
        ply = ROOT / "public" / "worlds" / f"{self.name}-4d" / "person" / "frame_000.ply"
        if ply.exists():
            body, feet = body_stats(ply)
            out.update(bodyHeight=body, feetY=feet)
            if out.get("height"):
                out["formulaScale"] = out["height"] / body
                out["floorMatchedScale"] = -out["marbleFloor"] / abs(feet)
        fit = self.state.data["stages"].get("_scale") or {}
        if fit.get("scale0"):
            out["fittedScale0"] = fit["scale0"]
            out["depthRatios"] = fit.get("ratios")
            out["scaleGateOk"] = fit.get("gateOk")
            if out.get("feetY") is not None:      # the shipped floor: feetY * fitted scale0
                out["floor"] = out["feetY"] * fit["scale0"]
        return out

    def summary(self, wall):
        p = self.placement()
        self.state.record("_placement", **p)
        print("\n" + "=" * 78)
        print(f"{self.name}: {self.info['duration']:.1f} s source, {self.info['frames']} frames "
              f"@ {self.info['fps']:.2f} fps -> {self.a.fps} fps preset")
        print("-" * 78)
        print(f"{'stage':<18}{'wall s':>9}  status")
        for s in self.stages:
            st = self.state.data["stages"].get(s, {})
            if st:
                print(f"{s:<18}{st.get('seconds', 0):>9.1f}  {st.get('status')}"
                      + (f"  {st.get('error', '')[:44]}"
                         if st.get("status") in ("failed", "gate") else ""))
        print(f"{'TOTAL WALL':<18}{wall:>9.1f}")
        print("-" * 78)
        if p:
            print("placement: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                              for k, v in p.items()}))
        world = "<world>.spz"
        for cand in (f"marble-{self.name}-finetuned.spz", f"marble-{self.name}-clean.spz",
                     f"marble-{self.name}-image.spz"):
            if (ROOT / "public" / cand).exists():
                world = "/" + cand
                break
        manifest = ROOT / "public" / "worlds" / f"{self.name}-4d" / "people.json"
        people_arg = ""
        if manifest.exists():
            doc = json.loads(manifest.read_text())
            print("people: " + json.dumps([dict(id=q["id"], track=q["track"], frames=q["frames"],
                                                bodyHeightUnits=round(q["bodyHeightUnits"], 4))
                                           for q in doc["people"]]))
            if doc["peopleCount"] > 1:
                people_arg = f"&people=/worlds/{self.name}-4d/people.json"
        base = (f"\n  http://127.0.0.1:5399/fourd.html?world={world}"
                f"&person=/worlds/{self.name}-4d/person/sequence.json{people_arg}"
                f"&video={self.video_url}")
        if p.get("floor") is not None:
            # fourd.html reads scale before height, and the fitted scale0 is the number to use.
            print(f"{base}&floor={p['floor']:.4f}&scale={p['fittedScale0']:.4f}")
            ps = self.state.data["stages"].get("_placement_solve")
            if ps:
                sc, res = ps["sizeCheck"], ps["residualCm"]
                print(f"  place: add &place=1 for the per-sample solve "
                      f"(contact residual "
                      + ", ".join(f"{res['before'][k]['rms']:.1f}->{res['after'][k]['rms']:.1f}"
                                  for k in res['before'])
                      + f" cm RMS; 1 u = {ps['metresPerWorldUnit']:.3f} m; size check "
                      + ("PASS)" if sc["pass_"] else "FAILED - DO NOT SHIP)"))
            if p.get("scaleGateOk") is False:
                print(f"  CAVEAT: depth ratios {[round(r, 3) for r in p['depthRatios']]} -- the "
                      f"avatar reads slightly large early in the clip and small late.")
        elif p.get("height") is not None:
            print(f"{base}&floor={p['marbleFloor']:.4f}&height={p['height']:.4f}")
            print("  UNFITTED: that height is 1.7 x Marble's metric_scale_factor, which has been "
                  "wrong by 2.9x. Run the scale_fit stage before shipping this preset.")
        else:
            print("\n  no Marble world yet: run the marble_video stage with a key to get the URL")
        if p.get("worldUrl"):
            print(f"  world: {p['worldUrl']}")
        print("=" * 78)

    # ---- scheduler ----------------------------------------------------------
    def go(self):
        a = self.a
        named = (set(a.only.split(",")) if a.only else set()) | (set(a.force.split(",")) if a.force else set())
        unknown = sorted(named - set(self.stages))
        if unknown:                     # never silently drop a stage the caller asked for
            sys.exit(f"unknown stage(s) {unknown} for --marble {self.marble}"
                     f"{' --people' if self.multi else ''}. This run's stages: {self.stages}")
        wanted = set(a.only.split(",")) if a.only else set(self.stages)
        if a.skip_finetune:
            wanted.discard("finetune")
        forced = set(a.force.split(",")) if a.force else set()
        todo = {s for s in wanted if s in self.stages and (s in forced or not self.state.done(s))}
        for s in todo:
            self.state.record(s, status="pending")
        say(f"marble mode: {self.marble}"
            + (f" (reusing world {a.reuse_world}, no credits)" if a.reuse_world else
               ", ONE generation = 1600 credits" if self.marble in ("video", "image") else
               ", TWO generations = 3200 credits" if self.marble == "both" else ""))
        say(f"stages to run: {sorted(todo)} (already done: {sorted(wanted - todo)})")
        running, failed, blocked, vacant, gated, t0 = {}, set(), set(), set(), set(), time.time()

        def launch(stage):
            def body():
                start = time.time()
                say(f"-> {stage}")
                try:
                    self.stage_fn(stage)()
                    self.state.record(stage, status="ok", seconds=time.time() - start)
                    say(f"<- {stage} ok ({time.time() - start:.0f}s)")
                except SkipStage as e:
                    vacant.add(stage)
                    self.state.record(stage, status="skipped", seconds=time.time() - start, error=str(e))
                    say(f"-- {stage} skipped: {e}")
                except GateStop as e:
                    gated.add(stage)
                    self.state.record(stage, status="gate", seconds=time.time() - start, error=str(e))
                    say(f"|| {stage} STOPPED AT THE GATE: {e}")
                except Exception as e:
                    self.state.record(stage, status="failed", seconds=time.time() - start, error=str(e))
                    failed.add(stage)
                    say(f"!! {stage} FAILED ({time.time() - start:.0f}s): {e}")
            th = threading.Thread(target=body, daemon=True)
            th.start()
            return th

        def status(dep):
            # a slot skipped in an earlier run (identity audit, no track) is recorded as "skipped",
            # not "ok", and must not block a resume the way a failed one does
            if dep in vacant or self.vacant(dep) or self.state.data["stages"].get(dep, {}).get("status") == "skipped":
                return "ok"
            if dep in gated:
                return "gate"
            if dep in failed or dep in blocked:
                return "dead"
            if dep in running or dep in pending:
                return "wait"
            return "ok" if self.state.done(dep) else "dead"

        pending = set(todo)
        while pending or running:
            for stage in sorted(pending):
                states = [status(d) for d in self.deps[stage]]
                if "gate" in states:
                    pending.discard(stage)
                    gated.add(stage)
                    self.state.record(stage, status="gate", seconds=0.0,
                                      error="waiting on the clean-review gate")
                    say(f"|| {stage} held: waiting on the clean-review gate")
                elif "dead" in states:
                    pending.discard(stage)
                    blocked.add(stage)
                    dead = [d for d in self.deps[stage] if status(d) == "dead"]
                    self.state.record(stage, status="failed", seconds=0.0,
                                      error=f"dependency did not finish: {dead}")
                    say(f"!! {stage} NOT RUN: dependency did not finish: {dead}")
                elif "wait" not in states:
                    pending.discard(stage)
                    running[stage] = launch(stage)
            for stage, th in list(running.items()):
                if not th.is_alive():
                    th.join()
                    del running[stage]
            time.sleep(1)
        wall = time.time() - t0
        self.state.record("_run", wallSeconds=wall)
        self.summary(wall)
        if failed or blocked:
            print(f"\nFAILED: {sorted(failed | blocked)}. Logs in {self.ctx}. Fix, then re-run the "
                  f"same command (finished stages are not redone).")
            return 1
        if gated:
            print(f"\nSTOPPED AT THE CLEAN-REVIEW GATE before spending any credits. Look at the "
                  f"frames listed above, then re-run the same command with --gate-pass.")
            return 2
        return 0


def cut_check(a) -> dict:
    """The shot report for this clip, cached under .context/run/<name>/cuts.json.

    First thing the run does, before a frame is cleaned or a credit is spent: ffmpeg only, no GPU.
    Re-reads the cache when the source file has not changed, so a resumed run does not re-score
    the shots and cannot change its mind about which one it picked half way through.
    """
    clip = Path(a.clip).resolve()
    out = ROOT / ".context" / "run" / a.name / "cuts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    st = clip.stat()
    if out.exists():
        try:
            doc = json.loads(out.read_text())
            if doc.get("_source") == [str(clip), st.st_size, int(st.st_mtime)]:
                say(f"cut check: {doc['cutCount']} cut(s) (cached from {out})")
                return doc
        except Exception:
            pass
    cmd = [PY, "scripts/shot_cuts.py", "--video", str(clip), "--json", str(out),
           "--threshold", str(a.cut_threshold)]
    if a.shot is not None or a.all_shots:
        cmd.append("--no-score")            # the choice is already made; do not pay to rank
    say("cut check: looking for cuts before anything is spent")
    p = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, **LOCAL_ENV),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout.strip().splitlines():
        say(f"   {line}")
    if not out.exists():
        raise SystemExit(f"the cut check failed (exit {p.returncode}); see the output above")
    doc = json.loads(out.read_text())
    doc["_source"] = [str(clip), st.st_size, int(st.st_mtime)]
    out.write_text(json.dumps(doc, indent=1))
    return doc


def resolve_shots(a) -> list[tuple[str, Path, dict | None]]:
    """(name, clip, shot) for each pipeline this invocation will run.

    One shot in the clip: the clip itself, untouched, exactly as before this stage existed.
    More than one: never the whole clip. The default picks the best shot by the criteria this
    project has always chosen shots by and says so; --shot N names one; --all-shots runs each.
    """
    clip = Path(a.clip).resolve()
    doc = cut_check(a)
    if doc["continuous"] and a.shot is None:
        return [(a.name, clip, None)]
    if a.one_shot:
        say(f"   --one-shot: processing the whole clip across {doc['cutCount']} detected cut(s) "
            f"at {[c['time'] for c in doc['cuts']]} s, as asked. The camera solve will be fitted "
            f"through them; the pose guard after pi3x is the only thing still checking.")
        return [(a.name, clip, None)]
    shots = doc["shots"]
    usable = [s for s in shots if not s["tooShort"]]
    if a.all_shots:
        chosen = usable
        say(f"   --all-shots: {len(chosen)} shot(s) of {doc['shotCount']}, one run each")
    elif a.shot is not None:
        chosen = [s for s in shots if s["index"] == a.shot]
        if not chosen:
            raise SystemExit(f"--shot {a.shot}: this clip has shots 0-{len(shots) - 1}")
        say(f"   --shot {a.shot}: {chosen[0]['start']:.2f}-{chosen[0]['end']:.2f} s")
    else:
        if "best" not in doc:
            raise SystemExit(f"{doc['cutCount']} cut(s) and no shot long enough to reconstruct "
                             f"(longest {max((s['seconds'] for s in shots), default=0):.1f} s). "
                             f"Trim a longer segment by hand, or pass --shot N.")
        chosen = [s for s in shots if s["index"] == doc["best"]]
        say(f"   {doc['cutCount']} cut(s) at {[c['time'] for c in doc['cuts']]} s. NOT processing "
            f"across them.")
        say(f"   chose shot {doc['best']} ({chosen[0]['start']:.2f}-{chosen[0]['end']:.2f} s): "
            f"{doc['bestWhy']}")
        say(f"   --shot N processes a different one, --all-shots processes every shot separately, "
            f"and every shot's score is in {ROOT / '.context/run' / a.name / 'cuts.json'}")
    plan = []
    for shot in chosen:
        dest = ROOT / "public" / "clips" / f"{a.name}-shot{shot['index']:02d}.mp4"
        name = a.name if len(chosen) == 1 and not a.all_shots else f"{a.name}-shot{shot['index']:02d}"
        if not dest.exists():
            subprocess.run([PY, "scripts/shot_cuts.py", "--report",
                            str(ROOT / ".context" / "run" / a.name / "cuts.json"),
                            "--trim", str(shot["index"]), "--out", str(dest)],
                           cwd=ROOT, env=dict(os.environ, **LOCAL_ENV), check=True)
        say(f"   shot {shot['index']} -> {dest} (run name {name})")
        plan.append((name, dest, shot))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--marble-key", default="WLT_API_KEY",
                    help="name of the env var holding the Marble key (never the key itself)")
    ap.add_argument("--marble", default="video", choices=MARBLE_MODES,
                    help="which world(s) to generate. video (default) = ONE 1600-credit world from "
                         "the cleaned clip; image = one from the first cleaned frame; both = TWO "
                         "worlds, 3200 credits; none = spend nothing")
    ap.add_argument("--reuse-world", metavar="ID",
                    help="adopt an existing Marble world instead of generating one: fetches its "
                         "metadata and splats, spends no credits")
    ap.add_argument("--gate-pass", "--no-gate", action="store_true", dest="gate_pass",
                    help="continue past the clean-review gate: pass it on the second run once you "
                         "have looked at the frames, or on the first for an unattended run")
    ap.add_argument("--poll-attempts", type=int, default=6,
                    help="network retries around the Marble POLL. Never around the submit -- a "
                         "resubmit after the operation exists buys a second 1600-credit world")
    ap.add_argument("--scale0", type=float, help="skip the fit and use this SfM-to-Marble scale")
    ap.add_argument("--scale-probes", type=int, default=5, help="depth-ratio probes for the scale fit")
    ap.add_argument("--ruler-tol", type=float, default=0.15,
                    help="how far the world ruler and the avatar ruler may disagree before the "
                         "anchors stage fails the run")
    ap.add_argument("--ruler-vlm", action="store_true",
                    help="also ask the VLM to name a standard-size object for the world ruler "
                         "(costs one image call per sampled frame)")
    ap.add_argument("--allow-weak-anchor", action="store_true",
                    help="record the anchor checks instead of failing on them. A clip run this way "
                         "has its metres on the avatar assumption alone; label them as avatar "
                         "metres wherever they are quoted.")
    ap.add_argument("--allow-bad-size", action="store_true",
                    help="write placement.json even when the size check says the person cannot be "
                         "1.70 m in that room. Read scripts/place_solve.py's message first: on gym "
                         "this would have shipped a 2.94 m man.")
    ap.add_argument("--allow-scale-spread", action="store_true",
                    help="ship the geometric-mean fit when no single scale puts every sampled depth "
                         "ratio inside 0.95-1.05 (gym, atrium and selfie needed this)")
    ap.add_argument("--ft-mask-dilate", type=int, default=16,
                    help="person-mask dilation for the fine-tune export; raise it (gym used 60) "
                         "when fragments of the subject get baked into the world")
    ap.add_argument("--ft-iters", type=int, default=4000)
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--skip-marble", action="store_true", help="alias for --marble none")
    ap.add_argument("--dilate", type=int, default=20)
    ap.add_argument("--moved-mask", action="store_true",
                    help="clean everything that MOVED, not just the person: cast shadow, carried "
                         "object, passer-by. Off by default, so every existing run is unchanged.")
    ap.add_argument("--bottom-extra", type=int, default=40)
    ap.add_argument("--lama-px", type=int, default=960)
    ap.add_argument("--person-frame", type=int, help="source frame for the LHM avatar (default: middle)")
    ap.add_argument("--person-method", default="maskrcnn", choices=["segformer", "maskrcnn"])
    ap.add_argument("--all-people", action="store_true",
                    help="track every person in the clip and build one avatar per track (up to 4); "
                         "degrades to the single-person result when only one track is found")
    ap.add_argument("--people", type=int, help="like --all-people with an explicit cap on the person count")
    ap.add_argument("--det-thresh", type=float, default=0.15,
                    help="MultiHMR detection threshold for the tracking stage")
    ap.add_argument("--shot", type=int, metavar="N",
                    help="process shot N of a clip with cuts in it instead of the best-scoring one "
                         "(indices and scores are printed by the cut check, and in cuts.json)")
    ap.add_argument("--all-shots", action="store_true",
                    help="process every shot of a multi-shot clip separately, as <name>-shotNN")
    ap.add_argument("--one-shot", action="store_true",
                    help="the detected cuts are not cuts: process the whole clip anyway. The solve "
                         "will be fitted across them, so read the cut list first")
    ap.add_argument("--cut-threshold", type=float, default=0.10,
                    help="scene score above which a frame is a candidate cut (see shot_cuts.py)")
    ap.add_argument("--object", action="append", metavar="ID:MOTION:PROMPT[:ROI]",
                    help="an object to segment, solve and package; repeatable. MOTION is "
                         "attached|free|handoff|worldDynamic, PROMPT is what the thing is in words, "
                         "ROI is joint=TRACK,JOINT,RADIUS or box=x0,y0,x1,y1. See docs/objects.md.")
    ap.add_argument("--no-objects", action="store_true",
                    help="skip the objects stage (by default it looks for thrown objects in every clip)")
    ap.add_argument("--no-shape", action="store_true",
                    help="objects: package a proxy instead of generating a shape on Modal")
    ap.add_argument("--only", help="comma-separated subset of stages")
    ap.add_argument("--force", help="comma-separated stages to redo even if state.json says ok")
    ap.add_argument("--gpu-box", default=os.environ.get("WANDER_GPU_BOX"), help="configured fine-tune host (or WANDER_GPU_BOX)")
    ap.add_argument("--gpu-key", default=os.environ.get("WANDER_GPU_KEY"), help="optional SSH identity (or WANDER_GPU_KEY)")
    ap.add_argument("--no-publish", action="store_true",
                    help="keep this run local; normally outputs and run artifacts are saved to private S3")
    a = ap.parse_args()
    if a.marble_key and os.environ.get(a.marble_key, "").startswith("-"):
        sys.exit("--marble-key takes the NAME of an env var, not a key")
    if a.shot is not None and a.all_shots:
        sys.exit("--shot and --all-shots ask for different things; pass one")
    rc = 0
    try:
        plan = resolve_shots(a)
        for name, clip, shot in plan:
            if len(plan) > 1:
                say(f"===== {name} =====")
            rc = max(rc, Pipeline(a, clip, name, shot).go())
    finally:
        if not a.no_publish:
            say("Saving viewer assets and run artifacts to private S3...")
            command = ["node", "scripts/publish-runs.mjs"]
            if os.environ.get("WANDER_EVIDENCE_DIR"):
                command += ["--evidence-dir", os.environ["WANDER_EVIDENCE_DIR"]]
            try:
                result = subprocess.run(command, cwd=ROOT)
                if result.returncode:
                    say("S3 publish FAILED; local results are intact. Retry: npm run runs:publish")
                    rc = max(rc, 3)
            except OSError as error:
                say(f"S3 publish could not start ({type(error).__name__}); retry npm run runs:publish")
                rc = max(rc, 3)
    sys.exit(rc)


if __name__ == "__main__":
    main()
