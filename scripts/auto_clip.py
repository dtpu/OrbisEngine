#!/usr/bin/env python3
"""video in -> best segment -> cleaned -> cameras -> world -> people -> packaged -> report.

A THIN orchestrator. It owns no measurement and no inference of its own: every step shells out to
a script that already exists and is already tested, and this file's whole job is to make the
operator judgement that used to live in a chat window into explicit, named, logged decisions.

  uv run --locked python scripts/auto_clip.py run --clip <video> --name <run-name> --dry-run
  uv run --locked python scripts/auto_clip.py reconcile --run .context/run/<name> --stage pi3x
  uv run --locked python scripts/auto_clip.py preflight

Everything it decides, and why, lands in `.context/run/<name>/auto-log.json`
(schema `wander.auto-clip/1`): the rule or judge opinion behind each choice, its inputs, the exact
commands run, their outcomes, the costs read from the paid ledger, and a final per-clip report.

`--dry-run` prints the exact argv of every command and every decision and runs nothing at all --
no ffmpeg, no Modal, no OpenAI. `--resume` reuses the decisions an earlier invocation already
recorded instead of paying for them again.

WHAT THIS FILE IS NOT. It is not a quality gate and not a judge. `docs/quality-rubric.md` makes an
automated judge triage, never proof, so a judge verdict here selects a retry or stops a run; it
never certifies a reconstruction. The segment judge's opinion is recorded with `applied: false` and
changes no ranking, exactly as `docs/segment-selection.md` requires. The report separates what was
observed in the footage from what the pipeline invented, and never claims fitted placement when the
scale gate fell back.

The runner is injected (`runner=`) so the tests drive the whole playbook with fakes and spend
nothing. Keep this module free of heavy imports: stdlib only at module scope, so `--dry-run` costs
nothing but a process start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PY = os.environ.get("WANDER_PYTHON", sys.executable)
SCHEMA = "wander.auto-clip/1"

# ---- named decision constants ----------------------------------------------------------------
# Every rule the playbook applies has a name and a number here. Nothing in this file branches on a
# clip name, a run name or a file path.

#: Share of sampled frame pixels labelled person above which the clip is a crowd and the cleaner
#: must not erase the background people. This is the same idea, and the same number, as
#: `CROWD_PERSON_FRACTION` in worker/stages/dense_pi3x.py (which switches its own anchor mask mode
#: on it); it is copied rather than imported because that module pulls in the GPU worker stack.
#: scripts/test_auto_clip.py asserts the two stay equal.
CROWD_PERSON_FRACTION = 0.25

#: Three executions per source/stage/window, from scripts/stage_attempts.py MAX_EXECUTIONS and
#: docs/quality-rubric.md ("two quality retries after the first execution"). The ledger enforces
#: it; this orchestrator refuses to plan past it so the refusal is a decision, not a crash.
MAX_STAGE_EXECUTIONS = 3

#: One judge-driven re-clean per window. The rubric's cap is three executions; the first clean and
#: one mapped retry leave one execution in hand for a human-diagnosed attempt.
CLEAN_JUDGE_RETRIES = 1

#: One split retry after a camera teleport, and one next-best-window retry after the solve refuses
#: for non-overlapping anchors. Both consume a ledger execution.
SOLVE_RETRIES = 1

#: The shortest sub-window worth re-solving after a teleport split, and the shortest window the
#: selector will choose at all (scripts/select_segment.py MIN_WINDOW_SECONDS).
MIN_SPLIT_SECONDS = 6.0

#: Default number of shots `--mode sequence` plans, longest usable first.
SEQUENCE_SHOT_CAP = 4

#: Default cap on reconstructed people (scripts/run_clip.py `--all-people` builds at most 4).
PEOPLE_CAP = 4

#: A frame-filling subject reported only in the late or early part of the window is a shortening
#: action: the window is cut at this share of its length, keeping the half the judge did not flag.
SUBJECT_SHORTEN_SHARE = 0.5

#: judge_clean answers `smearFraction`, "roughly how much of the frame is smeared". A repaired
#: region covering half the frame is the judge saying the subject filled it.
SUBJECT_FILL_FRACTION = 0.5

#: scripts/judge_clean.py exit codes.
JUDGE_PASS, JUDGE_RETRY, JUDGE_FAIL, JUDGE_ERROR = 0, 10, 11, 2

#: scripts/select_segment.py top-K for the judge hook (docs/segment-selection.md: "use --top 5").
SELECT_TOP_K = 5

#: Critical clean-judge finding that means the thing left behind is not a body. run_clip.py takes
#: `--moved-mask` for exactly that case.
NON_HUMAN_SUBJECT_ACTION = "enable_moved_mask"

#: Reconciliation evidence classes. A saved log matching a refusal pattern, or a recovery manifest
#: in a terminal non-success state, is provable failure. Anything else is ambiguous and stops.
PROVIDER_REFUSAL_PATTERNS = (
    r"exceeded its spend limit",
    r"workspace .{0,80}spend limit",
    r"insufficient (credit|funds|quota)",
    r"\bnot authorized\b",
    r"\bunauthorized\b",
    r"invalid token",
    r"token (is )?missing",
    r"quota exceeded",
)
WORKER_REFUSAL_PATTERNS = (
    r"Stage public LHM prior assets before GPU inference",
    r"Stage the public model cache before inference",
    r"Stage the public GFPGANv1\.3 checkpoint before inference",
    r"GFPGAN must be installed in the native worker image",
    r"Runtime model path conflicts with the staged cache",
    r"GFPGAN package weight conflicts with the staged cache",
)
#: Text that proves inference actually started, so a charge is possible and the evidence is no
#: longer "refused before inference". Presence of any of these makes the claim ambiguous.
INFERENCE_STARTED_PATTERNS = (
    r"estimatedComputeUSD",
    r"inferenceWallSeconds",
    r"peakVRAMGB",
    r"\bapp-[A-Za-z0-9]{8,}",
    r"\bap-[A-Za-z0-9]{8,}",
)
FAILED_MANIFEST_STATUSES = ("failed", "partial")

#: Messages the solve refuses with, and what the playbook does about each.
TELEPORT_MARKER = "camera teleport at frame"
NO_OVERLAP_MARKERS = ("do not agree on the same scene", "Select a shorter segment")
#: Person-stage refusals that make one subject unresolved rather than the run failed.
PREP_REFUSAL_MARKER = "never fully visible"
POSE_REFUSAL_MARKER = "No source person pose detected"
#: scripts/identity_audit.py sets this reason when its swap control was missed: the audit cannot
#: tell two look-alike subjects apart, so its "ok" means nothing.
IDENTITY_NO_POWER_MARKER = "has no power"

#: The paid ledger this project shares across candidates and worktrees (docs/paid-recovery.md).
DEFAULT_LEDGER = ROOT / ".context/pipeline-attempts.json"

#: General statements about which parts of a delivered replay were observed and which were made up.
#: Gated on the stages that actually ran, never on a clip.
OBSERVED_BY_STAGE = {
    "clean": "the source pixels of the chosen window, at the recorded source offsets",
    "solve": "the camera path, solved from the window's own frames",
    "tracks": "each person's on-screen box and mask, from the tracker's own detections",
    "people": "each reconstructed person's motion, solved from the frames they appear in",
}
INVENTED_BY_STAGE = {
    "clean": "the background behind every removed person, inpainted, not observed",
    "world": "all world geometry and appearance outside the observed camera views, generated",
    "people": "each avatar's unobserved side and its appearance beyond the observed view",
}


# ---- the runner -------------------------------------------------------------------------------


@dataclass
class Result:
    """What a command did. `returncode is None` means it was planned and never run."""

    argv: list[str]
    returncode: int | None = 0
    stdout: str = ""

    @property
    def ran(self) -> bool:
        return self.returncode is not None


def shell_runner(argv, *, cwd=ROOT, log=None, env=None) -> Result:
    """Run one command, tee its combined output into `log`, and return the result.

    Never raises on a non-zero exit: the playbook branches on exit codes (judge_clean returns 10
    for retry and 11 for fail), so a failure is data here, not an exception.
    """
    argv = [str(part) for part in argv]
    completed = subprocess.run(
        argv,
        check=False,
        cwd=str(cwd),
        env=dict(os.environ, **(env or {})),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if log is not None:
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            handle.write(f"$ {shlex.join(argv)}\n")
            handle.write(completed.stdout or "")
    return Result(argv=argv, returncode=completed.returncode, stdout=completed.stdout or "")


# ---- the decision log -------------------------------------------------------------------------


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def jsonable(value):
    """Paths and sets become strings and lists, so a decision is always writable."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def atomic_json(path: Path, document) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=1) + "\n")
    os.replace(temporary, path)


class DecisionLog:
    """One JSON file per clip holding every decision, command, cost and the final report."""

    def __init__(self, path, *, name, clip, dry_run=False, resume=False, mode="best-shot"):
        self.path = Path(path)
        previous = read_json(self.path) if resume else None
        self.prior = {
            entry["id"]: entry
            for entry in (previous or {}).get("decisions", [])
            if isinstance(entry, dict) and entry.get("id")
        }
        self.doc = {
            "schema": SCHEMA,
            "name": name,
            "clip": str(clip),
            "sourceSha256": (previous or {}).get("sourceSha256"),
            "mode": mode,
            "dryRun": bool(dry_run),
            "resume": bool(resume),
            "startedAt": (previous or {}).get("startedAt") or now(),
            # The log is append-only: a resume adds to the history instead of replacing it, so an
            # earlier attempt's reasoning survives.
            "decisions": list((previous or {}).get("decisions") or []),
            "commands": list((previous or {}).get("commands") or []),
            "costs": {},
            "report": {},
        }
        if previous:
            self.doc["resumedAt"] = now()

    # -- decisions
    def previous(self, ident):
        return self.prior.get(ident)

    def latest(self, ident):
        """The most recent decision with this id, because the log is append-only."""
        for entry in reversed(self.doc["decisions"]):
            if entry.get("id") == ident:
                return entry
        return None

    def decide(self, ident, *, step, rule, inputs, choice, because, evidence=None, resumed=False):
        entry = {
            "id": ident,
            "step": step,
            "rule": rule,
            "inputs": jsonable(inputs),
            "choice": jsonable(choice),
            "because": because,
            "evidence": [str(item) for item in (evidence or [])],
            "resumed": bool(resumed),
            "at": now(),
        }
        self.doc["decisions"].append(entry)
        self.save()
        return entry

    def command(self, decision_id, result: Result, *, log=None):
        entry = {
            "decision": decision_id,
            "argv": [str(part) for part in result.argv],
            "cwd": str(ROOT),
            "returncode": result.returncode,
            "log": str(log) if log else None,
            "dryRun": not result.ran,
        }
        self.doc["commands"].append(entry)
        self.save()
        return entry

    def save(self):
        atomic_json(self.path, self.doc)


# ---- the paid ledger --------------------------------------------------------------------------


def ledger_costs(ledger_path, *, source_sha256=None, candidates=()):
    """Costs and remaining executions read from the shared ledger. Read-only, never writes."""
    document = read_json(ledger_path) or {}
    attempts = document.get("attempts") or []
    names = set(candidates)
    stages, total, seconds = {}, 0.0, 0.0
    for attempt in attempts:
        claim = attempt.get("claim") or {}
        if source_sha256 and claim.get("sourceSha256") != source_sha256:
            continue
        if names and claim.get("candidate") not in names:
            continue
        stage = claim.get("stage") or claim.get("operation") or "unknown"
        row = stages.setdefault(
            stage, {"executions": 0, "estimatedComputeUSD": 0.0, "statuses": {}}
        )
        row["executions"] += 1
        status = ((attempt.get("events") or [{}])[-1]).get("status", "unknown")
        row["statuses"][status] = row["statuses"].get(status, 0) + 1
        for event in attempt.get("events") or []:
            costs = event.get("costs") or {}
            value = costs.get("estimatedComputeUSD")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row["estimatedComputeUSD"] += float(value)
                total += float(value)
            worker = costs.get("reportedWorkerSeconds")
            if isinstance(worker, (int, float)) and not isinstance(worker, bool):
                seconds += float(worker)
    for stage, row in stages.items():
        row["estimatedComputeUSD"] = round(row["estimatedComputeUSD"], 6)
        row["executionsRemaining"] = max(0, MAX_STAGE_EXECUTIONS - row["executions"])
    return {
        "ledger": str(ledger_path),
        "sourceSha256": source_sha256,
        "candidates": sorted(names),
        "byStage": stages,
        "estimatedComputeUSD": round(total, 6),
        "reportedWorkerSeconds": round(seconds, 3),
        "note": (
            "estimatedComputeUSD is the provider's estimate, not an invoice, and this ledger does "
            "not cover OpenAI, Marble or direct worker invocations (docs/paid-recovery.md)."
        ),
    }


def merge_flags(base, extra):
    """The judge's recommended change wins over the base flag it replaces; the rest survives.

    `--moved-mask` and `--people-mask` are mutually exclusive in run_clip.py (its `clean_flags()`
    drops the mask mode once the moved mask is on), so the base mask mode is dropped when the
    judge asks for either.
    """
    base, extra = list(base), list(extra)
    mask_flags = {"--people-mask", "--moved-mask"}
    replaced = {item for item in extra if item.startswith("--")}
    if replaced & mask_flags:
        replaced |= mask_flags
    merged, index = [], 0
    while index < len(base):
        item = base[index]
        takes_value = index + 1 < len(base) and not base[index + 1].startswith("--")
        if item.startswith("--") and item in replaced:
            index += 2 if takes_value else 1
            continue
        merged.append(item)
        index += 1
    return merged + extra


def ledger_refused(text: str) -> str | None:
    """The clean surface of a ledger refusal inside run_clip.py's output, or None."""
    for line in (text or "").splitlines():
        if "was not submitted:" in line:
            return line.strip().lstrip("! ")
        if "execution allowance exhausted" in line or "source windows exhausted" in line:
            return line.strip().lstrip("! ")
    return None


# ---- reconciliation ---------------------------------------------------------------------------


def classify_failure(log_text: str, manifests):
    """Is this attempt PROVABLY failed, and on what evidence?

    Returns (verdict, kind, reason) where verdict is 'failed' or 'ambiguous'. Provable failure is
    a provider refusal at admission, a worker-side refusal raised before inference, or a saved
    recovery manifest in a terminal non-success state. Anything else -- including a log that shows
    inference started, which could have been charged -- is ambiguous and must stop for a human.
    """
    text = log_text or ""
    for path, manifest in manifests:
        status = (manifest or {}).get("status")
        if status in FAILED_MANIFEST_STATUSES:
            reason = f"The saved recovery manifest {path} records status {status!r}."
            return "failed", "recovery_manifest", reason
    started = [p for p in INFERENCE_STARTED_PATTERNS if re.search(p, text)]
    provider_reason = (
        "The provider refused at admission: the saved log contains {quote} with no inference "
        "started."
    )
    worker_reason = (
        "The remote worker raised {quote} at its pre-inference check, so no inference ran."
    )
    checks = (
        ("provider_refusal", PROVIDER_REFUSAL_PATTERNS, re.IGNORECASE, provider_reason),
        ("worker_precheck_refusal", WORKER_REFUSAL_PATTERNS, 0, worker_reason),
    )
    for kind, patterns, flags, template in checks:
        for pattern in patterns:
            match = re.search(pattern, text, flags)
            if not match:
                continue
            quote = repr(match.group(0))
            if started:
                reason = (
                    f"The log matches the refusal {quote} but also shows inference had started "
                    f"({started[0]}); a partial charge is possible."
                )
                return "ambiguous", "refusal_after_start", reason
            return "failed", kind, template.format(quote=quote)
    reason = (
        "The saved log holds no provider or worker refusal and no recovery manifest records a "
        "terminal failure; the outcome and any charge are unknown."
    )
    return "ambiguous", "no_evidence", reason


def next_archive(directory: Path) -> Path:
    """`<dir>.failed-attemptN` for the first free N. Renames only; nothing is ever deleted."""
    number = 1
    while (candidate := directory.with_name(f"{directory.name}.failed-attempt{number}")).exists():
        number += 1
    return candidate


def pending_attempts(ledger_path, *, run_name, stage):
    """Pending/unknown claims for this candidate and logical stage, newest first."""
    document = read_json(ledger_path) or {}
    rows = []
    for attempt in document.get("attempts") or []:
        claim = attempt.get("claim") or {}
        if claim.get("candidate") != run_name:
            continue
        if stage and claim.get("stage") != stage and claim.get("operation") != stage:
            continue
        status = ((attempt.get("events") or [{}])[-1]).get("status")
        if status in ("pending", "unknown"):
            rows.append(attempt)
    rows.sort(key=lambda item: (item.get("claim") or {}).get("startedAtEpoch") or 0, reverse=True)
    return rows


# ---- preflight --------------------------------------------------------------------------------

#: What each Modal volume holds and where the GPU stages mount it, so the checklist can name a
#: concrete staging command. The volume names and the paths below are READ from the worker sources
#: at runtime; this table only supplies the human labels and the staging phase.
VOLUME_ROLES = {
    "wander-clean-video-cache": {
        "mount": "/cache",
        "stages": ["clean", "clean_first", "clean_multi"],
        "phases": [
            "clean-motion",
            "clean-motion-repair",
            "clean-motion-repair-v2",
            "clean-instance",
        ],
    },
    "wander-overnight-lhm-cache": {
        "mount": "/cache",
        "stages": ["tracks", "lhm_frozen_NN", "lhm_motion_NN"],
        "phases": ["lhm", "lhm-resume", "dino", "gfpgan"],
    },
    "wander-overnight-motion-cache": {
        "mount": "/huggingface",
        "stages": ["pi3x", "frame_align"],
        "phases": [],
    },
    "wander-weights": {"mount": "/weights", "stages": ["(staging source)"], "phases": []},
}


#: A quoted absolute path in a worker source is a DEMAND on the workspace only when the code
#: around it checks for it or refuses without it. Without this window the image-build paths
#: (`/opt/basicsr`, `/usr/local/cuda/include`) swamp the checklist.
DEMAND_MARKERS = (
    "exists(",
    "is_file(",
    "is_dir(",
    "Stage ",
    "staged",
    "RuntimeError",
    "getsize",
)
DEMAND_WINDOW = 3


def demanded_paths(text, worker):
    """Absolute model paths this worker checks for before inference, with their line context."""
    lines = text.splitlines()
    pattern = re.compile(r"[\"'](/(?:cache|opt|lhm|clean|usr)/[\w./-]+)[\"']")
    found = []
    for number, line in enumerate(lines):
        matches = pattern.findall(line)
        if not matches:
            continue
        window = "\n".join(lines[max(0, number - DEMAND_WINDOW) : number + DEMAND_WINDOW + 1])
        # A module-level constant (LAMA_PT = "/cache/lama/big-lama.pt") is a demand too: it is
        # what the worker hands the model loader.
        constant = re.match(r"[A-Z][A-Z0-9_]*\s*=", line.strip())
        if not constant and not any(marker in window for marker in DEMAND_MARKERS):
            continue
        for quoted in matches:
            found.append(
                {
                    "path": quoted,
                    "kind": "file" if "." in quoted.rsplit("/", 1)[-1] else "directory",
                    "worker": worker,
                    "line": number + 1,
                }
            )
    return found


def worker_requirements(root=ROOT):
    """Volumes, weight paths and staging phases, read out of the worker sources themselves.

    Parsing the sources rather than hard-coding a list is the point: a weight the workers start
    demanding shows up in the checklist without anyone remembering to edit this file.
    """
    files = {
        "clean": root / "worker/modal_clean_video.py",
        "multiperson": root / "worker/modal_multiperson.py",
        "lhm": root / "worker/modal_lhm.py",
        "staging": root / "worker/modal_stage_weights.py",
    }
    volumes, weights, phases = {}, [], []
    for label, path in files.items():
        if not path.is_file():
            continue
        text = path.read_text()
        for name in re.findall(r"modal\.Volume\.from_name\(\s*[\"']([\w-]+)[\"']", text):
            volumes.setdefault(name, set()).add(label)
        if label != "staging":
            weights += demanded_paths(text, str(path.relative_to(root)))
        if label == "staging":
            block = re.search(r"if phase not in \(([^)]*)\)", text)
            if block:
                phases = re.findall(r"[\"']([\w-]+)[\"']", block.group(1))
    seen, unique = set(), []
    for item in weights:
        key = (item["path"], item["worker"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return {
        "volumes": {name: sorted(labels) for name, labels in sorted(volumes.items())},
        "weights": unique,
        "stagingPhases": phases,
        "workerSources": {k: str(v.relative_to(root)) for k, v in files.items() if v.is_file()},
    }


def preflight_checklist(profile, requirements):
    """(rows, commands): what the GPU stages will demand, and how to stage anything missing."""
    rows, commands = [], []
    for volume, sources in requirements["volumes"].items():
        role = VOLUME_ROLES.get(volume, {})
        rows.append(
            {
                "volume": volume,
                "declaredIn": sources,
                "mount": role.get("mount"),
                "demandedBy": role.get("stages", []),
                "stagingPhases": [
                    p for p in role.get("phases", []) if p in requirements["stagingPhases"]
                ],
            }
        )
        commands.append(
            [
                "uv",
                "run",
                "--locked",
                "modal",
                "volume",
                "ls",
                "--profile",
                profile,
                volume,
                role.get("mount", "/"),
                "--json",
            ]
        )
    for phase in requirements["stagingPhases"]:
        commands.append(
            [
                "MODAL_PROFILE=" + profile,
                "uv",
                "run",
                "--locked",
                "python",
                "worker/modal_stage_weights.py",
                "--phase",
                phase,
                "--deadline-epoch",
                "<absolute epoch>",
                "--out",
                f".context/evidence/staging/{phase}.json",
            ]
        )
    return rows, commands


# ---- the orchestrator -------------------------------------------------------------------------


@dataclass
class Person:
    """One tracked subject, as the report will state it."""

    track: int
    label: str = "unknown"
    seen: bool = True
    selected: bool = False
    reconstructed: bool = False
    unresolvedReason: str | None = None
    why: str = ""


@dataclass
class StageOutcome:
    stage: str
    status: str
    reason: str
    evidence: list[str] = field(default_factory=list)


class AutoClip:
    """The playbook. Every method is one numbered step of docs/auto-clip.md."""

    def __init__(self, args, *, runner=shell_runner, root=ROOT):
        self.a = args
        self.root = Path(root)
        self.runner = runner
        self.name = args.name
        self.clip = Path(args.clip)
        self.dry_run = bool(args.dry_run)
        self.ctx = self.root / ".context" / "run" / self.name
        self.ctx.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.root / "public" / "clips"
        self.ledger = Path(getattr(args, "stage_ledger", None) or DEFAULT_LEDGER)
        self.log = DecisionLog(
            self.ctx / "auto-log.json",
            name=self.name,
            clip=self.clip,
            dry_run=self.dry_run,
            resume=bool(args.resume),
            mode=args.mode,
        )
        self.stages: list[StageOutcome] = []
        self.people: list[Person] = []
        self.identity_verified: bool | None = None
        self.identity_reason = "not audited"
        self.segment = None
        self.coverage = None
        self.dropped: list[dict] = []
        self.ran_steps: set[str] = set()
        self.outcome = "planned" if self.dry_run else "incomplete"
        self.stop_reason = None
        self.retry_hypothesis = "no retry has been justified yet"

    # -- plumbing ------------------------------------------------------------------------------

    def say(self, message):
        print(message, flush=True)

    def execute(self, decision_id, argv, *, log=None, paid=False):
        """Run (or, in a dry run, print) one command and record it against a decision."""
        argv = [str(part) for part in argv]
        if self.dry_run:
            kind = "PAID" if paid else "local"
            self.say(f"   [{kind}] {shlex.join(argv)}")
            result = Result(argv=argv, returncode=None)
        else:
            self.say(f"   $ {shlex.join(argv)}")
            result = self.runner(argv, cwd=self.root, log=log)
        self.log.command(decision_id, result, log=log)
        return result

    def step_done(self, ident):
        """True when --resume already has this decision recorded as done."""
        previous = self.log.previous(ident)
        return bool(previous and previous.get("choice", {}).get("status") == "done")

    def reuse(self, ident, step, stage=None):
        previous = self.log.previous(ident)
        if stage:
            # A reused stage still belongs in the report: a resume must not make a finished run
            # look like a stopped one.
            self.stage(stage, "reused", previous.get("because", "recorded by an earlier run"))
        self.log.decide(
            ident,
            step=step,
            rule="resume",
            inputs=previous.get("inputs", {}),
            choice=previous.get("choice", {}),
            because="--resume: this decision was already recorded, so it was not paid for again",
            evidence=previous.get("evidence", []),
            resumed=True,
        )
        self.say(f"== {step}: resumed ({previous.get('because', '')[:70]})")
        return previous.get("choice", {})

    def run_clip_argv(self, only, *extra):
        argv = [
            PY,
            "scripts/run_clip.py",
            "--clip",
            str(self.segment["workClip"]) if self.segment else str(self.clip),
            "--name",
            self.name,
            "--stage-ledger",
            str(self.ledger),
            "--only",
            ",".join(only),
            "--marble",
            self.a.marble,
            "--fps",
            str(self.a.fps),
            "--no-publish",
        ]
        if self.a.source_sha256:
            argv += ["--source-sha256", self.a.source_sha256]
        if not self.a.objects:
            argv.append("--no-objects")
        if not self.a.finetune:
            argv.append("--skip-finetune")
        return argv + [str(part) for part in extra]

    def stage(self, stage, status, reason, evidence=()):
        outcome = StageOutcome(stage, status, reason, [str(e) for e in evidence])
        self.stages.append(outcome)
        return outcome

    # -- 1. admission --------------------------------------------------------------------------

    def admit(self):
        """Probe codec/VFR/letterbox, record the source sha and the exact source offsets."""
        ident = "admission"
        if self.step_done(ident):
            return self.reuse(ident, "admission")
        self.say("== 1 admission")
        argv = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(self.clip),
        ]
        result = self.execute(ident, argv)
        probe = {}
        if result.ran and result.returncode == 0:
            probe = json.loads(result.stdout or "{}")
        video = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
        vfr = None
        if video.get("r_frame_rate") and video.get("avg_frame_rate"):
            vfr = video["r_frame_rate"] != video["avg_frame_rate"]
        source_sha = None if self.dry_run else sha256_file(self.clip)
        self.log.doc["sourceSha256"] = source_sha
        choice = {
            "status": "done",
            "sourceSha256": source_sha,
            "codec": video.get("codec_name"),
            "pixelFormat": video.get("pix_fmt"),
            "width": video.get("width"),
            "height": video.get("height"),
            "variableFrameRate": vfr,
            "rFrameRate": video.get("r_frame_rate"),
            "avgFrameRate": video.get("avg_frame_rate"),
        }
        self.log.decide(
            ident,
            step="admission",
            rule="probe the source before anything reads it; record its sha and exact offsets",
            inputs={"clip": str(self.clip)},
            choice=choice,
            because=(
                "The source bytes and their timing are what every later claim, trim and report is "
                "keyed on; a variable frame rate means the working segment must be re-encoded "
                "rather than stream-copied."
            ),
            evidence=[self.clip],
        )
        return choice

    # -- 2. segment ----------------------------------------------------------------------------

    def choose_segment(self):
        ident = "segment"
        if self.step_done(ident):
            choice = self.reuse(ident, "segment")
            self.segment = choice
            self.coverage = choice.get("coverageNotSelected")
            return choice
        self.say("== 2 segment")
        selection_path = self.ctx / "selection.json"
        sheets = self.ctx / "sheets"
        argv = [
            PY,
            "scripts/select_segment.py",
            "--video",
            str(self.clip),
            "--out",
            str(selection_path),
            "--top",
            str(SELECT_TOP_K),
            "--contact-sheet",
            str(sheets),
        ]
        if self.a.truth:
            argv += ["--truth", str(self.a.truth)]
        if self.a.force_window:
            if not self.a.reason:
                raise SystemExit(
                    "--force-window needs --reason: an override without one is a guess"
                )
            argv += [
                "--force-window",
                str(self.a.force_window[0]),
                str(self.a.force_window[1]),
                "--reason",
                self.a.reason,
            ]
        self.execute(ident, argv, log=self.ctx / "select-segment.log")
        document = read_json(selection_path) or {}
        chosen = document.get("chosen") or []
        if not chosen and not self.dry_run:
            blocked = document.get("blockedBest") or {}
            self.stop_reason = "the selector chose nothing: every candidate window was vetoed" + (
                f" (closest {blocked.get('id')}: {'; '.join(blocked.get('vetoes', []))})"
                if blocked
                else ""
            )
            self.stage("segment", "failed", self.stop_reason, [selection_path])
            self.log.decide(
                ident,
                step="segment",
                rule="a vetoed clip is an answer, not a crash",
                inputs={"selection": str(selection_path)},
                choice={"status": "blocked"},
                because=self.stop_reason,
                evidence=[selection_path],
            )
            return None
        window = (
            chosen[0]
            if chosen
            else {
                "id": "<rank-1>",
                "rank": 1,
                "startSeconds": 0.0,
                "endSeconds": 0.0,
                "score": None,
                "why": "(planned; the selector has not run)",
                "suggestedCrop": {"needed": False},
            }
        )
        opinion = self.segment_judge(sheets, document)
        self.coverage = document.get("coverageNotSelected")
        self.segment = {
            "status": "done",
            "windowId": window.get("id"),
            "rank": window.get("rank"),
            "startSeconds": window.get("startSeconds"),
            "endSeconds": window.get("endSeconds"),
            "score": window.get("score"),
            "why": window.get("why"),
            "forced": bool(document.get("override")),
            "overrideReason": (document.get("override") or {}).get("reason"),
            "suggestedCrop": window.get("suggestedCrop") or {"needed": False},
            "coverageNotSelected": self.coverage,
            "judgeOpinion": opinion,
            "selection": str(selection_path),
            "personPixelFractionMedian": (
                ((window.get("features") or {}).get("people") or {}).get(
                    "personPixelFractionMedian"
                )
            ),
            "alternates": [
                {
                    "id": c.get("id"),
                    "startSeconds": c.get("startSeconds"),
                    "endSeconds": c.get("endSeconds"),
                    "score": c.get("score"),
                }
                for c in chosen[1:]
            ],
        }
        # The working clip is cut before the segment decision is recorded, so the decision names
        # the file it produced and a resume can pick it up.
        self.build_work_segment()
        self.log.decide(
            ident,
            step="segment",
            rule=(
                "measured rank 1 of the selector wins; --force-window with a reason is the only "
                "override, and a judge answer is an opinion recorded with applied=false"
            ),
            inputs={
                "top": SELECT_TOP_K,
                "truth": str(self.a.truth) if self.a.truth else None,
                "forceWindow": list(self.a.force_window) if self.a.force_window else None,
            },
            choice=self.segment,
            because=(
                f"{window.get('why')}"
                + (
                    f"; operator override: {self.segment['overrideReason']}"
                    if self.segment["forced"]
                    else ""
                )
            ),
            evidence=[selection_path],
        )
        return self.segment

    def segment_judge(self, sheets: Path, document):
        """One receipted OpenAI opinion on the top-5 windows, recorded and never applied."""
        ident = "segment-judge"
        request = Path(document.get("judgeRequest") or (sheets / "judge_request.json"))
        if not self.a.judge_segment:
            return {"asked": False, "why": "--judge-segment was not given"}
        if not os.environ.get("OPENAI_API_KEY"):
            self.log.decide(
                ident,
                step="segment",
                rule="ask the judge only with a key present; never guess an opinion",
                inputs={"request": str(request)},
                choice={"asked": False},
                because="OPENAI_API_KEY is not set, so no opinion was bought.",
            )
            return {"asked": False, "why": "OPENAI_API_KEY is not set"}
        out = self.ctx / "segment-judge.json"
        argv = [
            PY,
            "scripts/auto_clip.py",
            "judge-segment",
            "--request",
            str(request),
            "--out",
            str(out),
            "--model",
            self.a.model,
        ]
        result = self.execute(ident, argv, log=self.ctx / "segment-judge.log", paid=True)
        answer = read_json(out) or {}
        opinion = {
            "asked": True,
            "applied": False,
            "windowId": answer.get("windowId"),
            "reason": answer.get("reason"),
            "model": answer.get("model") or self.a.model,
            "exit": result.returncode,
        }
        self.log.decide(
            ident,
            step="segment",
            rule=(
                "docs/segment-selection.md: the ranking is measured; a judge answer is an opinion "
                "recorded with applied=false and changes nothing"
            ),
            inputs={"request": str(request), "model": self.a.model},
            choice=opinion,
            because=(
                f"the judge preferred {opinion['windowId']}: {opinion['reason']}"
                if opinion["windowId"]
                else "no usable opinion came back"
            ),
            evidence=[out],
        )
        return opinion

    def build_work_segment(self, *, start=None, end=None, suffix="seg", reason=""):
        """Trim (and de-letterbox) the chosen window into the working clip.

        Stream copy is used when, and only when, it is both possible and safe: no crop filter, no
        variable frame rate to normalise, and the window is the whole file, because a copy trim
        can only cut at a keyframe and the recorded source offsets must be exact.
        """
        ident = f"work-segment-{suffix}"
        window = self.segment
        start = window["startSeconds"] if start is None else start
        end = window["endSeconds"] if end is None else end
        crop = window.get("suggestedCrop") or {}
        admission = (self.log.latest("admission") or {}).get("choice", {})
        whole = (
            not crop.get("needed")
            and admission.get("variableFrameRate") is not True
            and start <= 0.0 + 1e-6
        )
        destination = self.work_dir / f"{self.name}-{suffix}.mp4"
        if whole:
            argv = [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(self.clip),
                "-c",
                "copy",
                str(destination),
            ]
            rule = "stream copy: no crop, constant frame rate, whole file"
        else:
            filters = []
            if crop.get("needed") and crop.get("ffmpeg"):
                filters.append(crop["ffmpeg"])
            argv = [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(self.clip),
                "-t",
                f"{max(end - start, 0.0):.3f}",
            ]
            if filters:
                argv += ["-vf", ",".join(filters)]
            argv += [
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(destination),
            ]
            rule = "re-encode: " + ", ".join(
                filter(
                    None,
                    [
                        "letterbox crop" if crop.get("needed") else "",
                        "variable frame rate" if admission.get("variableFrameRate") else "",
                        "exact source offsets" if start > 0 else "",
                    ],
                )
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.execute(ident, argv)
        window["workClip"] = str(destination)
        window["workStartSeconds"] = start
        window["workEndSeconds"] = end
        choice = {
            "status": "done",
            "workClip": str(destination),
            "streamCopy": whole,
            "sourceStartSeconds": start,
            "sourceEndSeconds": end,
            "crop": crop.get("ffmpeg"),
            "barFrameFraction": crop.get("barFrameFraction"),
        }
        self.log.decide(
            ident,
            step="admission",
            rule="never re-encode when a stream copy is possible AND safe",
            inputs={"start": start, "end": end, "crop": crop},
            choice=choice,
            because=rule + (f"; {reason}" if reason else ""),
            evidence=[destination],
        )
        return choice

    # -- 3. clean ------------------------------------------------------------------------------

    def mask_mode(self):
        """`--people-mask foreground` when the window is a crowd, by measurement."""
        fraction = (self.segment or {}).get("personPixelFractionMedian")
        crowd = isinstance(fraction, (int, float)) and fraction > CROWD_PERSON_FRACTION
        mode = "foreground" if crowd else "semantic"
        self.log.decide(
            "clean-mask-mode",
            step="clean",
            rule=(
                f"people.personPixelFractionMedian > CROWD_PERSON_FRACTION "
                f"({CROWD_PERSON_FRACTION}) -> --people-mask foreground. The threshold is the "
                "same idea and the same number worker/stages/dense_pi3x.py uses to switch its "
                "own anchor mask mode."
            ),
            inputs={"personPixelFractionMedian": fraction, "threshold": CROWD_PERSON_FRACTION},
            choice={"status": "done", "peopleMask": mode, "crowd": crowd},
            because=(
                f"{fraction} of the sampled frame is person pixels, "
                + ("above" if crowd else "at or below")
                + " the crowd threshold, so the background people "
                + ("stay" if crowd else "are removed with everyone else")
            ),
        )
        return mode

    def clean(self):
        ident = "clean"
        if self.step_done(ident):
            return self.reuse(ident, "clean", stage="clean")
        self.say("== 3 clean")
        mask = self.mask_mode()
        flags = ["--people-mask", mask]
        executions = 0
        applied_retry = False
        while True:
            executions += 1
            if executions > MAX_STAGE_EXECUTIONS:
                reason = (
                    f"the {MAX_STAGE_EXECUTIONS}-execution cap for this stage and window is "
                    "reached; the ledger would refuse another claim"
                )
                self.stage("clean", "failed", reason)
                self.stop_reason = reason
                break
            argv = self.run_clip_argv(["clean"], *flags)
            if executions > 1:
                argv += ["--force", "clean", "--stage-hypothesis", self.retry_hypothesis]
            result = self.execute(
                f"{ident}-exec{executions}", argv, log=self.ctx / "auto-clean.log", paid=True
            )
            refusal = ledger_refused(result.stdout)
            if refusal:
                self.stage("clean", "blocked", f"paid-stage ledger refused: {refusal}")
                self.stop_reason = f"paid-stage ledger refused: {refusal}"
                break
            verdict = self.judge_clean(executions)
            if verdict["verdict"] == "pass":
                self.stage("clean", "passed", verdict["because"], verdict["evidence"])
                break
            if verdict["verdict"] == "fail" or applied_retry or not verdict["flags"]:
                status = "failed"
                reason = verdict["because"]
                if applied_retry:
                    reason += (
                        f"; the one judge-driven retry allowed by the rubric "
                        f"(CLEAN_JUDGE_RETRIES={CLEAN_JUDGE_RETRIES}) is spent"
                    )
                self.stage("clean", status, reason, verdict["evidence"])
                self.stop_reason = reason
                break
            flags = merge_flags(flags, verdict["flags"])
            self.retry_hypothesis = verdict["hypothesis"]
            applied_retry = True
            if verdict.get("shorten"):
                self.shorten_window(verdict["shorten"])
        choice = {
            "status": "done" if self.stages[-1].status == "passed" else "stopped",
            "peopleMask": mask,
            "executions": executions,
            "finalFlags": flags,
        }
        self.log.decide(
            ident,
            step="clean",
            rule=(
                f"judge_clean exit 0 pass / {JUDGE_RETRY} retry / {JUDGE_FAIL} fail; a retry "
                f"applies its actionFlags once (CLEAN_JUDGE_RETRIES={CLEAN_JUDGE_RETRIES}) and "
                f"never exceeds MAX_STAGE_EXECUTIONS={MAX_STAGE_EXECUTIONS}"
            ),
            inputs={"peopleMask": mask},
            choice=choice,
            because=self.stages[-1].reason,
        )
        return choice

    def judge_clean(self, execution):
        """Ask judge_clean, and map its recommendation onto the flags a retry would change."""
        out = self.ctx / f"clean-judge-{execution}.json"
        argv = [
            PY,
            "scripts/judge_clean.py",
            "--run",
            str(self.ctx),
            "--out",
            str(out),
            "--model",
            self.a.model,
        ]
        result = self.execute(
            f"clean-judge-{execution}", argv, log=self.ctx / "auto-clean.log", paid=True
        )
        record = read_json(out) or {}
        code = result.returncode
        if code is None:  # dry run: plan the pass branch and say so
            verdict = "pass"
            because = "planned: --dry-run asked nothing, so the pass branch is what is printed"
        elif code == JUDGE_PASS:
            verdict = "pass"
            because = (
                f"the judge passed {record.get('passShare')} of "
                f"{len(record.get('framesJudged') or [])} frame pairs "
                f"(threshold {record.get('passThreshold')})"
            )
        elif code == JUDGE_RETRY:
            verdict = "retry"
            because = (
                f"the judge asked for {record.get('recommendedAction')}: "
                f"{record.get('actionReason')}"
            )
        elif code == JUDGE_FAIL:
            verdict = "fail"
            because = "the judge vetoed the plate: " + "; ".join(
                f"frame {item.get('index')} {item.get('kinds')}"
                for item in (record.get("criticalFailures") or [])
            )
        else:
            verdict = "fail"
            because = f"the judge could not answer (exit {code}); an unknown verdict is not a pass"
        flags = list(record.get("actionFlags") or [])
        action = record.get("recommendedAction")
        if action == NON_HUMAN_SUBJECT_ACTION and "--moved-mask" not in flags:
            flags.append("--moved-mask")
        shorten = self.shorten_action(record)
        decision = {
            "verdict": verdict,
            "because": because,
            "flags": flags,
            "shorten": shorten,
            "hypothesis": (
                f"judge_clean recommended {action}; retrying with {' '.join(flags)}"
                + (f" and a window shortened to its {shorten['keep']} half" if shorten else "")
            ),
            "evidence": [out],
        }
        self.log.decide(
            f"clean-judge-{execution}",
            step="clean",
            rule=(
                "map the judge's recommendedAction onto run_clip flags; non_human_subject_left "
                "means --moved-mask; a frame-filling subject confined to one end of the window "
                "also shortens the window"
            ),
            inputs={
                "exit": code,
                "recommendedAction": action,
                "actionFlags": record.get("actionFlags"),
            },
            choice=decision,
            because=because,
            evidence=[out],
        )
        return decision

    def shorten_action(self, record):
        """A frame-filling subject in only the late or early part of the window shortens it.

        The judge answers per frame, so "only late" and "only early" are measurable: compare the
        indices whose subject filled the frame against the window's own midpoint.
        """
        frames = record.get("frames") or []
        judged = [f.get("index") for f in frames if isinstance(f.get("index"), int)]
        if len(judged) < 2:
            return None
        flagged = []
        for frame in frames:
            answers = frame.get("answers") or {}
            fraction = answers.get("smearFraction")
            filling = (
                isinstance(fraction, (int, float))
                and not isinstance(fraction, bool)
                and fraction >= SUBJECT_FILL_FRACTION
            ) or answers.get("subjectsRemoved") in ("partial", "none")
            if filling and isinstance(frame.get("index"), int):
                flagged.append(frame["index"])
        if not flagged or len(flagged) == len(judged):
            return None
        midpoint = (min(judged) + max(judged)) / 2.0
        if all(index > midpoint for index in flagged):
            return {"keep": "early", "share": SUBJECT_SHORTEN_SHARE, "flaggedFrames": flagged}
        if all(index < midpoint for index in flagged):
            return {"keep": "late", "share": SUBJECT_SHORTEN_SHARE, "flaggedFrames": flagged}
        return None

    def shorten_window(self, shorten):
        start, end = self.segment["startSeconds"], self.segment["endSeconds"]
        span = (end - start) * shorten["share"]
        if shorten["keep"] == "early":
            new_start, new_end = start, start + span
        else:
            new_start, new_end = end - span, end
        if new_end - new_start < MIN_SPLIT_SECONDS:
            self.log.decide(
                "clean-shorten",
                step="clean",
                rule=f"a shortened window must still be >= MIN_SPLIT_SECONDS ({MIN_SPLIT_SECONDS})",
                inputs={"start": start, "end": end, "shorten": shorten},
                choice={"status": "refused"},
                because=(
                    f"the {shorten['keep']} half is only {new_end - new_start:.2f} s, below the "
                    "shortest window this pipeline reconstructs"
                ),
            )
            return None
        self.dropped.append(
            {
                "reason": "subject filled the frame in the other half of the window",
                "startSeconds": round(end - span if shorten["keep"] == "early" else start, 3),
                "endSeconds": round(end if shorten["keep"] == "early" else start + span, 3),
            }
        )
        self.segment["startSeconds"], self.segment["endSeconds"] = new_start, new_end
        return self.build_work_segment(
            start=new_start,
            end=new_end,
            suffix="seg-short",
            reason=f"the judge saw a frame-filling subject only in the {shorten['keep']} half",
        )

    # -- 4. camera solve + teleport guard ------------------------------------------------------

    def solve(self):
        ident = "solve"
        if self.step_done(ident):
            return self.reuse(ident, "solve", stage="solve")
        self.say("== 4 camera solve")
        retries = 0
        while True:
            argv = self.run_clip_argv(["pi3x", "frame_align"])
            if retries:
                argv += ["--force", "pi3x,frame_align", "--stage-hypothesis", self.retry_hypothesis]
            result = self.execute(
                f"{ident}-exec{retries + 1}", argv, log=self.ctx / "auto-solve.log", paid=True
            )
            refusal = ledger_refused(result.stdout)
            if refusal:
                self.stage("solve", "blocked", f"paid-stage ledger refused: {refusal}")
                self.stop_reason = f"paid-stage ledger refused: {refusal}"
                break
            if result.returncode in (0, None):
                self.stage(
                    "solve",
                    "passed",
                    "the pose guard found a continuous camera path" if result.ran else "planned",
                )
                break
            if retries >= SOLVE_RETRIES:
                reason = (
                    f"the solve failed again and the one retry this playbook allows "
                    f"(SOLVE_RETRIES={SOLVE_RETRIES}) is spent"
                )
                self.stage("solve", "failed", reason)
                self.stop_reason = reason
                break
            if self.solve_recovery(result) is None:
                break  # solve_recovery recorded the stage and the reason it could not continue
            retries += 1
        choice = {
            "status": "done" if self.stages[-1].status == "passed" else "stopped",
            "retries": retries,
            "droppedCoverage": self.dropped,
        }
        self.log.decide(
            ident,
            step="solve",
            rule=(
                f"on a teleport at t, retry ONCE on the longer side of the split if it is still "
                f">= MIN_SPLIT_SECONDS ({MIN_SPLIT_SECONDS}); on a non-overlapping-anchor refusal "
                "take the selector's next-best window instead"
            ),
            inputs={"retries": retries},
            choice=choice,
            because=self.stages[-1].reason,
        )
        return choice

    def solve_recovery(self, result):
        """Teleport -> split; anchors do not overlap -> next-best window. The caller counts them."""
        guard = read_json(self.ctx / "pose-guard.json") or {}
        text = result.stdout or ""
        if guard.get("ok") is False or TELEPORT_MARKER in text:
            moment = (guard.get("worst") or {}).get("timeSeconds")
            if moment is None:
                match = re.search(r"t=([0-9.]+)s", text)
                moment = float(match.group(1)) if match else None
            if moment is None:
                self.stage(
                    "solve",
                    "failed",
                    "the pose guard reported a teleport but named no time to split at",
                )
                return None
            start, end = self.segment["startSeconds"], self.segment["endSeconds"]
            absolute = start + moment
            head, tail = absolute - start, end - absolute
            keep = ("head", start, absolute) if head >= tail else ("tail", absolute, end)
            if keep[2] - keep[1] < MIN_SPLIT_SECONDS:
                reason = (
                    f"the longer side of the split at {absolute:.2f} s is "
                    f"{keep[2] - keep[1]:.2f} s, below MIN_SPLIT_SECONDS ({MIN_SPLIT_SECONDS})"
                )
                self.stage("solve", "failed", reason)
                self.stop_reason = reason
                return None
            dropped = (absolute, end) if keep[0] == "head" else (start, absolute)
            self.dropped.append(
                {
                    "reason": f"camera teleport at {absolute:.2f} s on the source clock",
                    "startSeconds": round(dropped[0], 3),
                    "endSeconds": round(dropped[1], 3),
                }
            )
            self.segment["startSeconds"], self.segment["endSeconds"] = keep[1], keep[2]
            because = (
                f"the pose guard found a camera teleport at {absolute:.2f} s; the {keep[0]} side "
                f"({keep[2] - keep[1]:.2f} s) is the longer one and is still usable"
            )
            self.log.decide(
                "solve-teleport",
                step="solve",
                rule="split at the teleport, keep the longer usable side, record the coverage lost",
                inputs={"teleportSeconds": moment, "guard": guard.get("worst")},
                choice={
                    "status": "retry",
                    "keep": keep[0],
                    "startSeconds": keep[1],
                    "endSeconds": keep[2],
                    "droppedSeconds": round(dropped[1] - dropped[0], 3),
                },
                because=because,
                evidence=[self.ctx / "pose-guard.json"],
            )
            self.retry_hypothesis = because
            self.build_work_segment(start=keep[1], end=keep[2], suffix="seg-split", reason=because)
            return {"because": because}
        if any(marker in text for marker in NO_OVERLAP_MARKERS):
            alternates = (self.segment or {}).get("alternates") or []
            if not alternates:
                reason = (
                    "the solve says the anchors do not overlap and the selector offered no "
                    "next-best window"
                )
                self.stage("solve", "failed", reason)
                self.stop_reason = reason
                return None
            nxt = alternates.pop(0)
            because = (
                "the solve refused: its anchors do not agree on one scene, so the selector's "
                f"next-best window {nxt['id']} "
                f"({nxt['startSeconds']:.2f}-{nxt['endSeconds']:.2f} s) is tried once"
            )
            self.dropped.append(
                {
                    "reason": "anchors did not overlap over this window",
                    "startSeconds": self.segment["startSeconds"],
                    "endSeconds": self.segment["endSeconds"],
                }
            )
            self.segment["startSeconds"] = nxt["startSeconds"]
            self.segment["endSeconds"] = nxt["endSeconds"]
            self.segment["windowId"] = nxt["id"]
            self.log.decide(
                "solve-next-window",
                step="solve",
                rule="a non-overlapping-anchor refusal takes the selector's next-best window once",
                inputs={"window": nxt},
                choice={"status": "retry", **nxt},
                because=because,
            )
            self.retry_hypothesis = because
            self.build_work_segment(
                start=nxt["startSeconds"], end=nxt["endSeconds"], suffix="seg-alt", reason=because
            )
            return {"because": because}
        reason = (
            "the solve failed for a reason this playbook has no rule for "
            f"(exit {result.returncode}); diagnose it before paying for another attempt"
        )
        self.stage("solve", "failed", reason)
        self.stop_reason = reason
        return None

    # -- 5. world ------------------------------------------------------------------------------

    def world(self):
        ident = "world"
        if self.step_done(ident):
            return self.reuse(ident, "world", stage="world")
        if self.a.marble == "none":
            self.stage("world", "skipped", "--marble none: no world was generated")
            return {"status": "done", "generated": False}
        self.say("== 5 world")
        self.execute(
            f"{ident}-prompt", self.run_clip_argv(["world_prompt"]), log=self.ctx / "auto-world.log"
        )
        gate = self.world_gate()
        if not gate["pass"]:
            self.stage("world", "failed", gate["because"])
            self.stop_reason = gate["because"]
            return {"status": "stopped", **gate}
        stage_name = {
            "video": "marble_video",
            "image": "marble_image",
            "multi": "marble_multi",
            "both": "marble_video",
        }[self.a.marble]
        result = self.execute(
            f"{ident}-marble",
            self.run_clip_argv(["review", stage_name], "--gate-pass"),
            log=self.ctx / "auto-world.log",
            paid=True,
        )
        refusal = ledger_refused(result.stdout)
        if refusal:
            self.stage("world", "blocked", f"paid-stage ledger refused: {refusal}")
            return {"status": "stopped"}
        ok = result.returncode in (0, None)
        self.stage(
            "world",
            "passed" if ok else "failed",
            "one generation for this segment" if ok else f"marble stage exited {result.returncode}",
        )
        choice = {
            "status": "done" if ok else "stopped",
            "generated": True,
            "regenerated": False,
            "gate": gate,
        }
        self.log.decide(
            ident,
            step="world",
            rule=(
                "world_prompt -> gate -> ONE marble generation per segment. A world is never "
                "regenerated: docs/quality-rubric.md allows one generation per new source clip "
                "and zero replacements."
            ),
            inputs={"marble": self.a.marble},
            choice=choice,
            because=self.stages[-1].reason,
        )
        return choice

    def world_gate(self):
        """--gate-pass only when the judge passed, or an explicit operator override is logged."""
        passed = any(s.stage == "clean" and s.status == "passed" for s in self.stages)
        override = self.a.operator_override
        if passed:
            because = "the clean judge passed the plate, so the credit gate opens on evidence"
            decision = {"pass": True, "override": False, "because": because}
        elif override:
            because = (
                f"OPERATOR OVERRIDE, not a pass: the clean judge did not pass and the operator "
                f"asked for the gate to open anyway -- {override}"
            )
            decision = {"pass": True, "override": True, "reason": override, "because": because}
        else:
            because = (
                "the clean judge did not pass and no --operator-override was given, so no credits "
                "are spent"
            )
            decision = {"pass": False, "override": False, "because": because}
        self.log.decide(
            "world-gate",
            step="world",
            rule=(
                "--gate-pass is given only when the judge PASSES or --operator-override REASON is "
                "supplied; an override is logged as an override, never as a pass"
            ),
            inputs={"cleanJudgePassed": passed, "operatorOverride": override},
            choice=decision,
            because=because,
        )
        return decision

    # -- 6. people -----------------------------------------------------------------------------

    def cast(self):
        ident = "people"
        if self.step_done(ident):
            choice = self.reuse(ident, "people", stage="people")
            self.identity_verified = choice.get("identityVerified")
            self.identity_reason = "recorded by the earlier run this resume reuses"
            reconstructed = choice.get("reconstructed") or []
            for track in choice.get("selected") or []:
                self.people.append(
                    Person(
                        track=track,
                        selected=True,
                        reconstructed=track in reconstructed,
                        unresolvedReason=next(
                            (
                                item.get("reason")
                                for item in choice.get("unresolved") or []
                                if item.get("track") == track
                            ),
                            None,
                        ),
                    )
                )
            return choice
        self.say("== 6 people")
        result = self.execute(
            f"{ident}-tracks",
            self.run_clip_argv(["tracks"], "--people", str(self.a.people_cap)),
            log=self.ctx / "auto-people.log",
            paid=True,
        )
        refusal = ledger_refused(result.stdout)
        if refusal:
            self.stage("tracks", "blocked", f"paid-stage ledger refused: {refusal}")
            self.stop_reason = f"paid-stage ledger refused: {refusal}"
            return {"status": "stopped"}
        tracks = read_json(self.ctx / "tracks" / "tracks.json") or {}
        selected = self.judge_people(tracks)
        identity = self.identity(selected)
        selected = identity["selected"]
        for person in self.people:
            person.selected = person.track in selected
        if not selected:
            self.stage(
                "people",
                "skipped",
                "no track was selected for reconstruction; the run is world-only",
            )
            return {"status": "done", "selected": [], "worldOnly": True}
        stages = []
        for index in range(len(selected)):
            stages += [
                f"person_prep_{index:02d}",
                f"lhm_frozen_{index:02d}",
                f"lhm_motion_{index:02d}",
            ]
        stages.append("package_people")
        result = self.execute(
            f"{ident}-reconstruct",
            self.run_clip_argv(stages, "--people", str(len(selected))),
            log=self.ctx / "auto-people.log",
            paid=True,
        )
        self.resolve_people(result, selected)
        reconstructed = [p.track for p in self.people if p.reconstructed]
        unresolved = [p for p in self.people if p.selected and not p.reconstructed]
        if reconstructed:
            self.stage(
                "people",
                "passed",
                f"reconstructed track(s) {reconstructed}"
                + (f"; {len(unresolved)} selected subject(s) unresolved" if unresolved else ""),
            )
        else:
            self.stage(
                "people",
                "failed",
                "every selected subject was refused: "
                + "; ".join(f"track {p.track}: {p.unresolvedReason}" for p in unresolved)
                + ". The clip is world-only, which is a valid outcome.",
            )
        choice = {
            "status": "done",
            "selected": selected,
            "reconstructed": reconstructed,
            "unresolved": [{"track": p.track, "reason": p.unresolvedReason} for p in unresolved],
            "identityVerified": self.identity_verified,
        }
        self.log.decide(
            ident,
            step="people",
            rule=(
                f"reconstruct judge_people's `selected` main/secondary up to --people-cap "
                f"({self.a.people_cap}); the same subset goes to the identity audit via "
                "run_clip's --people N (which passes --only-tracks 0..N-1)"
            ),
            inputs={"trackCount": tracks.get("trackCount"), "cap": self.a.people_cap},
            choice=choice,
            because=self.stages[-1].reason,
        )
        return choice

    def judge_people(self, tracks):
        out = self.ctx / "people-judge.json"
        argv = [
            PY,
            "scripts/judge_people.py",
            "--tracks",
            str(self.ctx / "tracks"),
            "--clip",
            str(self.segment["workClip"]),
            "--out",
            str(out),
            "--model",
            self.a.model,
        ]
        self.execute("people-judge", argv, log=self.ctx / "auto-people.log", paid=True)
        record = read_json(out) or {}
        labels = {}
        for entry in record.get("labels") or []:
            if isinstance(entry, dict) and isinstance(entry.get("id"), int):
                labels[entry["id"]] = entry
        selected = [t for t in (record.get("selected") or []) if isinstance(t, int)]
        count = tracks.get("trackCount") or record.get("trackCount") or len(labels)
        self.people = [
            Person(
                track=index,
                label=(labels.get(index) or {}).get("label", "unknown"),
                seen=True,
                why=(labels.get(index) or {}).get("reason", ""),
            )
            for index in range(count)
        ]
        capped = selected[: self.a.people_cap]
        self.log.decide(
            "people-judge",
            step="people",
            rule=(
                "judge_people labels main/secondary/background/crowd/not_a_person and orders the "
                "main/secondary tracks by measured on-screen prominence; only main/secondary are "
                "reconstructed, capped at --people-cap"
            ),
            inputs={"trackCount": count, "selected": selected, "cap": self.a.people_cap},
            choice={
                "status": "done",
                "selected": capped,
                "labels": {str(k): v.get("label") for k, v in labels.items()},
            },
            because=(
                f"{len(selected)} track(s) were labelled main or secondary; "
                f"{len(capped)} fit under the cap"
            ),
            evidence=[out],
        )
        if capped and capped != list(range(len(capped))):
            self.log.decide(
                "people-cap-order",
                step="people",
                rule="run_clip's --people N takes track RANKS 0..N-1, not an arbitrary track list",
                inputs={"selected": capped},
                choice={"status": "note", "passedAs": f"--people {len(capped)}"},
                because=(
                    f"the judge's prominence order {capped} is not the tracker's rank order, and "
                    "run_clip has no flag for an explicit track list; the run reconstructs ranks "
                    f"0..{len(capped) - 1} and this difference is recorded, not hidden"
                ),
            )
        return capped

    def identity(self, selected):
        """Look-alike subjects: one subject, marked unverified, rather than a false 'ok'."""
        document = read_json(self.ctx / "identity.json") or {}
        reason = document.get("reason") or ""
        no_power = document.get("ok") is False and IDENTITY_NO_POWER_MARKER in reason
        if no_power and len(selected) > 1:
            kept = selected[:1]
            self.identity_verified = False
            self.identity_reason = reason
            because = (
                "the identity audit's swap control was missed, so it cannot tell these subjects "
                f"apart (same kit): reconstructing ONE of {selected} and marking "
                "identityVerified false rather than shipping a possible identity swap"
            )
        elif document.get("ok") is True:
            kept, self.identity_verified = selected, True
            self.identity_reason = "every track matched its own forward and backward reference"
            because = self.identity_reason
        elif not document:
            kept, self.identity_verified = selected, None
            self.identity_reason = "no identity.json was written"
            because = "the audit left no report, so identity is unverified, not verified"
        else:
            kept, self.identity_verified = selected, False
            self.identity_reason = reason or "the audit did not pass"
            because = self.identity_reason
        self.log.decide(
            "identity",
            step="people",
            rule=(
                "an audit with no power between look-alike subjects reconstructs ONE subject and "
                "records identityVerified: false; it never reports a pass"
            ),
            inputs={"selected": selected, "auditOk": document.get("ok"), "auditReason": reason},
            choice={"status": "done", "selected": kept, "identityVerified": self.identity_verified},
            because=because,
            evidence=[self.ctx / "identity.json"],
        )
        return {"selected": kept}

    def resolve_people(self, result, selected):
        """Mark each selected subject reconstructed, or unresolved with the refusal that says why."""
        text = result.stdout or ""
        log_text = ""
        log_path = self.ctx / "auto-people.log"
        if log_path.is_file():
            log_text = log_path.read_text()
        haystack = text + "\n" + log_text
        for index, track in enumerate(selected):
            person = next((p for p in self.people if p.track == track), None)
            if person is None:
                person = Person(track=track)
                self.people.append(person)
            # Attribute a refusal to the slot whose line it is on: run_clip.py reports a stage
            # failure as `!! <stage> FAILED (...): <error>` on one line.
            slot = re.compile(rf"(?:person_prep|lhm_frozen|lhm_motion)_{index:02d}\b[^\n]*")
            lines = slot.findall(haystack)
            refused = None
            if any(PREP_REFUSAL_MARKER in line for line in lines):
                refused = "person prep refused: the subject is never fully visible inside the frame"
            elif any(POSE_REFUSAL_MARKER in line for line in lines):
                refused = (
                    "the pose model refused: no source person pose was detected, and a canonical "
                    "pose substitute is not a reconstruction"
                )
            if refused and result.ran:
                person.unresolvedReason = refused
                person.reconstructed = False
            else:
                person.reconstructed = result.returncode in (0, None)
                if not person.reconstructed:
                    person.unresolvedReason = (
                        f"the people stages exited {result.returncode}; see {log_path}"
                    )

    # -- 7. reconcile --------------------------------------------------------------------------

    def reconcile(self, stage, *, attempt_id=None):
        """Archive a provably failed attempt's leftovers, reconcile its claim, unblock the retry."""
        self.say(f"== 7 reconcile {stage}")
        attempts = pending_attempts(self.ledger, run_name=self.name, stage=stage)
        if attempt_id:
            attempts = [a for a in attempts if a["id"] == attempt_id]
        if not attempts:
            self.log.decide(
                f"reconcile-{stage}",
                step="reconcile",
                rule="only a pending or unknown claim can be reconciled",
                inputs={"stage": stage, "attempt": attempt_id},
                choice={"status": "none"},
                because="no pending or unknown claim for this candidate and stage",
            )
            self.say("   nothing pending or unknown for this candidate and stage")
            return {"status": "none"}
        attempt = attempts[0]
        claim = attempt["claim"]
        log_path = Path(claim.get("log") or "")
        log_text = log_path.read_text() if log_path.is_file() else ""
        manifests = []
        for output in claim.get("results") or []:
            directory = Path(output)
            for candidate in (directory / "recovery-manifest.json", directory / "modal-run.json"):
                document = read_json(candidate)
                if isinstance(document, dict):
                    manifests.append((str(candidate), document))
        verdict, kind, reason = classify_failure(log_text, manifests)
        evidence = str(manifests[0][0]) if manifests else str(log_path)
        if verdict != "failed":
            self.log.decide(
                f"reconcile-{stage}",
                step="reconcile",
                rule=(
                    "never reconcile an unknown claim without evidence: ambiguous evidence stops "
                    "and reports"
                ),
                inputs={"attempt": attempt["id"], "kind": kind, "log": str(log_path)},
                choice={"status": "stopped", "kind": kind},
                because=reason,
                evidence=[log_path],
            )
            self.stop_reason = f"reconcile {stage}: {reason}"
            self.say(f"   STOP: {reason}")
            return {"status": "stopped", "kind": kind, "because": reason}
        archived = []
        for output in claim.get("results") or []:
            directory = Path(output)
            if directory.is_dir():
                destination = next_archive(directory)
                if not self.dry_run:
                    directory.rename(destination)
                archived.append(str(destination))
                self.say(f"   archived {directory} -> {destination}")
        argv = [
            PY,
            "scripts/stage_attempts.py",
            "--ledger",
            str(self.ledger),
            "--reconcile",
            attempt["id"],
            "--status",
            "failed",
            "--evidence",
            evidence,
            "--reason",
            reason,
        ]
        self.execute(f"reconcile-{stage}", argv, log=self.ctx / "auto-reconcile.log")
        refreshed = self.refresh_derived(stage, reason)
        self.log.decide(
            f"reconcile-{stage}",
            step="reconcile",
            rule=(
                "a provably failed attempt is archived by RENAMING its output directory to "
                "<dir>.failed-attemptN (never deleted), its ledger claim is reconciled with the "
                "saved log or manifest as evidence, and the derived files that would block the "
                "retry are marked stale"
            ),
            inputs={"attempt": attempt["id"], "kind": kind, "evidence": evidence},
            choice={"status": "done", "archived": archived, "refreshed": refreshed},
            because=reason,
            evidence=[evidence],
        )
        return {"status": "done", "kind": kind, "archived": archived, "refreshed": refreshed}

    def refresh_derived(self, stage, reason):
        """Stale identity.json aside; the stage's status in state.json marked stale with a reason."""
        refreshed = []
        identity = self.ctx / "identity.json"
        if identity.is_file():
            destination = identity.with_name(f"identity.json.stale-{int(time.time())}")
            if not self.dry_run:
                identity.rename(destination)
            refreshed.append(str(destination))
        state_path = self.ctx / "state.json"
        state = read_json(state_path)
        if isinstance(state, dict) and isinstance(state.get("stages"), dict):
            entry = state["stages"].get(stage)
            if isinstance(entry, dict):
                entry["status"] = "stale"
                entry["staleReason"] = reason
                if not self.dry_run:
                    atomic_json(state_path, state)
                refreshed.append(f"{state_path}#stages.{stage}=stale")
        return refreshed

    # -- 9. package and report -----------------------------------------------------------------

    def package(self):
        ident = "package"
        if self.step_done(ident):
            return self.reuse(ident, "package", stage="package")
        self.say("== 9 package")
        result = self.execute(
            ident,
            self.run_clip_argv(["scale_fit", "place_fit", "anchors", "verify"]),
            log=self.ctx / "auto-package.log",
            paid=False,
        )
        state = read_json(self.ctx / "state.json") or {}
        placement = (state.get("stages") or {}).get("_placement") or {}
        fitted = placement.get("floor") is not None and placement.get("fittedScale0") is not None
        gate_ok = placement.get("scaleGateOk")
        if fitted and gate_ok is not False:
            note = "fitted placement: the scale fit passed its gate"
        elif fitted and gate_ok is False:
            note = (
                "NOT fitted placement: the scale gate failed, so the run is viewable on the "
                "geometric-mean fallback scale and its metres are not established"
            )
        else:
            note = (
                "NOT fitted placement: no scale fit is recorded, so the viewer runs on Marble's "
                "own metric scale, which has been wrong by 2.9x"
            )
        self.stage("package", "passed" if result.returncode in (0, None) else "failed", note)
        choice = {"status": "done", "fitted": bool(fitted and gate_ok is not False), "note": note}
        self.log.decide(
            ident,
            step="package",
            rule=(
                "a failed scale_fit gate keeps the run VIEWABLE on the fallback scale and the "
                "report says so; it never claims fitted placement"
            ),
            inputs={"placement": placement},
            choice=choice,
            because=note,
        )
        if self.a.separate_audio:
            self.execute(
                "separate-audio",
                [
                    PY,
                    "scripts/separate_audio.py",
                    "--video",
                    str(self.segment["workClip"]),
                    "--out",
                    str(self.ctx / "audio"),
                ],
                log=self.ctx / "auto-package.log",
            )
        return choice

    def sequence(self, shots):
        """`--mode sequence`: merge the per-shot candidates back onto the original clock."""
        listing = self.ctx / "shot-sequence.json"
        entries = [
            {
                "candidateDir": f"public/worlds/{self.name}-shot{s['index']:02d}-4d",
                "world": f"/marble-{self.name}-shot{s['index']:02d}-clean.spz",
                "sourceStart": s["start"],
                "sourceEnd": s["end"],
                "stateJson": str(
                    self.root / ".context/run" / f"{self.name}-shot{s['index']:02d}" / "state.json"
                ),
            }
            for s in shots
        ]
        if not self.dry_run:
            atomic_json(listing, {"shots": entries})
        argv = [
            PY,
            "scripts/package_shot_sequence.py",
            "--source",
            str(self.clip),
            "--shots",
            str(listing),
            "--out",
            str(self.root / "public" / "worlds" / f"{self.name}-sequence-4d"),
        ]
        self.execute("sequence", argv, log=self.ctx / "auto-package.log")
        self.log.decide(
            "sequence",
            step="package",
            rule=(
                f"sequence mode plans the K longest usable shots (cap --sequence-cap, default "
                f"{SEQUENCE_SHOT_CAP}) and merges them with package_shot_sequence"
            ),
            inputs={"shots": entries},
            choice={"status": "done", "count": len(entries)},
            because=f"{len(entries)} shot(s) were planned, longest usable first",
            evidence=[listing],
        )
        return entries

    def plan_shots(self):
        """Cut report -> the K longest usable shots, longest first."""
        cuts_path = self.ctx / "cuts.json"
        argv = [
            PY,
            "scripts/shot_cuts.py",
            "--video",
            str(self.clip),
            "--json",
            str(cuts_path),
            "--no-score",
        ]
        self.execute("cuts", argv, log=self.ctx / "auto-cuts.log")
        document = read_json(cuts_path) or {}
        shots = [s for s in (document.get("shots") or []) if not s.get("tooShort")]
        shots.sort(key=lambda s: s.get("seconds", 0.0), reverse=True)
        chosen = shots[: self.a.sequence_cap]
        chosen.sort(key=lambda s: s.get("start", 0.0))
        self.log.decide(
            "shots",
            step="segment",
            rule=f"longest usable shots first, capped at --sequence-cap ({self.a.sequence_cap})",
            inputs={"cutCount": document.get("cutCount"), "shotCount": document.get("shotCount")},
            choice={"status": "done", "shots": [s.get("index") for s in chosen]},
            because=(
                f"{len(shots)} usable shot(s) of {document.get('shotCount')}; the "
                f"{len(chosen)} longest are planned"
            ),
            evidence=[cuts_path],
        )
        return chosen

    def report(self):
        """The per-clip report: what was chosen, who came back, what is invented, what is not."""
        finished = ("passed", "reused")
        ran = {s.stage for s in self.stages if s.status in finished}
        observed = [text for stage, text in OBSERVED_BY_STAGE.items() if stage in ran]
        invented = [text for stage, text in INVENTED_BY_STAGE.items() if stage in ran]
        packaged = any(s.stage == "package" and s.status in finished for s in self.stages)
        reconstructed = [p for p in self.people if p.reconstructed]
        if self.dry_run:
            outcome = "planned"
        elif packaged and reconstructed:
            outcome = "packaged"
        elif packaged:
            outcome = "world-only"
        else:
            outcome = "stopped"
        placement = (self.log.latest("package") or {}).get("choice", {})
        document = {
            "name": self.name,
            "clip": str(self.clip),
            "sourceSha256": self.log.doc.get("sourceSha256"),
            "mode": self.a.mode,
            "outcome": outcome,
            "stopReason": self.stop_reason,
            "segment": {
                "windowId": (self.segment or {}).get("windowId"),
                "startSeconds": (self.segment or {}).get("startSeconds"),
                "endSeconds": (self.segment or {}).get("endSeconds"),
                "why": (self.segment or {}).get("why"),
                "forced": (self.segment or {}).get("forced"),
                "judgeOpinion": (self.segment or {}).get("judgeOpinion"),
                "coverageNotSelected": self.coverage,
                "droppedCoverage": self.dropped,
            },
            "people": [
                {
                    "track": p.track,
                    "label": p.label,
                    "seen": p.seen,
                    "selected": p.selected,
                    "reconstructed": p.reconstructed,
                    "unresolvedReason": p.unresolvedReason,
                    "why": p.why,
                }
                for p in self.people
            ],
            "identityVerified": self.identity_verified,
            "identityReason": self.identity_reason,
            "stages": [
                {"stage": s.stage, "status": s.status, "reason": s.reason, "evidence": s.evidence}
                for s in self.stages
            ],
            "placement": {"fitted": bool(placement.get("fitted")), "note": placement.get("note")},
            "observed": observed,
            "invented": invented,
            "note": (
                "Judge verdicts recorded here are triage opinions (docs/quality-rubric.md), not "
                "proof of visual quality. Nothing in this report is a headset or viewer check."
            ),
        }
        self.log.doc["report"] = document
        self.log.doc["costs"] = ledger_costs(
            self.ledger,
            source_sha256=self.log.doc.get("sourceSha256"),
            candidates=[self.name],
        )
        self.log.doc["finishedAt"] = now()
        self.log.save()
        return document

    # -- the playbook --------------------------------------------------------------------------

    def go(self):
        shots = []
        self.admit()
        if self.a.mode == "sequence":
            shots = self.plan_shots()
        if not self.choose_segment():
            self.report()
            return 1
        self.clean()
        if self.stop_reason:
            self.report()
            return 1
        self.solve()
        if self.stop_reason:
            self.report()
            return 1
        self.world()
        if self.stop_reason:
            self.report()
            return 1
        self.cast()
        if self.stop_reason:
            self.report()
            return 1
        self.package()
        if self.a.mode == "sequence":
            self.sequence(shots)
        document = self.report()
        print_report(document)
        return 0 if document["outcome"] in ("packaged", "world-only", "planned") else 1


def print_report(document):
    line = "=" * 78
    print(f"\n{line}\n{document['name']}: {document['outcome']}\n{line}")
    segment = document["segment"]
    print(
        f"segment {segment['windowId']} {segment['startSeconds']}-{segment['endSeconds']} s"
        f"  {segment['why']}"
    )
    coverage = segment.get("coverageNotSelected") or {}
    if coverage:
        print(f"  not selected: {coverage.get('seconds')} s ({coverage.get('fraction')})")
    for dropped in segment.get("droppedCoverage") or []:
        print(f"  dropped {dropped['startSeconds']}-{dropped['endSeconds']} s: {dropped['reason']}")
    for person in document["people"]:
        state = (
            "reconstructed"
            if person["reconstructed"]
            else ("unresolved: " + (person["unresolvedReason"] or "not selected"))
        )
        print(
            f"  track {person['track']} ({person['label']}) selected="
            f"{person['selected']} -> {state}"
        )
    print(f"  identity verified: {document['identityVerified']} -- {document['identityReason']}")
    for stage in document["stages"]:
        print(f"  {stage['stage']:<10} {stage['status']:<8} {stage['reason']}")
    print(f"  placement: {document['placement']['note']}")
    for text in document["observed"]:
        print(f"  observed: {text}")
    for text in document["invented"]:
        print(f"  invented: {text}")
    print(line)


# ---- the segment judge ------------------------------------------------------------------------


JUDGE_BRIEF = """Each image is a contact sheet of ONE candidate window of the same video, in time
order, labelled with its window id.

A reconstruction pipeline will rebuild exactly one of these windows as a walkable 3D scene. The
windows were already ranked by measurement (camera travel, people, continuity, exposure). Your job
is the one thing measurement cannot do: say which window a viewer would most want to be inside --
the play, the action, the moment the clip is about.

{question}

Windows:
{windows}

Answer with JSON and nothing else:
{{"windowId": "<one of {ids}>", "reason": "<one clause>"}}
"""


def judge_segment(request_path, out, *, model, urlopen=None, report=None):
    """One receipted opinion about the selector's top-K windows. Imported lazily; costs one call."""
    # Imported here, not at module scope: --dry-run must cost nothing but a process start.
    import vlm_once

    request = json.loads(Path(request_path).read_text())
    windows = request.get("windows") or []
    if not windows:
        raise ValueError(f"{request_path} lists no windows to judge")
    images = [w["contactSheet"] for w in windows if w.get("contactSheet")]
    if not images:
        raise ValueError(f"{request_path} names no contact sheets")
    ids = [w["id"] for w in windows]
    brief = JUDGE_BRIEF.format(
        question=request.get("question", ""),
        windows="\n".join(
            f"  {w['id']} rank {w['rank']} {w['startSeconds']}-{w['endSeconds']} s: {w['why']}"
            for w in windows
        ),
        ids=", ".join(ids),
    )

    def validate(record):
        window_id = record.get("windowId")
        if window_id not in ids:
            raise ValueError(f"windowId must be one of {ids}, got {window_id!r}")
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be one non-empty clause")
        return {
            "schema": "wander.segment-judge/1",
            "windowId": window_id,
            "reason": reason.strip(),
            "model": model,
            "request": str(request_path),
            "videoSha256": request.get("videoSha256"),
            "applied": False,
            "note": (
                "An opinion about the contact sheets only. docs/segment-selection.md keeps the "
                "measured ranking; this changes nothing."
            ),
        }

    return vlm_once.request_once(
        Path(out),
        brief=brief,
        images=images,
        model=model,
        validate=validate,
        identity_extra={"windows": ids, "videoSha256": request.get("videoSha256")},
        urlopen=urlopen,
        label="segment judgement",
        report=report,
    )


# ---- cli --------------------------------------------------------------------------------------


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command")

    run = sub.add_parser("run", help="drive one clip through the playbook")
    run.add_argument("--clip", required=True, help="the ORIGINAL source video")
    run.add_argument("--name", required=True, help="the run name; .context/run/<name>")
    run.add_argument("--mode", default="best-shot", choices=["best-shot", "sequence"])
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print every command and decision; run nothing, spend nothing",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="reuse the decisions an earlier invocation already recorded",
    )
    run.add_argument("--truth", help="hand-labelled cut boundaries for the selector")
    run.add_argument("--force-window", nargs=2, type=float, metavar=("START", "END"))
    run.add_argument("--reason", help="why the override; required with --force-window")
    run.add_argument(
        "--judge-segment",
        action="store_true",
        help="buy ONE receipted opinion on the top-5 windows; recorded, never applied",
    )
    run.add_argument(
        "--operator-override",
        metavar="REASON",
        help="open the credit gate without a judge pass; logged as an override",
    )
    run.add_argument("--people-cap", type=int, default=PEOPLE_CAP)
    run.add_argument("--sequence-cap", type=int, default=SEQUENCE_SHOT_CAP)
    run.add_argument(
        "--marble", default="video", choices=["video", "image", "multi", "both", "none"]
    )
    run.add_argument("--fps", type=float, default=12)
    run.add_argument("--model", default="gpt-6-astra", help="the judge model")
    run.add_argument("--stage-ledger", default=str(DEFAULT_LEDGER))
    run.add_argument("--source-sha256", help="canonical source hash shared by reviewed aliases")
    run.add_argument(
        "--objects",
        action="store_true",
        help="look for thrown objects too (off by default: run_clip gets --no-objects)",
    )
    run.add_argument(
        "--finetune",
        action="store_true",
        help="run the gsplat fine-tune (off by default: run_clip gets --skip-finetune)",
    )
    run.add_argument(
        "--separate-audio",
        action="store_true",
        help="follow-up: run scripts/separate_audio.py on the working segment",
    )

    rec = sub.add_parser("reconcile", help="reconcile one provably failed paid claim")
    rec.add_argument("--run", required=True, help=".context/run/<name>")
    rec.add_argument("--stage", required=True, help="the logical stage, e.g. pi3x, clean, tracks")
    rec.add_argument("--attempt", help="one attempt id, when the stage has several")
    rec.add_argument("--stage-ledger", default=str(DEFAULT_LEDGER))
    rec.add_argument("--dry-run", action="store_true")

    pre = sub.add_parser("preflight", help="what the GPU stages will demand of this workspace")
    pre.add_argument("--profile", default=os.environ.get("MODAL_PROFILE", "dtpu"))
    pre.add_argument(
        "--check", action="store_true", help="also issue the read-only `modal volume ls` listings"
    )
    pre.add_argument("--dry-run", action="store_true")
    pre.add_argument("--json", help="write the checklist here")

    judge = sub.add_parser("judge-segment", help="one receipted opinion on the top-K windows")
    judge.add_argument("--request", required=True, help="the selector's judge_request.json")
    judge.add_argument("--out", required=True)
    judge.add_argument("--model", default="gpt-6-astra")
    return ap


def command_run(args, runner=shell_runner):
    return AutoClip(args, runner=runner).go()


def command_reconcile(args, runner=shell_runner):
    run_dir = Path(args.run)
    namespace = argparse.Namespace(
        clip=str(run_dir / "source.mp4"),
        name=run_dir.name,
        mode="best-shot",
        dry_run=bool(args.dry_run),
        resume=True,
        stage_ledger=args.stage_ledger,
        truth=None,
        force_window=None,
        reason=None,
        judge_segment=False,
        operator_override=None,
        people_cap=PEOPLE_CAP,
        sequence_cap=SEQUENCE_SHOT_CAP,
        marble="none",
        fps=12,
        model="gpt-6-astra",
        source_sha256=None,
        objects=False,
        finetune=False,
        separate_audio=False,
    )
    auto = AutoClip(namespace, runner=runner)
    result = auto.reconcile(args.stage, attempt_id=args.attempt)
    auto.log.save()
    return 0 if result["status"] in ("done", "none") else 1


def command_preflight(args, runner=shell_runner):
    requirements = worker_requirements()
    rows, commands = preflight_checklist(args.profile, requirements)
    print(f"workspace preflight for MODAL_PROFILE={args.profile}")
    print("what the GPU stages will demand (read from the worker sources):")
    for row in rows:
        print(f"  volume {row['volume']} at {row['mount']} -- {', '.join(row['demandedBy'])}")
        print(f"    declared in: {', '.join(row['declaredIn'])}")
        if row["stagingPhases"]:
            print(f"    staged by phases: {', '.join(row['stagingPhases'])}")
    print("  model paths the workers check for before inference:")
    for weight in requirements["weights"]:
        print(f"    [{weight['kind']:<9}] {weight['path']}  ({weight['worker']})")
    print("  staging commands for anything missing:")
    for argv in commands:
        print(f"    {shlex.join(str(p) for p in argv)}")
    if args.json:
        atomic_json(
            Path(args.json),
            {
                "schema": "wander.auto-clip-preflight/1",
                "profile": args.profile,
                "requirements": requirements,
                "rows": rows,
                "commands": [[str(p) for p in c] for c in commands],
            },
        )
    if args.check and not args.dry_run:
        for argv in commands:
            if argv[0] == "uv":
                runner([str(p) for p in argv], cwd=ROOT)
    elif args.check:
        print("  --dry-run: the checklist above is all this prints; nothing was listed remotely")
    return 0


def main(argv=None, runner=shell_runner):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command == "run":
        return command_run(args, runner=runner)
    if args.command == "reconcile":
        return command_reconcile(args, runner=runner)
    if args.command == "preflight":
        return command_preflight(args, runner=runner)
    if args.command == "judge-segment":
        record = judge_segment(args.request, args.out, model=args.model)
        print(json.dumps(record, indent=1))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
