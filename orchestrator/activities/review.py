"""The reviewing agent: judges a finished attempt and may revise its outputs.

After a rubric'd stage passes its automatic validators, the workflow hands the attempt here. The
attempt's outputs are copied into a scratch workspace, the task is written down for the harness,
and the harness runs with file tools of its own. Its ``decision.json`` decides whether the run
moves on, the stage is retried with a stated hypothesis, or a human is asked. If the agent edited
anything under ``outputs/``, the edited tree is frozen as a new attempt and that is what a pass
selects, so an agent fix is as immutable and auditable as a stage's own output.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from temporalio import activity

from orchestrator.agent.contracts import AgentArtifact, AgentAttemptSummary, AgentTaskPacket
from orchestrator.agent.harness import DECISION_FILE, HarnessAgent
from orchestrator.agent.tools import AskHuman, RequestRetry, SubmitVerdict
from orchestrator.activities.stage import attempt_identifier
from orchestrator.artifacts import Archive, LocalCAS, UploadOutbox, freeze_attempt
from orchestrator.contracts import StageDefinition
from orchestrator.database import ArtifactRecord, AttemptRecord, OperatorMessageRecord
from orchestrator.persistence import DatabaseAttemptLedger
from orchestrator.quality.catalog import quality_contract_for
from orchestrator.repository import PipelineRepository
from orchestrator.workflows.run import (
    ReviewDecision,
    ReviewInput,
    StageActivityInput,
    StageActivityResult,
)
from orchestrator.workspace import RunWorkspace

PERMITTED = ("quality.verdict", "attempt.retry", "human.ask")

# One Codex session and one queue per run, kept beside the reviews.
SESSION_FILE = "session.id"
TODO_FILE = "todo.md"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def read_session_id(reviews: Path) -> str | None:
    """The Codex session this run has been using, if it has started one."""
    try:
        value = (reviews / SESSION_FILE).read_text().strip()
    except OSError:
        return None
    return value or None


def write_session_id(reviews: Path, session: str) -> None:
    (reviews / SESSION_FILE).write_text(session + "\n")


def append_todo(todo: Path, request: ReviewInput) -> None:
    """Add this attempt to the run's running list of things to judge.

    One list per run, appended to as work arrives, so the agent sees the whole run rather than
    one attempt at a time and can notice a stage failing the same way twice.
    """
    if not todo.exists():
        todo.write_text(
            "# Review queue\n\n"
            "Each line is an attempt this run asked you to judge, oldest first. Mark a line\n"
            "`[x]` once you have decided it and say what you decided. Leave `[ ]` alone.\n\n"
        )
    stamp = datetime.now(UTC).strftime("%H:%M:%S")
    error = f" — {request.attempt_error}" if request.attempt_error else ""
    with todo.open("a") as stream:
        stream.write(
            f"- [ ] {stamp} `{request.node_id}` attempt `{request.attempt_id}` "
            f"{request.attempt_status}{error}\n"
        )


def parameter_lines(definition: dict[str, Any]) -> list[str]:
    """The stage's overridable defaults, written out for the agent that may move them.

    Everything here is already running at the value shown; a retry supplies only the names it
    wants different. Anything not listed cannot be set, and the stage refuses an attempt that
    names it, so the list is also the answer to "what can I even change".
    """
    properties = (definition.get("parameter_schema") or {}).get("properties") or {}
    if not properties:
        return [
            "## Parameters",
            "",
            "This stage takes none. A retry of it repeats the same command, so retry only when",
            "you believe the failure was transient; otherwise fix the output or ask.",
            "",
        ]
    lines = [
        "## Parameters you may set",
        "",
        "These are the stage's own defaults. It is running at these values now, so repeating one",
        "unchanged changes nothing. Move a knob only when you can name the thing you saw and say",
        "why that value addresses it; when you cannot, pass the attempt or ask rather than",
        "guessing at a number.",
        "",
    ]
    for name in sorted(properties):
        rule = properties[name]
        bounds = ""
        if "minimum" in rule or "maximum" in rule:
            bounds = f", {rule.get('minimum', '-')}..{rule.get('maximum', '-')}"
        if "enum" in rule:
            bounds = ", one of " + " | ".join(str(item) for item in rule["enum"])
        default = repr(rule["default"]) if "default" in rule else "unset"
        lines.append(
            f"- `{name}` ({rule.get('type', 'any')}{bounds}) = {default} — "
            f"{rule.get('description', '')}"
        )
    lines += [
        "",
        "A check you are not allowed to relax is not on this list. When the only way past a",
        "stage is to loosen a tolerance or skip a guard, that is `human.ask`, not a retry.",
        "",
    ]
    return lines


def render_instructions(
    packet: AgentTaskPacket,
    criteria: list[dict[str, Any]],
    attempt_status: str = "succeeded",
    todo: Path | None = None,
) -> str:
    latest = packet.attempts[-1].id if packet.attempts else ""
    failed = attempt_status != "succeeded"
    lines = [
        f"# Review stage `{packet.node_id}` of run `{packet.run_id}`",
        "",
        packet.objective,
        "",
        "## What you are judging",
        "",
        "The attempt's output files are under `outputs/` in this directory, and everything else",
        "it recorded is under `attempt/` including `stderr.log`, `stdout.log` and `result.json`.",
        "`task.json` holds the stage definition, every earlier attempt with its hypothesis and",
        "error, and messages from the operator. Look at the real files; do not decide from names.",
        "",
        "## Criteria",
        "",
    ]
    for criterion in criteria:
        roles = ", ".join(criterion["roles"]) or "any output"
        lines.append(f"- **{criterion['id']}** — {criterion['description']} (evidence: {roles})")
    if failed:
        lines += [
            "",
            f"## This attempt {attempt_status}",
            "",
            "Read `attempt/stderr.log` and `attempt/result.json` for the cause. You cannot pass a",
            "stage that did not produce valid outputs: choose `attempt.retry` with parameters you",
            "believe fix the cause, or `human.ask` when the cause is outside the stage's",
            "parameters, such as a bug or an unsuitable input.",
        ]
    lines += [
        "",
        "## You may edit",
        "",
        "You may fix files under `outputs/` directly (same file names and formats). Any change",
        "there becomes a new attempt attributed to you, and a `pass` then selects that attempt.",
        "Do not touch `task.json` or this file.",
        "",
        f"## Finish by writing `{DECISION_FILE}`",
        "",
        "Exactly one JSON object, one of:",
        "",
        "```json",
        f'{{"tool": "quality.verdict", "node_id": "{packet.node_id}", "attempt_id": "{latest}", "verdict": "pass",'
        ' "evidence_artifact_ids": [], "rationale": "what you checked and saw"}',
        "```",
        "`verdict` may be `pass`, `fail` (no way forward without a person) or `needs_human`.",
        "",
        "```json",
        f'{{"tool": "attempt.retry", "node_id": "{packet.node_id}", "attempt_id": "{latest}",'
        ' "hypothesis": "what will be different and why it should work",',
        ' "parameters": {"dilate": 24}}',
        "```",
        "Retrying is cheap relative to a wrong answer: prefer a retry you can justify over",
        "passing something weak.",
        "",
        *parameter_lines(packet.stage_definition),
        "```json",
        f'{{"tool": "human.ask", "node_id": "{packet.node_id}", "question": "what you need decided",'
        ' "artifact_ids": []}',
        "```",
        "",
        "Be specific in rationales: name the frames, values or regions you looked at.",
        "",
        "## If you get stuck",
        "",
        "Nothing here is worth waiting on forever. Give every command you run a timeout, and if",
        "one hangs, kill it and find the answer another way rather than starting it again the",
        "same way. Go quiet for five minutes and you will be interrupted and asked what you are",
        "waiting on; go quiet again and the stage is handed to a person with nothing to read.",
        "",
        "Stopping is allowed and is better than that. When a tool will not run, an input is not",
        "there, or the call is not yours to make, write `human.ask` and finish. The stage then",
        "waits for an operator, who can answer you and let the run carry on; the attempt's",
        "outputs are kept either way. An unanswerable question asked early costs the run far",
        "less than a silent hour.",
    ]
    if todo is not None:
        lines += [
            "",
            "## The run's queue",
            "",
            f"`{todo}` lists every attempt this run has asked you to judge, oldest first. You",
            "are the same session across the whole run, so you have seen the earlier ones. Read",
            "it before deciding: a stage failing the same way twice means the parameter you",
            "changed is not the cause, and a stage you already passed should not be re-argued.",
            "Tick your line `[x]` and say what you decided, so the record survives you.",
            "",
            "## Checking without filling this session",
            "",
            "Reading whole videos or mask archives here costs you the context you need for the",
            "rest of the run. For anything bulky, spawn a fresh agent that reports back a short",
            "answer, for example:",
            "",
            "```bash",
            "codex exec --skip-git-repo-check --sandbox read-only \\",
            "  'Decode outputs/clean.mp4, compare frames 0-20 against attempt/, and reply with",
            "   one line per frame: index, whether a person remains, and the evidence.'",
            "```",
            "",
            "Use its answer as evidence and keep the detail out of here. Measure rather than",
            "guess: a number you took from a file beats an impression of a thumbnail.",
        ]
    if packet.operator_messages:
        lines += ["", "## Operator messages", ""]
        for message in packet.operator_messages:
            lines.append(f"- {message.get('author', 'operator')}: {message.get('message', '')}")
    return "\n".join(lines) + "\n"


class ReviewActivities:
    def __init__(
        self,
        repository: PipelineRepository,
        *,
        workspace_root: Path,
        store: LocalCAS,
        outbox_archive: Archive,
        resolver,
        attempt_ledger: DatabaseAttemptLedger | None,
        harness: HarnessAgent,
    ):
        self.repository = repository
        self.workspace_root = Path(workspace_root).resolve()
        self.store = store
        self.archive = outbox_archive
        self.resolver = resolver
        self.attempt_ledger = attempt_ledger
        self.harness = harness

    @activity.defn(name="review_attempt")
    def review_attempt(self, request: ReviewInput) -> ReviewDecision:
        revised_id = attempt_identifier(request.run_id, activity.info(), suffix=":agent")
        return self.review(request, revised_id)

    def review(self, request: ReviewInput, revised_id: str) -> ReviewDecision:
        definition = StageDefinition.model_validate(request.definition)
        contract = quality_contract_for(definition)
        criteria = [
            {
                "id": criterion.id,
                "description": criterion.description,
                "roles": [getattr(item, "role", str(item)) for item in criterion.evidence],
            }
            for criterion in (contract.agent.criteria if contract.agent else ())
        ]
        # The stage activity already wrote this node as succeeded. Say that it is under review,
        # so the dashboard does not show a finished stage while the agent is still judging it.
        self.repository.set_node_status(request.run_id, request.node_id, "waiting_agent")
        attempts, artifacts, messages = self._load_context(request)
        workspace = RunWorkspace(self.workspace_root, request.run_id)
        workspace.initialize()
        reviews = workspace.root / "reviews"
        reviews.mkdir(parents=True, exist_ok=True)
        session = read_session_id(reviews)
        todo = reviews / TODO_FILE
        append_todo(todo, request)
        scratch = reviews / request.node_id / request.attempt_id
        outputs = scratch / "outputs"
        attempt_files = scratch / "attempt"
        for stale in (outputs, attempt_files):
            if stale.exists():
                shutil.rmtree(stale)
        outputs.mkdir(parents=True, exist_ok=True)
        roles = self._hydrate_outputs(artifacts, outputs, attempt_files)
        before = _snapshot(outputs)
        packet = AgentTaskPacket(
            run_id=request.run_id,
            node_id=request.node_id,
            objective=(
                f"Attempt {request.attempt_id} of stage '{definition.title}' "
                f"{request.attempt_status}"
                + (f" ({request.attempt_error})." if request.attempt_error else ".")
                + " Decide whether it is good enough, fix it if you can, or say what must change."
            ),
            stage_definition=request.definition,
            dependency_state={},
            artifacts=tuple(
                AgentArtifact(
                    id=record.id,
                    role=record.role,
                    media_type=record.media_type,
                    size=record.size,
                    sha256=record.sha256,
                    preview_path=(record.artifact_metadata or {}).get("relative_path"),
                )
                for record in artifacts
            ),
            attempts=tuple(
                AgentAttemptSummary(
                    id=record.id,
                    number=record.number,
                    status=record.status,
                    hypothesis=record.hypothesis,
                    parameters=record.parameters or {},
                    error=record.error,
                    artifact_ids=tuple(record.output_artifact_ids or ()),
                )
                for record in attempts
            ),
            operator_messages=tuple(messages),
            remaining_budget={
                "agent_retries_used": request.agent_retries,
                "attempt_status": request.attempt_status,
                "attempt_error": request.attempt_error,
            },
            permitted_tools=PERMITTED,
        )
        outcome = self.harness.run(
            packet,
            scratch,
            render_instructions(packet, criteria, request.attempt_status, todo),
            session=session,
        )
        if outcome.session:
            write_session_id(reviews, outcome.session)
        policy = self.harness.policy
        decided_by = f"agent:{policy.kind}:{policy.model or 'default'}:{policy.sandbox}"
        if outcome.decision is None:
            reason = outcome.result.error or outcome.decision_error or "no decision"
            if outcome.result.status == "stalled":
                # The stage is left for a person rather than failed: the attempt's outputs are
                # intact and whoever looks can approve it, retry it, or say what the agent was
                # missing. Resuming is their call, not a silent one made here.
                question = (
                    f"The reviewing agent stopped working on this attempt ({reason}). Its "
                    f"transcript is at {outcome.result.transcript_path}. The outputs are "
                    "unchanged, so you can approve the stage, send it back for another "
                    "attempt, or answer whatever it was stuck on."
                )
            else:
                question = f"The reviewing agent could not judge this attempt ({reason})."
            return self._record(
                request,
                ReviewDecision(
                    node_id=request.node_id,
                    attempt_id=request.attempt_id,
                    decision="needs_human",
                    rationale=f"agent gave no usable decision: {reason}",
                    artifacts=request.artifacts,
                    question=question,
                    decided_by=decided_by,
                ),
            )
        call = outcome.decision
        selected_id, selected_artifacts, revised = request.attempt_id, request.artifacts, False
        after = _snapshot(outputs)
        if after != before:
            selected_id, selected_artifacts = self._freeze_revision(
                request, revised_id, outputs, roles, decided_by
            )
            revised = True
        if isinstance(call, SubmitVerdict):
            if call.verdict == "pass":
                decision = ReviewDecision(
                    node_id=request.node_id,
                    attempt_id=selected_id,
                    decision="pass",
                    rationale=call.rationale,
                    artifacts=selected_artifacts,
                    revised=revised,
                    decided_by=decided_by,
                )
            else:
                decision = ReviewDecision(
                    node_id=request.node_id,
                    attempt_id=selected_id,
                    decision="needs_human",
                    rationale=call.rationale,
                    artifacts=selected_artifacts,
                    question=call.rationale,
                    revised=revised,
                    decided_by=decided_by,
                )
        elif isinstance(call, RequestRetry):
            decision = ReviewDecision(
                node_id=request.node_id,
                attempt_id=selected_id,
                decision="retry",
                rationale=call.hypothesis,
                artifacts=selected_artifacts,
                parameters=dict(call.parameters),
                hypothesis=call.hypothesis,
                revised=revised,
                decided_by=decided_by,
            )
        elif isinstance(call, AskHuman):
            decision = ReviewDecision(
                node_id=request.node_id,
                attempt_id=selected_id,
                decision="needs_human",
                rationale=call.question,
                artifacts=selected_artifacts,
                question=call.question,
                revised=revised,
                decided_by=decided_by,
            )
        else:  # pragma: no cover - PERMITTED limits the dispatcher to the three above
            raise TypeError(f"unsupported decision {call.tool}")
        return self._record(request, decision)

    def _load_context(self, request: ReviewInput):
        with self.repository.sessions() as session:
            attempts = session.scalars(
                select(AttemptRecord)
                .where(
                    AttemptRecord.run_id == request.run_id,
                    AttemptRecord.node_id == request.node_id,
                )
                .order_by(AttemptRecord.number)
            ).all()
            artifacts = session.scalars(
                select(ArtifactRecord)
                .where(
                    ArtifactRecord.run_id == request.run_id,
                    ArtifactRecord.producer_attempt_id == request.attempt_id,
                )
                .order_by(ArtifactRecord.created_at)
            ).all()
            messages = [
                {
                    "id": record.id,
                    "author": record.author,
                    "message": record.message,
                    "node_id": record.node_id,
                    "created_at": record.created_at.isoformat(),
                }
                for record in session.scalars(
                    select(OperatorMessageRecord)
                    .where(OperatorMessageRecord.run_id == request.run_id)
                    .order_by(OperatorMessageRecord.created_at)
                ).all()
            ]
            session.expunge_all()
        # Messages recorded by the workflow but not yet in the database are still worth showing.
        seen = {message["id"] for message in messages}
        for message in request.operator_messages:
            if message.get("id") not in seen:
                messages.append(dict(message))
        return attempts, artifacts, messages

    def _hydrate_outputs(self, artifacts, outputs: Path, attempt_files: Path) -> dict[str, str]:
        """Copy the attempt into the scratch tree; return relative path -> role for outputs.

        Outputs land under ``outputs/`` and are the only thing the agent may edit. Everything
        else the attempt recorded, notably ``stderr.log``, ``stdout.log`` and ``result.json``,
        lands under ``attempt/`` read-only-by-convention, because a failed attempt has no useful
        outputs and its logs are the whole evidence.
        """
        roles: dict[str, str] = {}
        for record in artifacts:
            relative = (record.artifact_metadata or {}).get("relative_path")
            if not relative:
                continue
            if relative.startswith("outputs/"):
                target = outputs / Path(relative).relative_to("outputs")
                roles[relative] = record.role
            else:
                target = attempt_files / relative
            if ".." in Path(relative).parts:
                continue
            hydrated = self.resolver.hydrate(record.id, target.parent)
            if hydrated.name != target.name:
                hydrated.rename(target)
        return roles

    def _freeze_revision(
        self, request: ReviewInput, revised_id: str, outputs: Path, roles: dict[str, str], by: str
    ) -> tuple[str, dict[str, list[str]]]:
        workspace = RunWorkspace(self.workspace_root, request.run_id)
        attempt = workspace.create_attempt(
            request.node_id,
            revised_id,
            {
                "runId": request.run_id,
                "nodeId": request.node_id,
                "stageType": request.stage_type,
                "definition": request.definition,
                "selectedInputs": {},
                "revisionOf": request.attempt_id,
                "revisedBy": by,
            },
        )
        stage_input = StageActivityInput(
            run_id=request.run_id,
            node_id=request.node_id,
            stage_type=request.stage_type,
            definition=request.definition,
            selected_inputs={},
            parameters={"hypothesis": f"agent revision of {request.attempt_id}"},
        )
        if self.attempt_ledger:
            self.attempt_ledger.started(stage_input, revised_id)
        shutil.copytree(outputs, attempt.outputs, dirs_exist_ok=True)
        attempt.finalize(
            {
                "status": "succeeded",
                "error": None,
                "command": [],
                "revisionOf": request.attempt_id,
                "revisedAt": datetime.now(UTC).isoformat(),
            }
        )
        manifest = freeze_attempt(attempt, self.store, status="succeeded", roles=roles)
        outbox = UploadOutbox(workspace.outbox, self.store, self.archive)
        outbox.enqueue(manifest)
        outbox.flush()
        artifacts: dict[str, list[str]] = {}
        for frozen in manifest.files:
            if frozen.role != "attempt_file":
                artifacts.setdefault(frozen.role, []).append(frozen.artifact_id)
        if self.attempt_ledger:
            self.attempt_ledger.finished(
                stage_input,
                StageActivityResult(
                    node_id=request.node_id,
                    attempt_id=revised_id,
                    status="succeeded",
                    artifacts=artifacts,
                ),
                manifest,
            )
        return revised_id, artifacts

    NODE_STATUS = {"pass": "succeeded", "retry": "queued", "needs_human": "waiting_human"}

    def _record(self, request: ReviewInput, decision: ReviewDecision) -> ReviewDecision:
        self.repository.set_node_status(
            request.run_id,
            request.node_id,
            self.NODE_STATUS[decision.decision],
            decision.question if decision.decision == "needs_human" else None,
        )
        self.repository.add_quality_decision(
            run_id=request.run_id,
            node_id=request.node_id,
            attempt_id=decision.attempt_id,
            verdict=decision.decision,
            rationale=decision.rationale,
            decided_by=decision.decided_by,
            evidence=[{"revised": decision.revised, "reviewed": request.attempt_id}],
            proposed_parameters=decision.parameters,
            hypothesis=decision.hypothesis,
        )
        return decision
