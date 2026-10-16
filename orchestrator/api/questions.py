"""Read-only question holds, separate from stage results and quality decisions."""

from datetime import UTC, datetime

from orchestrator.journal import Journal


def _time(value):
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
    except (ValueError, TypeError, AttributeError):
        return None


def restore_legacy_question_states(summary: dict, journal: Journal, repository) -> None:
    """Remove an old question projection only when its event history establishes ownership.

    Older builds wrote a waiting node event immediately followed by the agent's question
    message. New questions have a journal marker and never overwrite the underlying node.
    Missing/interleaved history or a later independent write is deliberately left intact.
    This changes only the response, never stored results, attempts or quality decisions.
    """
    questions = [entry for entry in journal.entries() if entry.kind == "question"]
    legacy = [entry for entry in questions if entry.data.get("statusSource") != "journal"]
    candidates = [
        node
        for node in summary["nodes"]
        if node["status"] == "waiting_human"
        and any(
            entry.data.get("step") == node["id"]
            and entry.data.get("question") == node.get("blockedReason")
            for entry in legacy
        )
    ]
    if not candidates:
        return
    events = []
    while batch := repository.events_after(summary["id"], events[-1].sequence if events else 0):
        events.extend(batch)

    def owns_question(index, node_id):
        event = events[index]
        if event.payload.get("status") != "waiting_human" or index + 1 >= len(events):
            return False
        message = events[index + 1]
        reason = event.payload.get("blockedReason")
        matching = [
            entry
            for entry in questions
            if entry.data.get("step") == node_id
            and entry.data.get("question") == reason
            and _time(entry.at) is not None
            and _time(entry.at) <= _time(event.created_at)
        ]
        return bool(
            matching
            and matching[-1].data.get("statusSource") != "journal"
            and message.sequence == event.sequence + 1
            and message.type == "operator_message"
            and message.subject_id == node_id
            and message.payload.get("author") == "agent"
            and message.payload.get("message") == reason
        )

    for node in candidates:
        history = [
            index
            for index, event in enumerate(events)
            if event.type == "node" and event.subject_id == node["id"] and "status" in event.payload
        ]
        if not history:
            continue
        latest = events[history[-1]]
        if (
            latest.payload.get("status") != node["status"]
            or latest.payload.get("blockedReason") != node.get("blockedReason")
            or not node.get("updatedAt")
            or _time(node["updatedAt"]) is None
            or _time(node["updatedAt"]) > _time(latest.created_at)
            or any(
                event.type == "quality" and event.subject_id == node["id"]
                for event in events[history[-1] + 1 :]
            )
        ):
            continue
        while history and owns_question(history[-1], node["id"]):
            history.pop()
        if history and events[history[-1]] is not latest:
            prior = events[history[-1]].payload
            node["status"] = prior["status"]
            node["blockedReason"] = prior.get("blockedReason")


def apply_question_hold(summary: dict, journal: Journal) -> None:
    """A pending question can hold a stage without replacing an independent review hold."""
    pending = journal.pending_question()
    if pending is None:
        return
    for node in summary["nodes"]:
        if node["id"] == pending.data.get("step") and node["status"] != "waiting_human":
            node["status"] = "waiting_human"
            node["blockedReason"] = pending.data.get("question")
