# Overnight log

Append-only; the newest entry goes at the bottom.
Add one entry after every experiment, in this shape:

```text
### <local time> - <backlog item>
Tried: <what, and why>
Command: <exact command>
Result: <measured numbers, evidence path under .context/evidence/>
Assumed: <any assumption made instead of asking>
Next: <what follows from this>
```

### Selected Marble upload credential fix

Ported Austin's API-origin credential restriction and transport regressions from `b028c7c`.
Presigned media uploads use their URL authentication and receive no World Labs API key.
This integration does not launch or authorize a replacement world generation.
