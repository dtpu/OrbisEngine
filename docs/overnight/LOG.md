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

Ported Austin's `31a6b69`: video submissions pin `marble-1.1` like image submissions, and all
new worlds explicitly request private permissions. Original author metadata is retained.
These request-contract changes do not establish improved generated geometry.
