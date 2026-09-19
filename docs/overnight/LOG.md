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

### Video-first input policy and submission evidence

The CLI already defaulted to cleaned video. The standalone selector now also defaults to video;
its still-image angular heuristic requires an explicit override. Video submissions retain the
pipeline source description, pin their model and request private permissions. Submitted input
bytes and SHA-256 plus the exact request are saved, and mutation is rejected before upload.
The client accepts the documented media-asset ID and required upload headers, retains legacy ID
compatibility, and keeps API credentials off presigned uploads and redirected requests.

The description brief now separates observed features from unknown regions instead of asserting
that input images pin all visible geometry. Image/multi-image modes remain explicit alternatives.
Combined validation passed 58 offline request/policy/recovery/attempt tests, build/typecheck and
full formatting. These checks use mocked providers. No generation, new visual-quality verdict,
or video-versus-stills comparison was performed; fewer hallucinations remain unverified.
