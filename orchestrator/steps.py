"""What the pipeline can be asked to do, described for the agent that decides.

This is a catalogue, not a schedule. Each entry names a stage `scripts/run_clip.py` already
knows how to run, says what it is for, what it usually wants done first, and which of its
defaults are worth moving. Nothing here is enforced: the suggested order is advice in the
agent's brief, and the agent is free to run steps in another order, repeat one, or reach past
this file into the underlying scripts when it has a reason to.

The one thing that is not advice is spend. A step marked `paid` costs real money every time it
runs, and `orchestrator/paid.py` still refuses a second launch of an operation that may already
be in flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def knob(kind: str, default: Any, description: str, **bounds: Any) -> dict[str, Any]:
    """One overridable default.

    The value a stage uses when nothing is supplied is written here rather than inside the
    command that consumes it, so an attempt's ``task.json`` shows what the stage really ran with
    and a reviewing agent can change one knob without restating the rest. ``description`` is for
    that reader: say what the knob does and which symptom would justify moving it.
    """
    rule = {"type": kind, "description": description, **bounds}
    if default is not None:
        rule["default"] = default
    return rule


def tunables(**knobs: dict[str, Any]) -> dict[str, Any]:
    """A stage's overridable defaults. Anything not named here cannot be set."""
    return {"type": "object", "properties": dict(knobs), "additionalProperties": False}


ADMISSION_PARAMETERS = tunables(
    cut_threshold=knob(
        "number",
        0.10,
        "scene score above which a frame is a candidate cut; raise it when a pan or a flash "
        "is being read as a cut, lower it when a real cut is being missed",
        minimum=0.0,
        maximum=1.0,
    ),
    min_seconds=knob(
        "number",
        1.5,
        "shots shorter than this are reported as too short to use",
        minimum=0.1,
        maximum=60.0,
    ),
)
CLEAN_PARAMETERS = tunables(
    fps=knob(
        "number",
        12,
        "frames per second the clip is cleaned at",
        minimum=1,
        maximum=60,
    ),
    dilate=knob(
        "integer",
        20,
        "pixels the person mask grows by before inpainting; raise it when a halo, a hand or a "
        "strand of hair survives the clean",
        minimum=0,
        maximum=200,
    ),
    bottom_extra=knob(
        "integer",
        40,
        "extra downward margin on the mask, for the contact shadow under the feet",
        minimum=0,
        maximum=400,
    ),
    lama_px=knob(
        "integer",
        960,
        "long edge LaMa inpaints at; larger is sharper and slower",
        minimum=256,
        maximum=2048,
    ),
    moved_mask=knob(
        "boolean",
        False,
        "clean everything that moved rather than only people: cast shadow, carried object, "
        "passer-by. Off by default, so an ordinary clip is unchanged",
    ),
)
WORLD_PROMPT_PARAMETERS = tunables(
    samples=knob(
        "integer",
        6,
        "frames sampled from the clip to describe the room from",
        minimum=1,
        maximum=32,
    ),
    model=knob(
        "string",
        "gpt-6-astra",
        "vision model that writes the prompt",
    ),
)
MARBLE_PARAMETERS = tunables(
    seed=knob(
        "integer",
        7,
        "generation seed for the image and multi-view modes; the video mode ignores it",
        minimum=0,
        maximum=2147483647,
    ),
)
MARBLE_POLL_PARAMETERS = tunables(
    interval=knob(
        "integer",
        60,
        "seconds between polls of a submitted world; this never resubmits and never spends",
        minimum=5,
        maximum=600,
    ),
)
TRACK_PARAMETERS = tunables(
    fps=knob(
        "number",
        12,
        "frames per second the clip is tracked at; the frame indices in the tracks are counted "
        "at this rate, so moving it away from the rate the clip was cleaned at puts the people "
        "on a different timeline from the world",
        minimum=1,
        maximum=60,
    ),
    det_thresh=knob(
        "number",
        0.15,
        "MultiHMR detection threshold; lower it when a person present in the frames is missing "
        "from the tracks, raise it when furniture is being tracked as a person",
        minimum=0.01,
        maximum=0.95,
    ),
)
PERSON_PREP_PARAMETERS = tunables(
    method=knob(
        "string",
        "maskrcnn",
        "segmenter for the reference person mask",
        enum=["segformer", "maskrcnn"],
    ),
    frame=knob(
        "integer",
        None,
        "source frame to build the avatar from; omit to let the frame scorer choose, set it "
        "when the chosen frame is occluded or motion-blurred",
        minimum=0,
    ),
    score_stride=knob(
        "integer",
        3,
        "frame stride for the reference-frame scorer; 1 scores every frame and is slower",
        minimum=1,
        maximum=30,
    ),
    dilate=knob(
        "integer",
        2,
        "pixels the reference mask grows by; raise it when the cutout clips the subject",
        minimum=0,
        maximum=64,
    ),
)
SCALE_PARAMETERS = tunables(
    scale_probes=knob(
        "integer",
        5,
        "depth-ratio probes the scale fit is allowed",
        minimum=1,
        maximum=20,
    ),
    scale0=knob(
        "number",
        None,
        "skip the fit and use this SfM-to-Marble scale; only with a measured value in hand",
        minimum=0.0001,
    ),
)
ANCHOR_PARAMETERS = tunables(
    ruler_tol=knob(
        "number",
        0.15,
        "how far the world ruler and the avatar ruler may disagree before the stage fails",
        minimum=0.01,
        maximum=1.0,
    ),
    ruler_vlm=knob(
        "boolean",
        False,
        "also ask the VLM to name a standard-size object for the world ruler; costs one image "
        "call per sampled frame",
    ),
)
FINETUNE_PARAMETERS = tunables(
    ft_iters=knob(
        "integer",
        4000,
        "fine-tune iterations",
        minimum=100,
        maximum=40000,
    ),
    ft_mask_dilate=knob(
        "integer",
        16,
        "person-mask dilation for the fine-tune export; raise it when fragments of the subject "
        "get baked into the world",
        minimum=0,
        maximum=200,
    ),
)
OBJECT_DETECT_PARAMETERS = tunables(
    joint=knob(
        "integer",
        21,
        "SMPL joint index of the throwing hand the flight is measured from",
        minimum=0,
        maximum=51,
    ),
    dilate=knob(
        "integer",
        9,
        "pixels the person mask grows by before an object is looked for beside it",
        minimum=0,
        maximum=64,
    ),
    min_area=knob("number", 20.0, "smallest object blob in pixels", minimum=1.0),
    max_area=knob("number", 6000.0, "largest object blob in pixels", minimum=10.0),
    max_gap=knob(
        "integer",
        2,
        "frames an object may vanish for and still be one flight",
        minimum=0,
        maximum=30,
    ),
    min_len=knob("integer", 6, "shortest flight kept, in frames", minimum=2, maximum=300),
    min_travel_px=knob(
        "number",
        150.0,
        "a flight must move this far in pixels; lower it when a real short throw is dropped",
        minimum=0.0,
    ),
    hand_px=knob(
        "number",
        45.0,
        "how near the hand a flight must start to count as thrown",
        minimum=0.0,
    ),
)
OBJECT_LIFT_PARAMETERS = tunables(
    joint=knob(
        "integer",
        21,
        "SMPL joint index of the throwing hand",
        minimum=0,
        maximum=51,
    ),
    hand_sigma_m=knob(
        "number",
        0.12,
        "how far, in metres, the fitted release point may sit from the hand",
        minimum=0.001,
        maximum=2.0,
    ),
)
OBJECT_DESCRIBE_PARAMETERS = tunables(
    n_crops=knob("integer", 6, "crops shown to the describing model", minimum=1, maximum=24),
    upscale=knob(
        "integer",
        6,
        "how much each crop is enlarged before it is shown; a small fast object needs more",
        minimum=1,
        maximum=16,
    ),
)
OBJECT_SHAPE_PARAMETERS = tunables(
    refine_strength=knob(
        "number",
        0.85,
        "how far the refiner may move from the source crop; lower it to keep the real object's "
        "appearance, raise it when the crop is too small to generate from",
        minimum=0,
        maximum=1,
    ),
    refine_first=knob(
        "boolean",
        True,
        "refine the crop into a clean product image before lifting it to 3D",
    ),
)


class RunOptions(BaseModel):
    """What a run was asked for. The same names the dashboard has always sent.

    These are run-wide: every legacy command for this run carries them, unlike a step's own
    knobs which the agent sets per call.
    """

    model_config = ConfigDict(extra="forbid")

    marble: Literal["video", "image", "multi", "both", "none"] = "video"
    people: int = Field(default=1, ge=1, le=16)
    all_people: bool = False
    objects: bool = True
    finetune: bool = False


@dataclass(frozen=True)
class Step:
    """One thing the pipeline can be asked to do.

    ``after`` is what this step usually wants finished first. run_clip.py enforces its own
    dependencies against the run directory, which is the real check; this is here so the agent
    can plan rather than discover the order by being refused.
    """

    name: str
    summary: str
    writes: str
    after: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    paid: bool = False
    # Environment this step cannot run without. Modal authenticates through ~/.modal.toml, so
    # a Modal step names nothing here; Marble and OpenAI read a key and fail instantly.
    credentials: tuple[str, ...] = ()
    # Said to the agent in its own words when the step is worth a second thought.
    caution: str = ""


STEPS: dict[str, Step] = {
    step.name: step
    for step in (
        Step(
            name="clean",
            summary="Remove the people from the clip, leaving the room they were in",
            writes="<name>-clean.mp4 in clips/, masks.npz and clean.json in the run directory",
            parameters=CLEAN_PARAMETERS,
            caution=(
                "Look at the cleaned frames before spending on a world. Inpainting can take a "
                "railing or a flight of stairs with the person, and that is only visible to "
                "someone looking."
            ),
        ),
        Step(
            name="clean_first",
            summary="Clean a single frame, for the image world modes",
            writes="first/f_0000.png and clean-first.json",
            parameters=CLEAN_PARAMETERS,
        ),
        Step(
            name="clean_multi",
            summary="Clean the measured set of frames the multi-view world mode submits",
            writes="clean-multi/ and clean-multi.json",
            parameters=CLEAN_PARAMETERS,
        ),
        Step(
            name="world_prompt",
            summary="Write a source-grounded description of the room for the world generator",
            writes="prompt.json",
            after=("clean",),
            parameters=WORLD_PROMPT_PARAMETERS,
            paid=True,
            caution="Costs one vision call per sampled frame.",
            credentials=("OPENAI_API_KEY",),
        ),
        Step(
            name="review",
            summary="Build the contact sheets that get looked at before a world is paid for",
            writes="review/ (sampled frames from the cleaned clip)",
            after=("clean",),
            caution=(
                "run_clip.py stops here unless it is told to carry on, because 1600 credits "
                "are about to be spent on whatever these frames show. Look at them first; "
                "`--gate-pass` is how you say you have and that they are worth it."
            ),
        ),
        Step(
            name="marble_video",
            summary="Generate the static world from the cleaned clip, and fetch it when it is ready",
            writes="marble/<world>-video.spz, world.spz and a thumbnail",
            after=("clean", "world_prompt", "review"),
            parameters=MARBLE_PARAMETERS | MARBLE_POLL_PARAMETERS,
            paid=True,
            caution=(
                "1600 credits per generation, and a quality failure does not authorise another. "
                "One world per new source clip. If a submission already exists, poll it -- never "
                "resubmit, because a resubmit after the operation exists buys a second world."
            ),
            credentials=("WLT_API_KEY",),
        ),
        Step(
            name="pi3x",
            summary="Solve the camera path and dense per-frame geometry from the source clip",
            writes="pi3x/cameras.json, pi3x/frame_*.ply, pi3x/anchors.npz, pose-guard.json",
            paid=True,
            caution=(
                "The pose guard raises when the camera teleports: a soft cut, or a solve that "
                "did not register. Either way nothing may be placed against those poses."
            ),
        ),
        Step(
            name="frame_align",
            summary="Find gravity in the solved frame so everything downstream is levelled",
            writes="pi3x/framealign.json",
            after=("pi3x",),
            caution=(
                "Everything that measures against the cameras reads framealign.json from beside "
                "cameras.json. Without it the whole clip inherits the phone's pitch at frame 0."
            ),
        ),
        Step(
            name="tracks",
            summary="Track every visible person and estimate their pose per sample",
            writes="tracks/tracks.json, tracks/track_NN/motion.json, tracks/masks.npz",
            after=("pi3x",),
            parameters=TRACK_PARAMETERS,
            paid=True,
        ),
        Step(
            name="person_prep",
            summary="Pick and cut out the source frame the avatar is built from",
            writes="prepared-person/ (source.png, mask.png, prepared.json, frame-scores.json)",
            after=("tracks",),
            parameters=PERSON_PREP_PARAMETERS,
            caution="Slow on CPU: it scores candidate frames with two detectors.",
        ),
        Step(
            name="lhm_frozen",
            summary="Reconstruct the canonical person from that one frame",
            writes="lhm-frozen/ (canonical-state.pt and its manifests)",
            after=("person_prep",),
            paid=True,
        ),
        Step(
            name="lhm_motion",
            summary="Animate the canonical person along the poses estimated from the clip",
            writes="lhm-motion/ (sequence.json, motion.json, frame_*.ply)",
            after=("lhm_frozen", "pi3x"),
            paid=True,
            caution=(
                "The sampling schedule has to match the cameras: both are derived from the clip, "
                "and a container that overstates its frame count makes them disagree."
            ),
        ),
        Step(
            name="package",
            summary="Package the animated person and the original playback into a viewer world",
            writes="public/worlds/<name>-4d/",
            after=("lhm_motion", "frame_align"),
        ),
        Step(
            name="package_people",
            summary="Package every tracked person into one viewer world",
            writes="public/worlds/<name>-4d/",
            after=("lhm_motion", "frame_align"),
        ),
        Step(
            name="objects",
            summary="Find, lift, describe and shape the objects that are neither body nor room",
            writes="public/worlds/<name>-4d/objects.json and the object folders",
            after=("package", "tracks"),
            parameters=OBJECT_DETECT_PARAMETERS
            | OBJECT_LIFT_PARAMETERS
            | OBJECT_DESCRIBE_PARAMETERS
            | OBJECT_SHAPE_PARAMETERS,
            paid=True,
            caution="Generating a shape is a paid Modal call per object.",
            credentials=("OPENAI_API_KEY",),
        ),
        Step(
            name="scale_fit",
            summary="Fit the scale between the solved geometry and the generated world",
            writes="scale-fit.json",
            after=("pi3x", "frame_align", "package", "marble_video"),
            parameters=SCALE_PARAMETERS,
        ),
        Step(
            name="place_fit",
            summary="Place the people and objects inside the world at that scale",
            writes="public/worlds/<name>-4d/placement.json",
            after=("scale_fit",),
        ),
        Step(
            name="anchors",
            summary="Check the placement against an external sense of size",
            writes="anchors.json",
            after=("place_fit",),
            parameters=ANCHOR_PARAMETERS,
            paid=True,
            caution=(
                "The tolerance and the bypass flags are an operator's call, not a retry knob. If "
                "the only way past this is to loosen a guard, ask."
            ),
            credentials=("OPENAI_API_KEY",),
        ),
        Step(
            name="finetune",
            summary="Fine-tune the generated world against the source frames",
            writes="finetuned.spz and its receipt",
            after=("scale_fit",),
            parameters=FINETUNE_PARAMETERS,
            paid=True,
            caution="Needs a configured fine-tune host (WANDER_GPU_BOX); there may not be one.",
        ),
        Step(
            name="verify",
            summary="Render the static world and compare it against the source",
            writes="the review record in state.json",
            after=("scale_fit",),
        ),
    )
}


# Stages run_clip.py names once per person when a run asks for more than one.
PER_PERSON = ("person_prep", "lhm_frozen", "lhm_motion")


def base_step(name: str) -> str:
    """The catalogue entry behind a per-person stage name: person_prep_00 -> person_prep."""
    head, _, tail = name.rpartition("_")
    return head if tail.isdigit() and head in PER_PERSON else name


def for_person(name: str, index: int) -> str:
    return f"{name}_{index:02d}"


def planned_steps(options: "RunOptions | dict | None" = None) -> list[str]:
    """The steps this run probably wants, in an order that holds together.

    A suggestion, shown in the dashboard so a run is not an empty page and given to the agent
    as a starting point. The agent may run something not on this list, and the journal adds a
    row when it does.
    """
    if options is None:
        options = RunOptions()
    elif isinstance(options, dict):
        options = RunOptions.model_validate(options)
    many = options.all_people or options.people > 1
    wanted = ["pi3x", "frame_align", "tracks", *PER_PERSON]
    wanted.append("package_people" if many else "package")
    if options.marble == "none":
        wanted.insert(0, "clean")
    else:
        wanted = ["clean", "world_prompt", f"marble_{options.marble}", *wanted]
        wanted += ["scale_fit", "place_fit", "anchors", "verify"]
        if options.finetune:
            wanted.append("finetune")
    if options.objects:
        wanted.append("objects")
    return suggested_order([name for name in dict.fromkeys(wanted) if name in STEPS])


def people_steps(names: list[str], people: int, *, multiperson: bool) -> list[str]:
    """The plan with its per-person stages named once per person.

    run_clip.py calls them `person_prep_00`, `lhm_frozen_00` and so on whenever the run asked
    for the multiperson graph -- including when tracking then finds a single person, which is
    why the count and the shape are separate arguments. The catalogue describes them by their
    base name; how many there really are is only known once tracking has run.
    """
    if not multiperson:
        return list(names)
    people = max(1, people)
    expanded: list[str] = []
    for name in names:
        if name in PER_PERSON:
            expanded.extend(for_person(name, index) for index in range(people))
        else:
            expanded.append(name)
    return expanded


# The many-person packager stands in for the single-person one: a step that wants "package"
# done is satisfied by either.
SAME_AS = {"package_people": "package"}


def suggested_order(names: list[str] | None = None) -> list[str]:
    """The given steps, each after the ones it says it wants.

    Advice for the brief and for the dashboard. Ties are broken by the order they were asked
    for rather than by name, so a plan reads the way its author meant it to. A cycle or an
    unknown name is the catalogue's problem, not the agent's, so this raises rather than
    quietly emitting an order that cannot be run.
    """
    wanted = list(STEPS) if names is None else list(names)
    unknown = sorted(set(wanted) - set(STEPS))
    if unknown:
        raise KeyError(f"not steps this pipeline has: {', '.join(unknown)}")
    satisfied_by = {SAME_AS.get(name, name) for name in wanted} | set(wanted)
    ordered: list[str] = []
    done: set[str] = set()
    while len(ordered) < len(wanted):
        ready = [
            name
            for name in wanted
            if name not in done
            and all(need in done or need not in satisfied_by for need in STEPS[name].after)
        ]
        if not ready:
            remaining = [name for name in wanted if name not in done]
            raise ValueError(f"steps wait on each other: {', '.join(remaining)}")
        for name in ready:
            ordered.append(name)
            done.add(name)
            done.add(SAME_AS.get(name, name))
    return ordered
