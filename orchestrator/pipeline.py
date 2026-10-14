"""The stage graph shared by submissions, the agent and the clip runner.

Pure planning only: execution and outputs still belong to the run directory.
"""

MARBLE_MODES = ("video", "image", "multi", "both", "none")


def world_half(marble: str):
    """(stages, deps, world_stage) for the scene half.

    ONE generation by default. A marble-1.1 world is 1600 credits, so `video` submits exactly one
    and `both` (two worlds, 3200 credits) has to be asked for by name. `review` is the human gate
    that sits between the clean pass and the spend: the clean pass can leave a removed arm's
    dumbbells hanging in mid-air, and that is only visible to a person looking at the frames.
    """
    if marble not in MARBLE_MODES:
        raise ValueError(f"unknown world mode: {marble}")
    stages, deps = [], {}
    if marble in ("image", "both"):
        stages.append("clean_first")
        deps["clean_first"] = []
    if marble in ("video", "both", "none"):
        stages.append("clean")
        # Both inpainting operations share one paid allowance, so claim them in sequence.
        deps["clean"] = ["clean_first"] if marble == "both" else []
    if marble == "multi":
        stages.append("clean_multi")
        deps["clean_multi"] = ["pi3x"]
    if marble == "none":
        return stages, deps, None
    # Write a source-grounded description before generation. Image modes and prompted video request
    # disable_recaption; this records intent, not proof that the provider retained the supplied text.
    stages.append("world_prompt")
    deps["world_prompt"] = []
    stages.append("review")
    deps["review"] = list(stages[:-1])
    if marble in ("image", "both"):
        stages.append("marble_image")
        deps["marble_image"] = ["clean_first", "review"]
    if marble in ("video", "both"):
        stages.append("marble_video")
        deps["marble_video"] = ["clean", "review"]
    if marble == "multi":
        stages.append("marble_multi")
        deps["marble_multi"] = ["clean_multi", "review"]
    world = {"multi": "marble_multi", "video": "marble_video", "both": "marble_video"}.get(
        marble, "marble_image"
    )
    return stages, deps, world


def _finish_world(stages, deps, world, package, objects, finetune):
    if world:
        tail = {
            "scale_fit": ["pi3x", "frame_align", package, world],
        }
        if finetune:
            tail["finetune"] = ["scale_fit"]
        tail["place_fit"] = ["scale_fit"] + (["finetune"] if finetune else [])
        tail["anchors"] = ["place_fit"]
        tail["verify"] = (
            ["anchors"] + (["objects"] if objects else []) + (["finetune"] if finetune else [])
        )
        stages.extend(tail)
        deps.update(tail)
        if objects and package == "package":
            deps["objects"].append("place_fit")
    return stages, deps


def single_graph(marble: str, objects: bool = True, finetune: bool = True):
    stages, deps, world = world_half(marble)
    stages += ["pi3x", "frame_align", "person_prep", "lhm_frozen", "lhm_motion", "package"]
    deps.update(
        {
            "pi3x": [],
            "frame_align": ["pi3x"],
            "person_prep": [],
            "lhm_frozen": ["person_prep"],
            "lhm_motion": ["lhm_frozen", "pi3x"],
            "package": ["lhm_motion", "frame_align"],
        }
    )
    if objects:
        stages += ["objects"]
        deps["objects"] = ["package"]
    return _finish_world(stages, deps, world, "package", objects, finetune)


def multiperson_graph(limit: int, marble: str, objects: bool = True, finetune: bool = True):
    """Stage list and dependencies for up to `limit` people.

    The real track count is only known once `tracks` has run, so the graph is built for `limit`
    slots and the slots no track claims mark themselves skipped. With one person found, only
    slot 00 runs and `package_people` also writes the compatible single-person layout.
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
    return _finish_world(stages, deps, world, "package_people", objects, finetune)
