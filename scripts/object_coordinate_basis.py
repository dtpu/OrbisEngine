"""Coordinate corrections shared by object lifting and packaging.

Fresh people packages have no floor measurement. Their scale is the same 1.70 m
avatar stature prior used by the detector, with no invented floor or drift.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np


def finite_number(value, name, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    return float(value)


def vector(value, name, length=None):
    if not isinstance(value, list) or not value or (length and len(value) != length):
        raise ValueError(f"{name} must be a nonempty numeric array")
    return np.array([finite_number(v, name) for v in value])


@dataclass
class ObjectCoordinateBasis:
    metres_per_world_unit: float
    scale: dict
    constants: dict
    drift_frames: np.ndarray
    drift_offsets: np.ndarray
    provenance: dict

    def drift_at(self, frame):
        return float(np.interp(frame, self.drift_frames, self.drift_offsets))

    def validate_fit(self, fit):
        recorded = fit.get("coordinateBasis")
        if recorded is None:
            # Older fits were exclusively made against floorFit. Never reinterpret
            # one in the new uncorrected frame merely because its scale is similar.
            if self.provenance["mode"] != "floorFit":
                raise ValueError("fit has no coordinateBasis; re-lift for the avatar stature prior")
        elif recorded != self.provenance:
            raise ValueError("fit coordinateBasis conflicts with people.json; re-lift the object")
        if "metresPerWorldUnit" in fit:
            mpu = finite_number(fit["metresPerWorldUnit"], "fit.metresPerWorldUnit", True)
            if not np.isclose(mpu, self.metres_per_world_unit, rtol=1e-9, atol=0):
                raise ValueError("fit metresPerWorldUnit conflicts with people.json")


def resolve_object_basis(people):
    """Accept complete legacy floor corrections or fresh identity placement only."""
    try:
        return _resolve(people)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"incomplete object coordinate basis in people.json: {exc}") from exc


def _resolve(people):
    source = vector(people["sourceIndices"], "sourceIndices")
    if np.any(source < 0) or np.any(source != np.floor(source)) or np.any(np.diff(source) <= 0):
        raise ValueError("sourceIndices must be increasing nonnegative integers")
    rows = people["people"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("people must be a nonempty array")
    legacy = "floorFit" in people  # Present-but-invalid is never a prior fallback.
    if legacy:
        floor = people["floorFit"]
        mpu = finite_number(floor["metresPerWorldUnit"], "floorFit.metresPerWorldUnit", True)
        drift = floor["sharedCameraDrift"]
        samples = vector(drift["samples"], "floorFit.sharedCameraDrift.samples")
        offsets = vector(drift["offsetYUnits"], "floorFit.sharedCameraDrift.offsetYUnits")
        if (
            len(samples) != len(offsets)
            or np.any(samples != np.floor(samples))
            or np.any(samples < 0)
            or np.any(samples >= len(source))
            or np.any(np.diff(samples) <= 0)
        ):
            raise ValueError(
                "floorFit drift needs matching offsets and increasing valid sample indices"
            )
        drift_frames = source[samples.astype(int)]
        levels = [v for k, v in floor["levels"].items() if k.startswith("C_")]
        if len(levels) != 1:
            raise ValueError("floorFit needs exactly one C_ correction level")
        translations = levels[0]["translationY"]
        provenance = dict(mode="floorFit", scaleSource="floorFit.metresPerWorldUnit")
    else:
        height = finite_number(
            people["sharedPlacement"]["bodyHeightUnits"], "sharedPlacement.bodyHeightUnits", True
        )
        mpu = finite_number(1.70 / height, "avatar prior metresPerWorldUnit", True)
        drift_frames, offsets = source, np.zeros(len(source))
        translations = {p["id"]: 0.0 for p in rows}
        provenance = dict(
            mode="avatarStaturePrior",
            scaleSource="1.70 m avatar stature prior / sharedPlacement.bodyHeightUnits",
            staturePriorMetres=1.70,
            bodyHeightUnits=height,
            caveat="Assumed avatar stature; no measured metric anchor, floor fit or camera drift.",
        )
    finite_number(9.80665 / mpu, "gravityUnitsPerS2", True)
    scale, constants, ids = {}, {}, set()
    for person in rows:
        track, pid = person["track"], person["id"]
        if isinstance(track, bool) or not isinstance(track, int) or track < 0 or track in scale:
            raise ValueError("person tracks must be unique nonnegative integers")
        if not isinstance(pid, str) or not pid or pid in ids:
            raise ValueError("person ids must be unique nonempty strings")
        ids.add(pid)
        scale[track] = finite_number(person["registrationScale"], "registrationScale", True)
        scale[track] *= finite_number(
            person.get("sizeCorrection", {}).get("multiplier", 1.0),
            "sizeCorrection.multiplier",
            True,
        )
        finite_number(scale[track], "combined registration scale", True)
        constants[track] = finite_number(translations[pid], f"floorFit translationY[{pid}]")
        transform = person.get("transform", {}) if legacy else person["transform"]
        translation = vector(transform.get("translation", [0, 0, 0]), "transform.translation", 3)
        quaternion = vector(
            transform.get("quaternionXYZW", [0, 0, 0, 1]), "transform.quaternionXYZW", 4
        )
        transform_scale = finite_number(transform.get("scale", 1), "transform.scale", True)
        # Legacy level C replaces the old per-person vertical translation in the
        # viewer. Other rotations/scales/translations have never been supported.
        if (
            np.any(translation[[0, 2]] != 0)
            or (not legacy and translation[1] != 0)
            or np.any(quaternion[:3] != 0)
            or abs(quaternion[3]) != 1
            or transform_scale != 1
        ):
            raise ValueError(f"unsupported nonidentity transform for person {pid}")
    material = dict(
        sourceIndices=source.tolist(),
        people=[
            dict(
                track=p["track"],
                id=p["id"],
                scale=scale[p["track"]],
                constantY=constants[p["track"]],
            )
            for p in rows
        ],
        driftSourceFrames=drift_frames.tolist(),
        driftOffsetYUnits=offsets.tolist(),
    )
    provenance.update(
        metresPerWorldUnit=mpu,
        correctionsSha256=hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest(),
    )
    return ObjectCoordinateBasis(mpu, scale, constants, drift_frames, offsets, provenance)
