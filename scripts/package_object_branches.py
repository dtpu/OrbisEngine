#!/usr/bin/env python3
"""Package independently retained object branches into one viewer manifest."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for branch in config["branches"]:
        record = branch["description"]
        description = record["description"]
        command = [
            args.python,
            "scripts/package_objects.py",
            "--world",
            config["world"],
            "--fit",
            branch["fit"],
            "--tracks-dir",
            config["tracksDir"],
            "--cameras",
            config["cameras"],
            "--id",
            branch["id"],
            "--label",
            description["label"],
            "--prompt",
            description["segmentWord"],
            "--object-class",
            "thrown",
            "--model",
            branch["model"],
            "--invented",
            "--shape-provenance",
            "INVENTED. Image-to-3D from retained in-flight source crops.",
            "--size-m",
            ",".join(f"{value:.4f}" for value in record["sizeMetres"]),
            "--color-srgb",
            ",".join(f"{value:.3f}" for value in description["colourSRGB"]),
            "--colour-provenance",
            "Measured source-crop pixels and recorded object-description evidence.",
            "--observed-views",
            str(len(record.get("cropsUsed", []))),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
