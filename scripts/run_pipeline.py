#!/usr/bin/env python3
"""Submit a durable pipeline run and follow its ordered event stream."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.client import PipelineClient, token_from_environment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--name")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--token-env", default="WANDER_API_TOKEN")
    parser.add_argument(
        "--marble",
        choices=["none", "image", "video", "multi", "both"],
        default="video",
    )
    parser.add_argument("--all-people", action="store_true")
    parser.add_argument("--people", type=int, default=1)
    parser.add_argument("--no-objects", action="store_true")
    parser.add_argument("--reviewed-audio", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--maximum-cost", type=float)
    parser.add_argument("--no-follow", action="store_true")
    args = parser.parse_args(argv)
    client = PipelineClient(args.api, token_from_environment(args.token_env))
    response = client.submit(
        args.clip,
        run_id=args.name,
        options={
            "marble": args.marble,
            "all_people": args.all_people or args.people > 1,
            "people": args.people,
            "objects": not args.no_objects,
            "reviewed_audio": args.reviewed_audio,
            "finetune": args.finetune,
        },
        budget={"maximum_cost": args.maximum_cost} if args.maximum_cost is not None else {},
    )
    print(
        f"submitted {response['id']} ({response['sourceSha256']}); "
        f"dashboard: http://127.0.0.1:5399/dashboard.html"
    )
    if args.no_follow:
        return 0
    status = client.follow(response["id"])
    return 0 if status == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
