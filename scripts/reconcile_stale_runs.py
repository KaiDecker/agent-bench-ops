import argparse
import asyncio
import json
from datetime import timedelta

from app.agent.reconciliation import (
    StaleRunReconciler,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Inspect or reconcile stale AgentRun rows."))

    parser.add_argument(
        "--older-than-minutes",
        type=int,
        default=60,
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect without changing database rows.",
    )

    mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply safe reconciliation actions.",
    )

    parser.add_argument(
        "--mark-inconclusive",
        action="append",
        default=[],
        metavar="OPERATION_ID",
        help=(
            "Explicitly authorize an unknown historical "
            "operation to be closed as inconclusive. "
            "May be supplied more than once."
        ),
    )

    return parser.parse_args()


async def async_main() -> None:
    arguments = parse_arguments()

    if arguments.older_than_minutes <= 0:
        raise ValueError("--older-than-minutes must be positive")

    reconciler = StaleRunReconciler()

    results = await reconciler.reconcile(
        older_than=timedelta(minutes=arguments.older_than_minutes),
        apply=arguments.apply,
        inconclusive_operation_ids=(arguments.mark_inconclusive),
    )

    payload = {
        "mode": ("apply" if arguments.apply else "dry-run"),
        "matched_runs": len(results),
        "applied_runs": sum(1 for result in results if result.applied),
        "results": [result.to_dict() for result in results],
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(async_main())
