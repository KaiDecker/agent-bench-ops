import argparse
import asyncio
import json
import selectors
import sys
from datetime import timedelta


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Release an expired resume claim for one AgentRun.")
    )

    parser.add_argument(
        "run_id",
        help="AgentRun ID to reconcile.",
    )

    parser.add_argument(
        "--older-than-seconds",
        type=float,
        default=2.0,
        help=("Resume claim age required before it is considered expired."),
    )

    return parser.parse_args()


async def async_main(
    *,
    run_id: str,
    older_than_seconds: float,
) -> None:
    from app.agent.reconciliation import (
        StaleRunReconciler,
    )

    if older_than_seconds <= 0:
        raise ValueError("older-than-seconds must be positive")

    reconciler = StaleRunReconciler()

    results = await reconciler.reconcile(
        older_than=timedelta(seconds=older_than_seconds),
        apply=True,
        run_ids=[
            run_id,
        ],
    )

    print(
        json.dumps(
            [result.to_dict() for result in results],
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def create_windows_selector_event_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


if __name__ == "__main__":
    arguments = parse_arguments()

    if sys.platform == "win32":
        asyncio.run(
            async_main(
                run_id=arguments.run_id,
                older_than_seconds=(arguments.older_than_seconds),
            ),
            loop_factory=(create_windows_selector_event_loop),
        )
    else:
        asyncio.run(
            async_main(
                run_id=arguments.run_id,
                older_than_seconds=(arguments.older_than_seconds),
            )
        )
