import argparse
import asyncio
import json
import selectors
import sys


def non_empty_string(
    value: str,
) -> str:
    normalized = value.strip()

    if not normalized:
        raise argparse.ArgumentTypeError("value cannot be empty")

    return normalized


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Read a persisted AgentBenchOps benchmark experiment.")
    )

    parser.add_argument(
        "experiment_id",
        type=non_empty_string,
        help=("The AgentRun experiment_id to query."),
    )

    parser.add_argument(
        "--include-runs",
        action="store_true",
        help=("Include every individual AgentRun in the JSON output."),
    )

    return parser.parse_args()


async def async_main(
    arguments: argparse.Namespace,
) -> None:
    from app.benchmark.results import (
        ExperimentResultService,
    )
    from app.persistence.database import engine

    try:
        experiment = await ExperimentResultService().get_experiment(
            experiment_id=(arguments.experiment_id)
        )

        if arguments.include_runs:
            payload = experiment.to_dict()
        else:
            payload = experiment.to_summary_dict()

        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    finally:
        await engine.dispose()


def create_windows_selector_event_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def main() -> None:
    arguments = parse_arguments()

    if sys.platform == "win32":
        asyncio.run(
            async_main(arguments),
            loop_factory=(create_windows_selector_event_loop),
        )

    else:
        asyncio.run(async_main(arguments))


if __name__ == "__main__":
    main()
