import argparse
import asyncio
import json
import selectors
import sys
from typing import Any

from app.evaluation.evaluator import (
    EvaluationReport,
    EvaluationService,
)
from app.persistence.database import engine


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate a terminal AgentRun and persist its EvaluationResult.")
    )

    parser.add_argument(
        "run_id",
        help="AgentRun ID to evaluate.",
    )

    parser.add_argument(
        "--capture-live-state",
        action="store_true",
        help=(
            "Explicitly capture the current business database state. Required for first evaluation."
        ),
    )

    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Print the complete EvaluationReport.",
    )

    return parser.parse_args()


def compact_report(
    report: EvaluationReport,
) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "task_key": report.task_key,
        "task_version": report.task_version,
        "run_status": report.run_status,
        "state_source": report.state_source,
        "passed": report.passed,
        "scores": {
            "overall": report.overall_score,
            "final_state": (report.final_state_score),
            "trace": report.trace_score,
            "temporal": (report.temporal_score),
            "budget": report.budget_score,
        },
        "violations": [
            {
                "oracle": violation.oracle,
                "code": violation.code,
                "message": violation.message,
                "details": violation.details,
            }
            for violation in report.violations
        ],
        "evaluated_at": (report.evaluated_at.isoformat()),
    }


async def async_main(
    arguments: argparse.Namespace,
) -> None:
    try:
        service = EvaluationService()

        report = await service.evaluate_run(
            run_id=arguments.run_id,
            capture_live_state=(arguments.capture_live_state),
        )

        payload = report.to_dict() if arguments.full_json else compact_report(report)

        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        await engine.dispose()


def create_windows_selector_event_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


if __name__ == "__main__":
    args = parse_arguments()

    try:
        if sys.platform == "win32":
            asyncio.run(
                async_main(args),
                loop_factory=(create_windows_selector_event_loop),
            )
        else:
            asyncio.run(async_main(args))

    except Exception as exc:
        print(
            (f"Evaluation failed: {type(exc).__name__}: {exc}"),
            file=sys.stderr,
        )

        raise SystemExit(1) from exc
