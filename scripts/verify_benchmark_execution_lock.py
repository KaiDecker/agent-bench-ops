import asyncio
import json
import selectors
import sys


async def async_main() -> None:
    from app.benchmark.locking import (
        BenchmarkExecutionLockBusyError,
        postgres_benchmark_execution_lock,
    )
    from app.persistence.database import engine

    second_acquisition_blocked = False
    reacquired_after_release = False

    try:
        async with postgres_benchmark_execution_lock():
            try:
                async with postgres_benchmark_execution_lock():
                    raise RuntimeError("Second lock acquisition unexpectedly succeeded.")

            except BenchmarkExecutionLockBusyError:
                second_acquisition_blocked = True

        async with postgres_benchmark_execution_lock():
            reacquired_after_release = True

        if not second_acquisition_blocked:
            raise RuntimeError("Concurrent lock acquisition was not blocked.")

        if not reacquired_after_release:
            raise RuntimeError("Lock could not be reacquired after release.")

        print(
            json.dumps(
                {
                    "second_acquisition_blocked": (second_acquisition_blocked),
                    "reacquired_after_release": (reacquired_after_release),
                },
                indent=2,
            )
        )

    finally:
        await engine.dispose()


def create_windows_selector_event_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.run(
            async_main(),
            loop_factory=(create_windows_selector_event_loop),
        )

    else:
        asyncio.run(async_main())
