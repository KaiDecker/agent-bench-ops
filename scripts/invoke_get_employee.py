import argparse
import asyncio
import json

from app.persistence.database import AsyncSessionFactory
from app.tools.gateway import ToolGateway
from app.tools.registry import build_default_registry
from app.tools.schemas import ToolExecutionContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoke the get_employee tool.")

    selectors = parser.add_mutually_exclusive_group(required=True)

    selectors.add_argument("--employee-id")
    selectors.add_argument("--employee-no")
    selectors.add_argument("--name")

    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    arguments = {
        key: value
        for key, value in {
            "employee_id": args.employee_id,
            "employee_no": args.employee_no,
            "name": args.name,
        }.items()
        if value is not None
    }

    gateway = ToolGateway(build_default_registry())

    context = ToolExecutionContext(
        actor_id="local-demo",
        available_tools={"get_employee"},
        permissions={"employee.read"},
    )

    async with AsyncSessionFactory() as session:
        response = await gateway.execute(
            session=session,
            tool_name="get_employee",
            arguments=arguments,
            context=context,
        )

    print(
        json.dumps(
            response.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(async_main())
