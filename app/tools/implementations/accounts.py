from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.accounts import Account
from app.tools.schemas import (
    ToolBusinessError,
    ToolDefinition,
    ToolExecutionContext,
    ToolMetadata,
)


class GetAccountArguments(BaseModel):
    """查询账号工具的输入参数。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    employee_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
    )

    username: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_exactly_one_selector(self) -> Self:
        selectors = [
            self.employee_id,
            self.username,
        ]

        provided_count = sum(selector is not None for selector in selectors)

        if provided_count != 1:
            raise ValueError("Exactly one of employee_id or username must be provided")

        return self


class AccountResult(BaseModel):
    """账号数据。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    employee_id: str
    username: str
    status: str
    version: int


class GetAccountResult(BaseModel):
    """查询账号工具的输出。"""

    model_config = ConfigDict(extra="forbid")

    account: AccountResult


async def get_account(
    session: AsyncSession,
    arguments: GetAccountArguments,
    context: ToolExecutionContext,
) -> GetAccountResult:
    """根据员工 ID 或用户名查询账号。"""

    del context

    statement = select(Account)

    if arguments.employee_id is not None:
        statement = statement.where(Account.employee_id == arguments.employee_id)
    else:
        statement = statement.where(Account.username == arguments.username)

    result = await session.execute(statement.order_by(Account.id).limit(2))

    accounts = result.scalars().all()

    if not accounts:
        raise ToolBusinessError(
            code="account_not_found",
            message=("No account matched the provided selector."),
            details=arguments.model_dump(
                mode="json",
                exclude_none=True,
            ),
        )

    if len(accounts) > 1:
        raise ToolBusinessError(
            code="account_ambiguous",
            message=("Multiple accounts matched the provided selector."),
            details={
                "matched_account_ids": [account.id for account in accounts],
            },
        )

    account = accounts[0]

    return GetAccountResult(
        account=AccountResult(
            id=account.id,
            employee_id=account.employee_id,
            username=account.username,
            status=account.status,
            version=account.version,
        )
    )


GET_ACCOUNT_TOOL = ToolDefinition(
    metadata=ToolMetadata(
        name="get_account",
        description=("Query one employee account by employee ID or username."),
        risk_level="low",
        required_permissions={"account.read"},
        requires_approval=False,
        is_idempotent=True,
        read_only=True,
        timeout_seconds=3.0,
    ),
    arguments_model=GetAccountArguments,
    result_model=GetAccountResult,
    handler=get_account,
)
