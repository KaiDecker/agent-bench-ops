from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.base import Base, TimestampMixin


def generate_id() -> str:
    """生成应用侧 UUID 字符串。"""
    return str(uuid4())


class BenchmarkTask(TimestampMixin, Base):
    """固定评测任务定义。"""

    __tablename__ = "benchmark_tasks"

    __table_args__ = (
        CheckConstraint(
            "version > 0",
            name="version_positive",
        ),
        UniqueConstraint(
            "task_key",
            "version",
            name="uq_benchmark_tasks_task_key_version",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )

    task_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
    )

    dataset_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="v1",
        server_default="v1",
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    user_request: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    initial_state: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    available_tools: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    expected_state: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    required_events: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    forbidden_events: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    temporal_rules: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    budget: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, object]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
        index=True,
    )


class AgentRun(TimestampMixin, Base):
    """一次 Agent 评测运行。"""

    __tablename__ = "agent_runs"

    __table_args__ = (
        CheckConstraint(
            (
                "status IN ("
                "'queued', "
                "'running', "
                "'waiting_approval', "
                "'succeeded', "
                "'failed', "
                "'cancelled', "
                "'timed_out'"
                ")"
            ),
            name="status_valid",
        ),
        CheckConstraint(
            "resume_count >= 0",
            name="resume_count_non_negative",
        ),
        CheckConstraint(
            "total_steps >= 0",
            name="total_steps_non_negative",
        ),
        CheckConstraint(
            "total_tool_calls >= 0",
            name="total_tool_calls_non_negative",
        ),
        CheckConstraint(
            "input_tokens >= 0",
            name="input_tokens_non_negative",
        ),
        CheckConstraint(
            "output_tokens >= 0",
            name="output_tokens_non_negative",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )

    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "benchmark_tasks.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    experiment_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="queued",
        server_default="queued",
        index=True,
    )

    model_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    prompt_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="v1",
        server_default="v1",
    )

    agent_strategy: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="react",
        server_default="react",
    )

    memory_strategy: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="none",
        server_default="none",
    )

    input_payload: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    configuration: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    random_seed: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    checkpoint_ref: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    trace_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    resume_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    total_steps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    total_tool_calls: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    output_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    final_response: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class RunStep(TimestampMixin, Base):
    """Agent 运行中的一个执行步骤。"""

    __tablename__ = "run_steps"

    __table_args__ = (
        CheckConstraint(
            "step_no > 0",
            name="step_no_positive",
        ),
        CheckConstraint(
            ("step_type IN ('model', 'tool', 'approval', 'checkpoint', 'system')"),
            name="step_type_valid",
        ),
        CheckConstraint(
            ("status IN ('pending', 'running', 'succeeded', 'failed', 'skipped')"),
            name="status_valid",
        ),
        UniqueConstraint(
            "run_id",
            "step_no",
            name="uq_run_steps_run_id_step_no",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "agent_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    parent_step_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "run_steps.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    step_no: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    step_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
        server_default="pending",
        index=True,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    tool_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    input_payload: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    output_payload: Mapped[dict[str, object] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    output_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    error_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ToolOperation(TimestampMixin, Base):
    """一次工具操作及其副作用执行记录。"""

    __tablename__ = "tool_operations"

    __table_args__ = (
        CheckConstraint(
            "risk_level IN ('low', 'medium', 'high', 'critical')",
            name="risk_level_valid",
        ),
        CheckConstraint(
            (
                "status IN ("
                "'prepared', "
                "'running', "
                "'succeeded', "
                "'failed', "
                "'unknown', "
                "'rejected', "
                "'cancelled'"
                ")"
            ),
            name="status_valid",
        ),
        CheckConstraint(
            "retry_count >= 0",
            name="retry_count_non_negative",
        ),
        UniqueConstraint(
            "run_id",
            "idempotency_key",
            name="uq_tool_operations_run_id_idempotency_key",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )

    operation_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "agent_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    step_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "run_steps.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    tool_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    arguments: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    arguments_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    idempotency_key: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="low",
        server_default="low",
        index=True,
    )

    requires_approval: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )

    is_idempotent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="prepared",
        server_default="prepared",
        index=True,
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    result: Mapped[dict[str, object] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    external_reference: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        index=True,
    )

    error_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class Approval(TimestampMixin, Base):
    """高风险工具操作的人工审批记录。"""

    __tablename__ = "approvals"

    __table_args__ = (
        CheckConstraint(
            ("status IN ('pending', 'approved', 'rejected', 'expired', 'cancelled')"),
            name="status_valid",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "agent_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    tool_operation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "tool_operations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
        server_default="pending",
        index=True,
    )

    approval_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    binding_payload: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    requested_by: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    decided_by: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    decision_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class EvaluationResult(TimestampMixin, Base):
    """一次 Agent 运行的自动评测结果。"""

    __tablename__ = "evaluation_results"

    __table_args__ = (
        CheckConstraint(
            "final_state_score >= 0 AND final_state_score <= 1",
            name="final_state_score_range",
        ),
        CheckConstraint(
            "trace_score >= 0 AND trace_score <= 1",
            name="trace_score_range",
        ),
        CheckConstraint(
            "budget_score >= 0 AND budget_score <= 1",
            name="budget_score_range",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "agent_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
        index=True,
    )

    passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        index=True,
    )

    evaluator_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="v1",
        server_default="v1",
    )

    final_state_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    trace_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    budget_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    scores: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    violations: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    actual_final_state: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
