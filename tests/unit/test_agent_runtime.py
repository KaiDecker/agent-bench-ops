from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from app.agent.runtime import (
    AgentRuntimeResult,
    GraphInvocationOutcome,
    RunStatistics,
)


def test_graph_outcome_detects_pause() -> None:
    outcome = GraphInvocationOutcome(
        state={
            "step_count": 1,
        },
        next_nodes=("tools",),
    )

    assert outcome.is_paused is True


def test_run_statistics_derives_total_tokens() -> None:
    statistics = RunStatistics(
        persisted_step_count=3,
        model_step_count=2,
        tool_step_count=1,
        input_tokens=1562,
        output_tokens=99,
    )

    assert statistics.total_tokens == 1661


def test_runtime_result_serializes_messages() -> None:
    result = AgentRuntimeResult(
        run_id="run_001",
        checkpoint_ref="run_001",
        task_key="employee_lookup_001",
        task_version=1,
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        status="succeeded",
        next_nodes=(),
        total_steps=2,
        total_tool_calls=1,
        persisted_step_count=3,
        input_tokens=1562,
        output_tokens=99,
        latency_ms=2500.0,
        final_response="查询成功。",
        error=None,
        messages=(
            HumanMessage(content="查询张三"),
            AIMessage(content="查询成功。"),
        ),
    )

    payload = result.to_dict()

    assert payload["status"] == "succeeded"
    assert payload["total_steps"] == 2
    assert payload["persisted_step_count"] == 3
    assert payload["total_tokens"] == 1661

    assert payload["messages"][0]["type"] == ("human")
    assert payload["messages"][1]["type"] == "ai"

    assert payload["checkpoint_ref"] == "run_001"
    assert payload["next_nodes"] == []
