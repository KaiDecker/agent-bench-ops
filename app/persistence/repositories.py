from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.benchmark.schemas import BenchmarkTaskSpec
from app.persistence.platform_models import BenchmarkTask


class BenchmarkTaskRepository:
    """固定评测任务的数据库访问层。"""

    async def upsert(
        self,
        session: AsyncSession,
        spec: BenchmarkTaskSpec,
    ) -> BenchmarkTask:
        """
        根据 task_key 和 version 幂等写入任务。

        已存在时更新任务内容，不创建重复记录。
        """
        result = await session.execute(
            select(BenchmarkTask).where(
                BenchmarkTask.task_key == spec.task_key,
                BenchmarkTask.version == spec.version,
            )
        )

        task = result.scalar_one_or_none()

        values = {
            "dataset_version": spec.dataset_version,
            "name": spec.name,
            "category": spec.category,
            "description": spec.description,
            "user_request": spec.user_request,
            "initial_state": spec.initial_state.model_dump(mode="json"),
            "available_tools": spec.available_tools,
            "expected_state": spec.expected_state,
            "required_events": spec.required_events,
            "forbidden_events": spec.forbidden_events,
            "temporal_rules": spec.temporal_rules,
            "budget": spec.budget.model_dump(),
            "metadata_json": spec.metadata,
            "is_active": spec.is_active,
        }

        if task is None:
            task = BenchmarkTask(
                task_key=spec.task_key,
                version=spec.version,
                **values,
            )
            session.add(task)
        else:
            for field_name, value in values.items():
                setattr(task, field_name, value)

        await session.flush()
        return task
