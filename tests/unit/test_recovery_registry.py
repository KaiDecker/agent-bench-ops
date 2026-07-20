import pytest

from app.tools.recovery import (
    RecoveryRegistry,
    build_default_recovery_registry,
    recover_create_ticket,
)


def test_default_recovery_registry() -> None:
    registry = build_default_recovery_registry()

    assert registry.names() == ["create_ticket"]
    assert registry.get("create_ticket") is not None


def test_recovery_registry_rejects_duplicate() -> None:
    registry = RecoveryRegistry()
    registry.register(
        "create_ticket",
        recover_create_ticket,
    )

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(
            "create_ticket",
            recover_create_ticket,
        )
