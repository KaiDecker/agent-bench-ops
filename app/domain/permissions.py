from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.base import Base, TimestampMixin


class Permission(TimestampMixin, Base):
    """系统权限定义。"""

    __tablename__ = "permissions"

    __table_args__ = (
        CheckConstraint(
            "risk_level IN ('low', 'medium', 'high', 'critical')",
            name="risk_level_valid",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="low",
        server_default="low",
    )

    requires_approval: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )


class EmployeePermission(TimestampMixin, Base):
    """员工与权限之间的授权关系。"""

    __tablename__ = "employee_permissions"

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'revoked')",
            name="status_valid",
        ),
    )

    employee_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "employees.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    permission_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "permissions.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default="active",
        index=True,
    )

    granted_by: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
