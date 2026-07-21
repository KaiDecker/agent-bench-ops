from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.base import Base, TimestampMixin


class Ticket(TimestampMixin, Base):
    """账号和权限操作工单。"""

    __tablename__ = "tickets"

    __table_args__ = (
        CheckConstraint(
            (
                "ticket_type IN ("
                "'permission_grant', "
                "'permission_revoke', "
                "'account_recovery', "
                "'general'"
                ")"
            ),
            name="ticket_type_valid",
        ),
        CheckConstraint(
            (
                "status IN ("
                "'open', "
                "'pending_approval', "
                "'approved', "
                "'rejected', "
                "'resolved', "
                "'cancelled'"
                ")"
            ),
            name="status_valid",
        ),
        CheckConstraint(
            "risk_level IN ('low', 'medium', 'high', 'critical')",
            name="risk_level_valid",
        ),
        CheckConstraint(
            "version > 0",
            name="version_positive",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    source_operation_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        unique=True,
    )

    requester_employee_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "employees.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    target_employee_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "employees.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    ticket_type: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="open",
        server_default="open",
        index=True,
    )

    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="medium",
        server_default="medium",
    )

    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    resolution: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
    )


class TicketMutation(TimestampMixin, Base):
    """一次已经提交的工单更新操作。"""

    __tablename__ = "ticket_mutations"

    __table_args__ = (
        CheckConstraint(
            "previous_version > 0",
            name=("ck_ticket_mutations_previous_version_positive"),
        ),
        CheckConstraint(
            "new_version = previous_version + 1",
            name=("ck_ticket_mutations_version_increment"),
        ),
    )

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    ticket_id: Mapped[str] = mapped_column(
        ForeignKey(
            "tickets.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    operation_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
    )

    previous_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    new_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    change_payload: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
    )

    result_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
    )
