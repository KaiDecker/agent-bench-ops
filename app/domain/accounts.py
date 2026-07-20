from sqlalchemy import CheckConstraint, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.base import Base, TimestampMixin


class Account(TimestampMixin, Base):
    """员工登录账号。"""

    __tablename__ = "accounts"

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'disabled', 'locked')",
            name="status_valid",
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

    employee_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "employees.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
        index=True,
    )

    username: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default="active",
        index=True,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
    )
