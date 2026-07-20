from sqlalchemy import CheckConstraint, String
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.base import Base, TimestampMixin


class Employee(TimestampMixin, Base):
    """企业员工。"""

    __tablename__ = "employees"

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'inactive', 'terminated')",
            name="status_valid",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    employee_no: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        unique=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    department: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default="active",
        index=True,
    )
