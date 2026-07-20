from app.persistence.base import Base
from app.persistence.models import (
    Account,
    Employee,
    EmployeePermission,
    Permission,
    Ticket,
)


def test_core_models_are_registered() -> None:
    expected_tables = {
        Employee.__tablename__,
        Account.__tablename__,
        Permission.__tablename__,
        EmployeePermission.__tablename__,
        Ticket.__tablename__,
    }

    registered_tables = set(Base.metadata.tables)

    assert expected_tables == registered_tables
