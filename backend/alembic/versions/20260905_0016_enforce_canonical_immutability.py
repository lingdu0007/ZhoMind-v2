"""enforce canonical record and event immutability at the database boundary

Revision ID: 20260905_0016
Revises: 20260905_0015
Create Date: 2026-09-05
"""

from alembic import op

revision = "20260905_0016"
down_revision = "20260905_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE FUNCTION reject_canonical_mutation() RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'canonical records and events are immutable';
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        for table_name in ("canonical_records", "canonical_events"):
            op.execute(
                f"""
                CREATE TRIGGER {table_name}_immutable
                BEFORE UPDATE OR DELETE ON {table_name}
                FOR EACH ROW EXECUTE FUNCTION reject_canonical_mutation();
                """
            )
    elif bind.dialect.name == "sqlite":
        for table_name in ("canonical_records", "canonical_events"):
            for operation in ("UPDATE", "DELETE"):
                op.execute(
                    f"""
                    CREATE TRIGGER {table_name}_immutable_{operation.lower()}
                    BEFORE {operation} ON {table_name}
                    BEGIN
                        SELECT RAISE(ABORT, 'canonical records and events are immutable');
                    END;
                    """
                )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for table_name in ("canonical_records", "canonical_events"):
            op.execute(f"DROP TRIGGER IF EXISTS {table_name}_immutable ON {table_name};")
        op.execute("DROP FUNCTION IF EXISTS reject_canonical_mutation();")
    elif bind.dialect.name == "sqlite":
        for table_name in ("canonical_records", "canonical_events"):
            for operation in ("update", "delete"):
                op.execute(f"DROP TRIGGER IF EXISTS {table_name}_immutable_{operation};")
