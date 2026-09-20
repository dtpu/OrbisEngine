"""Make run state reconstructable from the database.

Node selection, the reviewing agent's retry budget and the operator's pause/cancel intent lived
only in the workflow's memory, so they existed solely inside a replay log. A restart could not
rebuild them, which is why changing workflow code meant losing in-flight runs. With these
columns the rows are the source of truth and a run resumes from data.

Revision ID: d5e6f7a8b9c0
Revises: c4d2e5f6a7b8
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "d5e6f7a8b9c0"
down_revision = "c4d2e5f6a7b8"
branch_labels = None
depends_on = None


def _json():
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    with op.batch_alter_table("pipeline_nodes") as batch_op:
        batch_op.add_column(
            sa.Column("selected_artifacts", _json(), nullable=False, server_default="{}")
        )
        batch_op.add_column(
            sa.Column("agent_retries", sa.BigInteger(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column("retry_parameters", _json(), nullable=False, server_default="{}")
        )
    with op.batch_alter_table("pipeline_runs") as batch_op:
        batch_op.add_column(
            sa.Column("paused", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.add_column(
            sa.Column("canceled", sa.Boolean(), nullable=False, server_default=sa.false())
        )


def downgrade() -> None:
    with op.batch_alter_table("pipeline_runs") as batch_op:
        batch_op.drop_column("canceled")
        batch_op.drop_column("paused")
    with op.batch_alter_table("pipeline_nodes") as batch_op:
        batch_op.drop_column("retry_parameters")
        batch_op.drop_column("agent_retries")
        batch_op.drop_column("selected_artifacts")
