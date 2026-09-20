"""Let a run's expanded nodes survive losing their scheduler.

Stages that fan out — one person per track, one branch per detected object — create nodes while
the run is in flight. Those nodes existed only in the scheduler's memory, so a resumed run came
back with the original graph and no expansion, and the work they represent was silently dropped.
Recording the stage type alongside the definition is what lets a node be rebuilt from its row.

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
"""

import sqlalchemy as sa
from alembic import op

revision = "e6f7a8b9c0d1"
down_revision = "d5e6f7a8b9c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("pipeline_nodes") as batch_op:
        batch_op.add_column(
            sa.Column("stage_type", sa.String(length=128), nullable=False, server_default="")
        )
    # Existing rows are all base-graph nodes, whose stage type equals their node id.
    op.execute("UPDATE pipeline_nodes SET stage_type = node_id WHERE stage_type = ''")


def downgrade() -> None:
    with op.batch_alter_table("pipeline_nodes") as batch_op:
        batch_op.drop_column("stage_type")
