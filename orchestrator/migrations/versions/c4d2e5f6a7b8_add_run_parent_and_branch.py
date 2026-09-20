"""Record which parent run and admitted shot a child run belongs to.

The workflow spawns one child run per continuous shot that admission finds. Those runs need
their own rows so stages can record attempts against them, and the parent link lets the
dashboard group them under the run the operator submitted.

Revision ID: c4d2e5f6a7b8
Revises: b3f1c2d4e5a6
"""

import sqlalchemy as sa
from alembic import op

revision = "c4d2e5f6a7b8"
down_revision = "b3f1c2d4e5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("pipeline_runs") as batch_op:
        batch_op.add_column(sa.Column("parent_run_id", sa.String(length=128), nullable=True))
        batch_op.add_column(sa.Column("branch_key", sa.String(length=128), nullable=True))
        batch_op.create_foreign_key(
            "fk_pipeline_runs_parent_run_id",
            "pipeline_runs",
            ["parent_run_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    with op.batch_alter_table("pipeline_runs") as batch_op:
        batch_op.drop_constraint("fk_pipeline_runs_parent_run_id", type_="foreignkey")
        batch_op.drop_column("branch_key")
        batch_op.drop_column("parent_run_id")
