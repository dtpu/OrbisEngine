"""Drop the content-keyed uniqueness on pipeline_artifacts.

``freeze_attempt`` derives an artifact ID from run, attempt, relative path and digest, so a
single attempt may legitimately record identical bytes at two paths: stages stage their input
under ``inputs/`` and some republish it under ``outputs/public/``. The old
``UNIQUE (run_id, role, sha256, producer_attempt_id)`` rejected that second row, which failed
every such attempt. The primary key already covers run, attempt, path and digest.

Revision ID: b3f1c2d4e5a6
Revises: 2af5c697ace9
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b3f1c2d4e5a6"
down_revision = "2af5c697ace9"
branch_labels = None
depends_on = None

CONSTRAINT = "pipeline_artifacts_run_id_role_sha256_producer_attempt_id_key"


def _artifacts(*, unique: bool) -> sa.Table:
    """The table as it exists on either side of this migration.

    SQLite cannot drop an inline unnamed constraint, so batch mode rebuilds the table from this
    definition instead of issuing ALTER.
    """
    constraints = (
        [sa.UniqueConstraint("run_id", "role", "sha256", "producer_attempt_id")] if unique else []
    )
    return sa.Table(
        "pipeline_artifacts",
        sa.MetaData(),
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("role", sa.String(length=128), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size", sa.BigInteger(), nullable=False),
        sa.Column("media_type", sa.String(length=256), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("producer_attempt_id", sa.String(length=128), nullable=True),
        sa.Column("contract", sa.String(length=128), nullable=True),
        sa.Column("preview_artifact_id", sa.String(length=128), nullable=True),
        sa.Column(
            "artifact_metadata",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["preview_artifact_id"], ["pipeline_artifacts.id"]),
        sa.ForeignKeyConstraint(["producer_attempt_id"], ["pipeline_attempts.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["pipeline_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        # Declared here so the SQLite batch rebuild recreates it with the table.
        sa.Index("ix_pipeline_artifacts_sha256", "sha256"),
        *constraints,
    )


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.drop_constraint(CONSTRAINT, "pipeline_artifacts", type_="unique")
        return
    with op.batch_alter_table(
        "pipeline_artifacts", copy_from=_artifacts(unique=False), recreate="always"
    ):
        pass


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.create_unique_constraint(
            CONSTRAINT, "pipeline_artifacts", ["run_id", "role", "sha256", "producer_attempt_id"]
        )
        return
    with op.batch_alter_table(
        "pipeline_artifacts", copy_from=_artifacts(unique=True), recreate="always"
    ):
        pass
