"""Migration and relational-schema tests for durable pipeline metadata."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from orchestrator.database import Base

EXPECTED_TABLES = {
    "pipeline_approvals",
    "pipeline_artifacts",
    "pipeline_attempts",
    "pipeline_events",
    "pipeline_nodes",
    "pipeline_operator_messages",
    "pipeline_provider_claims",
    "pipeline_quality_decisions",
    "pipeline_runs",
}


class DatabaseTests(unittest.TestCase):
    def test_metadata_declares_complete_pipeline_schema(self):
        self.assertEqual(set(Base.metadata.tables), EXPECTED_TABLES)
        events = Base.metadata.tables["pipeline_events"]
        self.assertEqual({column.name for column in events.primary_key.columns}, {"id"})
        self.assertTrue(
            any(
                {column.name for column in constraint.columns} == {"run_id", "sequence"}
                for constraint in events.constraints
                if hasattr(constraint, "columns")
            )
        )

    def test_initial_migration_upgrades_and_downgrades(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "pipeline.db"
            url = f"sqlite:///{database}"
            config = Config(root / "alembic.ini")
            with patch.dict(os.environ, {"WANDER_DATABASE_URL": url}):
                command.upgrade(config, "head")
                engine = create_engine(url)
                try:
                    self.assertTrue(
                        EXPECTED_TABLES.issubset(set(inspect(engine).get_table_names()))
                    )
                finally:
                    engine.dispose()
                command.downgrade(config, "base")
                engine = create_engine(url)
                try:
                    remaining = set(inspect(engine).get_table_names())
                    self.assertFalse(EXPECTED_TABLES.intersection(remaining))
                finally:
                    engine.dispose()


if __name__ == "__main__":
    unittest.main()
