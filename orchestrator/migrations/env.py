"""Alembic migration environment for pipeline metadata."""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from orchestrator.database import Base

config = context.config
if database_url := os.environ.get("WANDER_DATABASE_URL"):
    config.set_main_option("sqlalchemy.url", database_url)
if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def configure_connection(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations(configuration: dict[str, str]) -> None:
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(configure_connection)
    await connectable.dispose()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    if "+asyncpg" in configuration["sqlalchemy.url"]:
        asyncio.run(run_async_migrations(configuration))
        return
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        configure_connection(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
