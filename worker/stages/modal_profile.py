"""Resolve local Modal profile aliases for provider call receipts."""

import os


def effective_modal_profile() -> str | None:
    from modal import config as modal_config

    # Modal 1.5.5 updates this selection for CLI --profile before entering main.
    profile = modal_config._profile
    if profile not in modal_config.config_profiles():
        return None
    # The SDK resolves credential settings from the environment before the profile.
    credential_overrides = (
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_OAUTH_REFRESH_TOKEN",
        "MODAL_OAUTH_CLIENT_ID",
        "MODAL_OAUTH_CLIENT_SECRET",
    )
    if any(name in os.environ for name in credential_overrides):
        return None
    return profile
