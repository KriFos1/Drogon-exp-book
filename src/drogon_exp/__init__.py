"""Drogon book-chapter experiment adapters and schemes."""

# Install the blockwise equivalent of PET's adaptive taper before PET builds
# localization objects from the experiment configs.
from . import localization as _localization  # noqa: F401
