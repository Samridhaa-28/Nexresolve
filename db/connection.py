"""
MongoDB connection — single client instance for the entire application lifetime.
"""
import os
import yaml
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient

# Resolve config relative to project root (two levels up from this file)
_ROOT = Path(__file__).resolve().parent.parent
_cfg_path = _ROOT / "configs" / "db_config.yaml"

with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_client = AsyncIOMotorClient(_cfg["mongo_uri"])
_db = _client[_cfg["db_name"]]


def get_db():
    """Return the shared Motor database instance."""
    return _db
