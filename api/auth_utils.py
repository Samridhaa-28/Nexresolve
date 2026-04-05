"""
JWT authentication utilities.
"""
import os
import yaml
from pathlib import Path
from datetime import datetime, timedelta

from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ── Config ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_cfg_path = _ROOT / "configs" / "auth_config.yaml"

with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

SECRET_KEY: str = _cfg["secret_key"]
EXPIRY_HOURS: int = int(_cfg["expiry_hours"])
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Token operations ─────────────────────────────────────────────────────────

def create_token(user_id: str) -> str:
    """Encode a signed JWT that expires after EXPIRY_HOURS."""
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(hours=EXPIRY_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> str:
    """Decode token and return user_id string; raise 401 on any failure."""
    _401 = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise _401
        return str(user_id)
    except JWTError:
        raise _401


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """FastAPI dependency — resolves the Bearer token to a user_id."""
    return verify_token(token)
