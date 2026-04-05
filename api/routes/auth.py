"""
Authentication routes: /auth/register and /auth/login
"""
import yaml
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext

from db.connection import get_db
from db.models import build_user_doc
from api.auth_utils import create_token, get_current_user

# ── Password context ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_cfg_path = _ROOT / "configs" / "auth_config.yaml"

with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_pwd_ctx = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=int(_cfg["bcrypt_rounds"]),
)

router = APIRouter(prefix="/auth")


# ── Request models ────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/register")
async def register(req: RegisterRequest):
    db = get_db()

    if await db.users.find_one({"username": req.username}):
        raise HTTPException(status_code=400, detail="Username already taken")

    if await db.users.find_one({"email": req.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = _pwd_ctx.hash(req.password)
    doc = build_user_doc(req.username, req.email, hashed)
    await db.users.insert_one(doc)

    token = create_token(doc["_id"])
    return {"access_token": token, "token_type": "bearer", "username": req.username}


@router.post("/login")
async def login(req: LoginRequest):
    db = get_db()

    user = await db.users.find_one({"email": req.email})
    if not user or not _pwd_ctx.verify(req.password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(str(user["_id"]))
    return {"access_token": token, "token_type": "bearer", "username": user.get("username", "")}


@router.get("/me")
async def get_me(user_id: str = Depends(get_current_user)):
    db = get_db()
    user = await db.users.find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user["_id"], "username": user["username"], "email": user["email"]}
