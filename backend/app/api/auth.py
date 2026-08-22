"""
NyayaGuide AI — Admin Authentication Dependency
Validates administrative requests via X-Admin-Key header using constant-time comparison.
"""
from typing import Optional
from fastapi import Header, HTTPException, status
from ..config import is_admin_key_valid


async def require_admin_key(
    x_admin_key: Optional[str] = Header(
        default=None,
        alias="X-Admin-Key",
        description="Administrative secret key for Knowledge Base management"
    )
) -> str:
    """
    Dependency that enforces admin authentication for document management routes.
    Raises HTTP 401 Unauthorized if the header is missing or invalid.
    """
    if not x_admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required. Missing 'X-Admin-Key' header."
        )

    if not is_admin_key_valid(x_admin_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid administrative API key."
        )

    return x_admin_key
