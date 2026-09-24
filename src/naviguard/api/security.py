"""naviguard.api.security — optional API-key guard.

Set NAVIGUARD_API_KEY to require an `X-API-Key` header on every endpoint
except /health. Unset (default) leaves the API open for local development.
"""

import secrets

from fastapi import Header, HTTPException

from naviguard.config import APISettings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = APISettings().api_key
    if not expected:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
