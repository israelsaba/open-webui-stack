import logging
from typing import Dict, Optional
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Middleware to verify bearer tokens in the format op_wui_xxx."""
    
    def __init__(self, app, valid_tokens: Dict[str, str]):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            valid_tokens: Dictionary mapping tokens to usernames for logging
        """
        super().__init__(app)
        self.valid_tokens = valid_tokens
        logger.info(f"Bearer token authentication initialized with {len(valid_tokens)} valid tokens")
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and verify bearer token.
        
        Excludes health check and root endpoints from authentication.
        """
        # Skip auth for health check and root endpoints
        if request.url.path in ["/health", "/"]:
            return await call_next(request)
        
        # Get authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            logger.warning(f"Missing authorization header for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if it's a bearer token
        if not auth_header.startswith("Bearer "):
            logger.warning(f"Invalid authorization format for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization format. Expected: Bearer <token>",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Extract token
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Validate token format (should start with op_wui_)
        if not token.startswith("op_wui_"):
            logger.warning(f"Invalid token format for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format. Expected format: op_wui_xxx",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify token exists in valid tokens
        username = self.valid_tokens.get(token)
        if not username:
            logger.warning(f"Invalid token attempted for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Add username to request state for logging
        request.state.username = username
        logger.info(f"Authenticated request from user: {username} for {request.url.path}")
        
        response = await call_next(request)
        return response


def parse_api_keys(api_keys_string: str) -> Dict[str, str]:
    """
    Parse API keys from environment variable format.
    
    Format: username1:token1;username2:token2;...
    
    Args:
        api_keys_string: String containing semicolon-separated username:token pairs
        
    Returns:
        Dictionary mapping tokens to usernames
        
    Example:
        >>> parse_api_keys("alice:op_wui_abc123;bob:op_wui_def456")
        {'op_wui_abc123': 'alice', 'op_wui_def456': 'bob'}
    """
    if not api_keys_string or not api_keys_string.strip():
        logger.warning("No API keys configured")
        return {}
    
    tokens = {}
    pairs = api_keys_string.split(";")
    
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
            
        if ":" not in pair:
            logger.warning(f"Invalid API key format (missing colon): {pair}")
            continue
        
        username, token = pair.split(":", 1)
        username = username.strip()
        token = token.strip()
        
        if not username or not token:
            logger.warning(f"Invalid API key format (empty username or token): {pair}")
            continue
        
        if not token.startswith("op_wui_"):
            logger.warning(f"Token for user {username} does not start with op_wui_")
            continue
        
        tokens[token] = username
        logger.debug(f"Registered token for user: {username}")
    
    return tokens
