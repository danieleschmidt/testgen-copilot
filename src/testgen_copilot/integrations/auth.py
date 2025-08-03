"""Authentication and authorization services for TestGen Copilot."""

from __future__ import annotations

import os
import secrets
import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import base64

import httpx
import jwt
from pydantic import BaseModel

from ..logging_config import get_logger


class AuthProvider(str, Enum):
    """Authentication provider types."""
    LOCAL = "local"
    OAUTH2 = "oauth2"
    GITHUB = "github"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    SAML = "saml"


@dataclass
class UserProfile:
    """User profile information."""
    user_id: str
    email: str
    name: str
    provider: AuthProvider
    roles: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = ["user"]
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AccessToken:
    """Access token information."""
    token: str
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    scope: List[str] = None
    user_id: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at


class JWTManager:
    """JWT token management."""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        token_expiry: timedelta = timedelta(hours=24)
    ):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY") or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.token_expiry = token_expiry
        self.logger = get_logger("testgen_copilot.integrations.auth.jwt")
    
    def generate_token(self, user_profile: UserProfile) -> AccessToken:
        """Generate JWT token for user."""
        now = datetime.now(timezone.utc)
        expires_at = now + self.token_expiry
        
        payload = {
            "sub": user_profile.user_id,
            "email": user_profile.email,
            "name": user_profile.name,
            "provider": user_profile.provider.value,
            "roles": user_profile.roles,
            "iat": now.timestamp(),
            "exp": expires_at.timestamp(),
            "iss": "testgen-copilot"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        self.logger.info("Generated JWT token", {
            "user_id": user_profile.user_id,
            "provider": user_profile.provider.value,
            "expires_at": expires_at.isoformat()
        })
        
        return AccessToken(
            token=token,
            expires_at=expires_at,
            user_id=user_profile.user_id,
            scope=user_profile.roles
        )
    
    def verify_token(self, token: str) -> Optional[UserProfile]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.now(timezone.utc).timestamp() > exp:
                self.logger.warning("Token expired", {"user_id": payload.get("sub")})
                return None
            
            return UserProfile(
                user_id=payload["sub"],
                email=payload["email"],
                name=payload["name"],
                provider=AuthProvider(payload["provider"]),
                roles=payload.get("roles", ["user"])
            )
            
        except jwt.InvalidTokenError as e:
            self.logger.warning("Invalid JWT token", {"error": str(e)})
            return None
        except Exception as e:
            self.logger.error("Token verification failed", {"error": str(e)})
            return None
    
    def refresh_token(self, token: str) -> Optional[AccessToken]:
        """Refresh JWT token if valid."""
        user_profile = self.verify_token(token)
        if user_profile:
            return self.generate_token(user_profile)
        return None


class OAuthHandler:
    """OAuth 2.0 authentication handler."""
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        provider: AuthProvider = AuthProvider.OAUTH2
    ):
        self.client_id = client_id or os.getenv(f"{provider.value.upper()}_CLIENT_ID")
        self.client_secret = client_secret or os.getenv(f"{provider.value.upper()}_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("OAUTH_REDIRECT_URI")
        self.provider = provider
        
        self.logger = get_logger(f"testgen_copilot.integrations.auth.oauth.{provider.value}")
        
        # Provider-specific endpoints
        self.endpoints = self._get_provider_endpoints()
        
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def _get_provider_endpoints(self) -> Dict[str, str]:
        """Get OAuth endpoints for provider."""
        endpoints = {
            AuthProvider.GITHUB: {
                "authorize": "https://github.com/login/oauth/authorize",
                "token": "https://github.com/login/oauth/access_token",
                "user": "https://api.github.com/user"
            },
            AuthProvider.GOOGLE: {
                "authorize": "https://accounts.google.com/o/oauth2/v2/auth",
                "token": "https://oauth2.googleapis.com/token",
                "user": "https://www.googleapis.com/oauth2/v2/userinfo"
            },
            AuthProvider.MICROSOFT: {
                "authorize": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                "token": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                "user": "https://graph.microsoft.com/v1.0/me"
            }
        }
        
        return endpoints.get(self.provider, {})
    
    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """Get OAuth authorization URL."""
        if not self.endpoints.get("authorize"):
            raise ValueError(f"Authorization endpoint not configured for {self.provider}")
        
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": self._get_default_scope()
        }
        
        if state:
            params["state"] = state
        
        # Build URL
        auth_url = self.endpoints["authorize"]
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        
        return f"{auth_url}?{query_string}"
    
    def _get_default_scope(self) -> str:
        """Get default OAuth scope for provider."""
        scopes = {
            AuthProvider.GITHUB: "user:email",
            AuthProvider.GOOGLE: "openid email profile",
            AuthProvider.MICROSOFT: "openid email profile"
        }
        
        return scopes.get(self.provider, "openid email profile")
    
    async def exchange_code(self, code: str, state: Optional[str] = None) -> AccessToken:
        """Exchange authorization code for access token."""
        if not self.endpoints.get("token"):
            raise ValueError(f"Token endpoint not configured for {self.provider}")
        
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri
        }
        
        headers = {"Accept": "application/json"}
        
        response = await self.client.post(
            self.endpoints["token"],
            data=data,
            headers=headers
        )
        response.raise_for_status()
        
        token_data = response.json()
        
        expires_at = None
        if "expires_in" in token_data:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=token_data["expires_in"])
        
        self.logger.info("Exchanged OAuth code for token", {
            "provider": self.provider.value,
            "expires_in": token_data.get("expires_in")
        })
        
        return AccessToken(
            token=token_data["access_token"],
            token_type=token_data.get("token_type", "Bearer"),
            expires_at=expires_at,
            scope=token_data.get("scope", "").split() if token_data.get("scope") else None
        )
    
    async def get_user_profile(self, access_token: AccessToken) -> UserProfile:
        """Get user profile from OAuth provider."""
        if not self.endpoints.get("user"):
            raise ValueError(f"User endpoint not configured for {self.provider}")
        
        headers = {"Authorization": f"{access_token.token_type} {access_token.token}"}
        
        response = await self.client.get(
            self.endpoints["user"],
            headers=headers
        )
        response.raise_for_status()
        
        user_data = response.json()
        
        # Parse user data based on provider
        return self._parse_user_data(user_data, access_token)
    
    def _parse_user_data(self, user_data: Dict[str, Any], access_token: AccessToken) -> UserProfile:
        """Parse user data from OAuth provider."""
        if self.provider == AuthProvider.GITHUB:
            return UserProfile(
                user_id=str(user_data["id"]),
                email=user_data.get("email", ""),
                name=user_data.get("name") or user_data.get("login", ""),
                provider=self.provider,
                metadata={
                    "username": user_data.get("login"),
                    "avatar_url": user_data.get("avatar_url"),
                    "company": user_data.get("company")
                }
            )
        
        elif self.provider == AuthProvider.GOOGLE:
            return UserProfile(
                user_id=user_data["id"],
                email=user_data["email"],
                name=user_data["name"],
                provider=self.provider,
                metadata={
                    "picture": user_data.get("picture"),
                    "verified_email": user_data.get("verified_email")
                }
            )
        
        elif self.provider == AuthProvider.MICROSOFT:
            return UserProfile(
                user_id=user_data["id"],
                email=user_data.get("mail") or user_data.get("userPrincipalName", ""),
                name=user_data.get("displayName", ""),
                provider=self.provider,
                metadata={
                    "job_title": user_data.get("jobTitle"),
                    "office_location": user_data.get("officeLocation")
                }
            )
        
        else:
            # Generic parsing
            return UserProfile(
                user_id=str(user_data.get("id") or user_data.get("sub", "")),
                email=user_data.get("email", ""),
                name=user_data.get("name", ""),
                provider=self.provider
            )
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class APIKeyManager:
    """API key management for service authentication."""
    
    def __init__(self):
        self.logger = get_logger("testgen_copilot.integrations.auth.apikey")
        self.valid_keys = self._load_valid_keys()
    
    def _load_valid_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load valid API keys from environment or config."""
        keys = {}
        
        # Load from environment variable (comma-separated)
        env_keys = os.getenv("VALID_API_KEYS", "")
        if env_keys:
            for key in env_keys.split(","):
                key = key.strip()
                if key:
                    keys[key] = {
                        "name": f"key_{key[:8]}",
                        "created_at": datetime.now(timezone.utc),
                        "permissions": ["read", "write"]
                    }
        
        return keys
    
    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate a new API key."""
        if permissions is None:
            permissions = ["read"]
        
        api_key = f"tgc_{secrets.token_urlsafe(32)}"
        
        self.valid_keys[api_key] = {
            "name": name,
            "created_at": datetime.now(timezone.utc),
            "permissions": permissions,
            "last_used": None
        }
        
        self.logger.info("Generated new API key", {
            "name": name,
            "permissions": permissions,
            "key_prefix": api_key[:12]
        })
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return metadata."""
        if api_key in self.valid_keys:
            key_info = self.valid_keys[api_key].copy()
            
            # Update last used timestamp
            self.valid_keys[api_key]["last_used"] = datetime.now(timezone.utc)
            
            self.logger.debug("Valid API key used", {
                "name": key_info["name"],
                "permissions": key_info["permissions"]
            })
            
            return key_info
        
        self.logger.warning("Invalid API key attempted", {
            "key_prefix": api_key[:12] if len(api_key) > 12 else api_key
        })
        
        return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.valid_keys:
            key_info = self.valid_keys.pop(api_key)
            
            self.logger.info("Revoked API key", {
                "name": key_info["name"],
                "key_prefix": api_key[:12]
            })
            
            return True
        
        return False
    
    def list_api_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without revealing the actual keys)."""
        keys = []
        
        for api_key, info in self.valid_keys.items():
            keys.append({
                "key_prefix": api_key[:12] + "...",
                "name": info["name"],
                "permissions": info["permissions"],
                "created_at": info["created_at"],
                "last_used": info.get("last_used")
            })
        
        return keys