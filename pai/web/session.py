"""Session management for web interface.

Each browser session gets its own isolated pai client and conversation state.
Sessions are self-managing with automatic cleanup on expiration.
"""

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..client import PolyglotClient
from ..models import (
    Conversation,
    EndpointConfig,
    PolyglotConfig,
    RuntimeConfig,
    SessionStats,
)
from ..pricing import PricingService


@dataclass
class WebSession:
    """Encapsulates all state for a single web user session.

    Self-managing: tracks its own activity and knows when it's expired.
    """

    session_id: str
    client: PolyglotClient
    conversation: Conversation
    runtime_config: RuntimeConfig
    toml_config: PolyglotConfig
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    _generation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Session settings
    session_timeout_seconds: int = 3600  # 1 hour default

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    @property
    def is_expired(self) -> bool:
        """Check if session has expired due to inactivity."""
        return (time.time() - self.last_activity) > self.session_timeout_seconds

    @property
    def age_seconds(self) -> float:
        """How long this session has existed."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """How long since last activity."""
        return time.time() - self.last_activity

    def get_stats_dict(self) -> dict[str, Any]:
        """Get session statistics as a dictionary for rendering."""
        stats = self.client.stats
        return {
            "total_requests": stats.total_requests,
            "total_input_tokens": stats.total_input_tokens,
            "total_output_tokens": stats.total_output_tokens,
            "total_cost": f"${stats.total_cost:.4f}",
            "session_age": f"{self.age_seconds / 60:.1f} min",
            "current_model": self.client.config.model_name,
            "current_endpoint": self.client.config.name,
        }


class SessionManager:
    """Manages web sessions with automatic cleanup.

    Single responsibility: session lifecycle management.
    Self-managing: runs background cleanup task.
    """

    def __init__(
        self,
        toml_config: PolyglotConfig,
        pricing_service: PricingService,
        cleanup_interval_seconds: int = 300,  # 5 minutes
    ):
        self._sessions: dict[str, WebSession] = {}
        self._toml_config = toml_config
        self._pricing_service = pricing_service
        self._cleanup_interval = cleanup_interval_seconds
        self._cleanup_task: asyncio.Task[None] | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the session manager and its background tasks."""
        self._http_client = httpx.AsyncClient(timeout=60.0)
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the session manager and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._http_client:
            await self._http_client.aclose()

        self._sessions.clear()

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            await asyncio.sleep(self._cleanup_interval)
            await self._cleanup_expired()

    async def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        async with self._lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if session.is_expired
            ]
            for sid in expired:
                del self._sessions[sid]

    def _generate_session_id(self) -> str:
        """Generate a secure random session ID."""
        return secrets.token_urlsafe(32)

    async def create_session(self) -> WebSession:
        """Create a new session with fresh client and conversation."""
        if not self._http_client:
            raise RuntimeError("SessionManager not started")

        session_id = self._generate_session_id()

        # Create runtime config with defaults
        runtime_config = RuntimeConfig(
            max_tokens=4096,
            temperature=0.7,
            stream=True,
            verbose=False,
        )

        # Get default endpoint config
        endpoint_name = self._toml_config.default_endpoint or "openai"
        endpoint_config = self._toml_config.get_endpoint(endpoint_name)
        if not endpoint_config:
            # Fallback to first available endpoint
            if self._toml_config.endpoints:
                endpoint_config = list(self._toml_config.endpoints.values())[0]
            else:
                raise ValueError("No endpoints configured")

        # Create client
        client = PolyglotClient(
            runtime_config=runtime_config,
            toml_config=self._toml_config,
            http_session=self._http_client,
            pricing_service=self._pricing_service,
            version="web",
        )

        # Create conversation
        conversation = Conversation()

        session = WebSession(
            session_id=session_id,
            client=client,
            conversation=conversation,
            runtime_config=runtime_config,
            toml_config=self._toml_config,
        )

        async with self._lock:
            self._sessions[session_id] = session

        return session

    async def get_session(self, session_id: str) -> WebSession | None:
        """Get a session by ID, returning None if not found or expired."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                session.touch()
                return session
            elif session and session.is_expired:
                del self._sessions[session_id]
        return None

    async def get_or_create_session(self, session_id: str | None) -> WebSession:
        """Get existing session or create a new one."""
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session
        return await self.create_session()

    @property
    def active_session_count(self) -> int:
        """Number of active (non-expired) sessions."""
        return sum(1 for s in self._sessions.values() if not s.is_expired)
