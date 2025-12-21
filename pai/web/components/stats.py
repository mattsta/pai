"""Stats component for session statistics display.

Single responsibility: Display usage statistics
- Token counts
- Cost tracking
- Request metrics
- Auto-refresh capability
"""

import html
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from .base import Component

if TYPE_CHECKING:
    pass


class StatsComponent(Component):
    """Handles session statistics display with auto-refresh."""

    @property
    def component_id(self) -> str:
        return "stats"

    def get_routes(self) -> list[Route]:
        return [
            Route("/stats", self.get_stats, methods=["GET"]),
            Route("/stats/detailed", self.get_detailed_stats, methods=["GET"]),
        ]

    async def get_stats(self, request: Request) -> HTMLResponse:
        """Get compact session statistics."""
        session_id = request.query_params.get("session_id", "")
        session = await self.session_manager.get_session(session_id)

        if not session:
            return HTMLResponse("""
            <div class="stats-compact">
                <span class="stat-item">No active session</span>
            </div>
            """)

        stats = session.client.stats

        return HTMLResponse(f"""
        <div class="stats-compact"
             hx-get="/stats?session_id={session.session_id}"
             hx-trigger="every 5s"
             hx-swap="outerHTML">
            <span class="stat-item" title="Total requests">
                <span class="stat-icon">requests</span>
                <span class="stat-value">{stats.total_requests}</span>
            </span>
            <span class="stat-item" title="Input tokens">
                <span class="stat-icon">in</span>
                <span class="stat-value">{stats.total_input_tokens:,}</span>
            </span>
            <span class="stat-item" title="Output tokens">
                <span class="stat-icon">out</span>
                <span class="stat-value">{stats.total_output_tokens:,}</span>
            </span>
            <span class="stat-item" title="Estimated cost">
                <span class="stat-icon">cost</span>
                <span class="stat-value">${stats.total_cost:.4f}</span>
            </span>
        </div>
        """)

    async def get_detailed_stats(self, request: Request) -> HTMLResponse:
        """Get detailed session statistics."""
        session_id = request.query_params.get("session_id", "")
        session = await self.session_manager.get_session(session_id)

        if not session:
            return HTMLResponse("""
            <div class="stats-panel empty">
                <p>No active session</p>
            </div>
            """)

        stats = session.client.stats
        session_stats = session.get_stats_dict()

        # Calculate averages
        avg_input = (
            stats.total_input_tokens / stats.total_requests
            if stats.total_requests > 0
            else 0
        )
        avg_output = (
            stats.total_output_tokens / stats.total_requests
            if stats.total_requests > 0
            else 0
        )

        return HTMLResponse(f"""
        <div class="stats-panel"
             hx-get="/stats/detailed?session_id={session.session_id}"
             hx-trigger="every 10s"
             hx-swap="outerHTML">
            <h3>Session Statistics</h3>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Current Model</div>
                    <div class="stat-value">{html.escape(session_stats["current_model"])}</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Endpoint</div>
                    <div class="stat-value">{html.escape(session_stats["current_endpoint"])}</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Total Requests</div>
                    <div class="stat-value">{stats.total_requests}</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Session Age</div>
                    <div class="stat-value">{session_stats["session_age"]}</div>
                </div>
            </div>

            <h4>Token Usage</h4>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Input Tokens</div>
                    <div class="stat-value">{stats.total_input_tokens:,}</div>
                    <div class="stat-secondary">avg: {avg_input:,.0f}/request</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Output Tokens</div>
                    <div class="stat-value">{stats.total_output_tokens:,}</div>
                    <div class="stat-secondary">avg: {avg_output:,.0f}/request</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Total Tokens</div>
                    <div class="stat-value">{stats.total_input_tokens + stats.total_output_tokens:,}</div>
                </div>

                <div class="stat-card highlight">
                    <div class="stat-label">Estimated Cost</div>
                    <div class="stat-value">{session_stats["total_cost"]}</div>
                </div>
            </div>
        </div>
        """)
