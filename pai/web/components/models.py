"""Models component for model discovery and selection.

Single responsibility: Model browsing and switching
- List available models with filtering
- Select/switch active model
- Display model information
"""

import html
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from .base import Component

if TYPE_CHECKING:
    pass


class ModelsComponent(Component):
    """Handles model listing, filtering, and selection."""

    @property
    def component_id(self) -> str:
        return "models"

    def get_routes(self) -> list[Route]:
        return [
            Route("/models/list", self.list_models, methods=["GET"]),
            Route("/models/select", self.select_model, methods=["POST"]),
            Route("/models/search", self.search_models, methods=["GET"]),
            Route("/models/current", self.get_current, methods=["GET"]),
        ]

    async def list_models(self, request: Request) -> HTMLResponse:
        """List available models with optional filtering."""
        session_id = request.query_params.get("session_id", "")
        search = request.query_params.get("search", "").strip()
        limit = int(request.query_params.get("limit", "25"))
        offset = int(request.query_params.get("offset", "0"))

        session = await self.session_manager.get_or_create_session(session_id)

        # Parse search terms
        search_terms = search.split() if search else None

        try:
            result = await session.client.list_models(
                search_terms=search_terms,
                limit=limit if limit > 0 else None,
            )

            html_parts = []

            # Header with counts
            if result.is_filtered:
                html_parts.append(f"""
                <div class="models-header">
                    <span class="model-count">
                        Showing {len(result.models)} of {result.filtered_count} matches
                        (from {result.total_count} total)
                    </span>
                </div>
                """)
            else:
                html_parts.append(f"""
                <div class="models-header">
                    <span class="model-count">
                        Showing {len(result.models)} of {result.total_count} models
                    </span>
                </div>
                """)

            # Model list
            html_parts.append('<div class="model-list">')
            current_model = session.client.config.model_name

            for model_id in result.models:
                is_current = model_id == current_model
                current_class = "model-item current" if is_current else "model-item"
                current_badge = (
                    '<span class="badge">current</span>' if is_current else ""
                )

                html_parts.append(f'''
                <div class="{current_class}"
                     hx-post="/models/select"
                     hx-vals='{{"model_id": "{html.escape(model_id)}", "session_id": "{session.session_id}"}}'
                     hx-target="#current-model"
                     hx-swap="innerHTML">
                    <span class="model-name">{html.escape(model_id)}</span>
                    {current_badge}
                </div>
                ''')

            html_parts.append("</div>")

            # Show more button if limited
            if result.is_limited:
                new_limit = limit + 25
                html_parts.append(f"""
                <div class="load-more">
                    <button hx-get="/models/list?session_id={session.session_id}&search={html.escape(search)}&limit={new_limit}"
                            hx-target="#model-list-container"
                            hx-swap="innerHTML"
                            class="btn btn-secondary">
                        Load more ({result.filtered_count - len(result.models)} remaining)
                    </button>
                </div>
                """)

            return HTMLResponse("".join(html_parts))

        except Exception as e:
            return HTMLResponse(
                f'<div class="error">Error loading models: {html.escape(str(e))}</div>',
                status_code=500,
            )

    async def select_model(self, request: Request) -> HTMLResponse:
        """Select a model as the active model."""
        form = await request.form()
        model_id = str(form.get("model_id", ""))
        session_id = str(form.get("session_id", ""))

        if not model_id:
            return HTMLResponse(
                '<span class="error">No model specified</span>',
                status_code=400,
            )

        session = await self.session_manager.get_session(session_id)
        if not session:
            return HTMLResponse(
                '<span class="error">Session not found</span>',
                status_code=404,
            )

        session.client.set_model(model_id)

        return HTMLResponse(f"""
        <span class="current-model-display">
            <strong>{html.escape(model_id)}</strong>
            <span class="success-indicator">switched</span>
        </span>
        """)

    async def search_models(self, request: Request) -> HTMLResponse:
        """Search models with live filtering (for search-as-you-type)."""
        # Delegates to list_models with search parameter
        return await self.list_models(request)

    async def get_current(self, request: Request) -> HTMLResponse:
        """Get the currently selected model."""
        session_id = request.query_params.get("session_id", "")
        session = await self.session_manager.get_session(session_id)

        if not session:
            return HTMLResponse('<span class="muted">No session</span>')

        model = session.client.config.model_name
        endpoint = session.client.config.name

        return HTMLResponse(f"""
        <span class="current-model-display">
            <strong>{html.escape(model)}</strong>
            <span class="muted">on {html.escape(endpoint)}</span>
        </span>
        """)
