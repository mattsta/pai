"""Settings component for runtime configuration.

Single responsibility: Manage runtime settings
- Temperature control
- Max tokens
- Endpoint switching
- Other runtime options
"""

import html
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from .base import Component

if TYPE_CHECKING:
    pass


class SettingsComponent(Component):
    """Handles runtime configuration settings."""

    @property
    def component_id(self) -> str:
        return "settings"

    def get_routes(self) -> list[Route]:
        return [
            Route("/settings", self.get_settings, methods=["GET"]),
            Route("/settings/update", self.update_settings, methods=["POST"]),
            Route("/settings/endpoints", self.list_endpoints, methods=["GET"]),
            Route("/settings/switch-endpoint", self.switch_endpoint, methods=["POST"]),
        ]

    async def get_settings(self, request: Request) -> HTMLResponse:
        """Get the settings panel."""
        session_id = request.query_params.get("session_id", "")
        session = await self.session_manager.get_or_create_session(session_id)

        config = session.runtime_config

        return HTMLResponse(f'''
        <div class="settings-panel" id="settings-panel">
            <h3>Settings</h3>

            <form hx-post="/settings/update"
                  hx-target="#settings-feedback"
                  hx-swap="innerHTML">
                <input type="hidden" name="session_id" value="{session.session_id}">

                <div class="form-group">
                    <label for="temperature">Temperature</label>
                    <input type="range"
                           id="temperature"
                           name="temperature"
                           min="0"
                           max="2"
                           step="0.1"
                           value="{config.temperature}"
                           oninput="this.nextElementSibling.value = this.value">
                    <output>{config.temperature}</output>
                    <small>Higher = more creative, Lower = more focused</small>
                </div>

                <div class="form-group">
                    <label for="max_tokens">Max Tokens</label>
                    <input type="number"
                           id="max_tokens"
                           name="max_tokens"
                           min="1"
                           max="32000"
                           value="{config.max_tokens}">
                    <small>Maximum response length</small>
                </div>

                <div class="form-group">
                    <label>
                        <input type="checkbox"
                               name="stream"
                               {"checked" if config.stream else ""}>
                        Stream responses
                    </label>
                </div>

                <button type="submit" class="btn btn-primary">Save Settings</button>
            </form>

            <div id="settings-feedback"></div>

            <hr>

            <h4>Endpoint</h4>
            <div id="endpoints-list"
                 hx-get="/settings/endpoints?session_id={session.session_id}"
                 hx-trigger="load">
                Loading endpoints...
            </div>
        </div>
        ''')

    async def update_settings(self, request: Request) -> HTMLResponse:
        """Update runtime settings."""
        form = await request.form()
        session_id = str(form.get("session_id", ""))

        session = await self.session_manager.get_session(session_id)
        if not session:
            return HTMLResponse(
                '<div class="error">Session not found</div>',
                status_code=404,
            )

        # Update settings
        try:
            temperature = float(form.get("temperature", 0.7))
            max_tokens = int(form.get("max_tokens", 4096))
            stream = form.get("stream") == "on"

            session.runtime_config.temperature = max(0.0, min(2.0, temperature))
            session.runtime_config.max_tokens = max(1, min(32000, max_tokens))
            session.runtime_config.stream = stream

            return HTMLResponse('''
            <div class="success">
                Settings saved successfully
            </div>
            ''')

        except (ValueError, TypeError) as e:
            return HTMLResponse(
                f'<div class="error">Invalid value: {html.escape(str(e))}</div>',
                status_code=400,
            )

    async def list_endpoints(self, request: Request) -> HTMLResponse:
        """List available endpoints for switching."""
        session_id = request.query_params.get("session_id", "")
        session = await self.session_manager.get_session(session_id)

        if not session:
            return HTMLResponse('<div class="error">Session not found</div>')

        endpoints = session.toml_config.endpoints
        current = session.client.config.name

        html_parts = ['<div class="endpoint-list">']

        for name, endpoint in endpoints.items():
            is_current = name.lower() == current.lower()
            current_class = "endpoint-item current" if is_current else "endpoint-item"
            badge = '<span class="badge">active</span>' if is_current else ""

            html_parts.append(f'''
            <div class="{current_class}"
                 hx-post="/settings/switch-endpoint"
                 hx-vals='{{"endpoint": "{html.escape(name)}", "session_id": "{session.session_id}"}}'
                 hx-target="#endpoints-list"
                 hx-swap="innerHTML">
                <div class="endpoint-name">{html.escape(name)} {badge}</div>
                <div class="endpoint-details">
                    <small>{html.escape(endpoint.protocol)} | {html.escape(endpoint.default_model or "no default")}</small>
                </div>
            </div>
            ''')

        html_parts.append("</div>")
        return HTMLResponse("".join(html_parts))

    async def switch_endpoint(self, request: Request) -> HTMLResponse:
        """Switch to a different endpoint."""
        form = await request.form()
        endpoint_name = str(form.get("endpoint", ""))
        session_id = str(form.get("session_id", ""))

        session = await self.session_manager.get_session(session_id)
        if not session:
            return HTMLResponse(
                '<div class="error">Session not found</div>',
                status_code=404,
            )

        try:
            session.client.switch_endpoint(endpoint_name)
            # Re-render the endpoints list
            return await self.list_endpoints(request)
        except Exception as e:
            return HTMLResponse(
                f'<div class="error">Failed to switch: {html.escape(str(e))}</div>',
                status_code=400,
            )
