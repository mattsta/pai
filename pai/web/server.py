"""Main web server for pai using Starlette and HTMX.

Single responsibility: HTTP server setup and routing coordination.
Self-managing: Handles lifecycle of session manager and components.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..models import PolyglotConfig
from ..pricing import PricingService
from .components import (
    ChatComponent,
    ModelsComponent,
    SettingsComponent,
    StatsComponent,
)
from .session import SessionManager

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


class WebServer:
    """Main web server coordinating all components.

    Follows single-point-of-responsibility:
    - Initializes and manages component lifecycle
    - Coordinates routing across components
    - Provides the index page
    """

    def __init__(
        self,
        toml_config: PolyglotConfig,
        pricing_service: PricingService,
        host: str = "127.0.0.1",
        port: int = 8080,
    ):
        self.toml_config = toml_config
        self.pricing_service = pricing_service
        self.host = host
        self.port = port

        self.session_manager = SessionManager(toml_config, pricing_service)

        # Initialize components
        self.chat = ChatComponent(self.session_manager)
        self.models = ModelsComponent(self.session_manager)
        self.stats = StatsComponent(self.session_manager)
        self.settings = SettingsComponent(self.session_manager)

        # Jinja environment for main templates
        self._jinja = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "xml"]),
            enable_async=True,
        )

        self.app = self._create_app()

    def _create_app(self) -> Starlette:
        """Create the Starlette application with all routes."""

        @asynccontextmanager
        async def lifespan(app: Starlette):
            # Startup
            await self.session_manager.start()
            yield
            # Shutdown
            await self.session_manager.stop()

        # Collect routes from all components
        routes: list[Route | Mount] = [
            Route("/", self.index, methods=["GET"]),
            Route("/health", self.health, methods=["GET"]),
            Route("/session/new", self.new_session, methods=["POST"]),
        ]

        # Add component routes
        routes.extend(self.chat.get_routes())
        routes.extend(self.models.get_routes())
        routes.extend(self.stats.get_routes())
        routes.extend(self.settings.get_routes())

        # Static files (if directory exists)
        if STATIC_DIR.exists():
            routes.append(
                Mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
            )

        return Starlette(
            debug=False,
            routes=routes,
            lifespan=lifespan,
        )

    async def index(self, request: Request) -> HTMLResponse:
        """Render the main application page."""
        # Get or create session from cookie
        session_id = request.cookies.get("pai_session")
        session = await self.session_manager.get_or_create_session(session_id)

        template = self._jinja.get_template("index.html")
        html = await template.render_async(
            session_id=session.session_id,
            current_model=session.client.config.model_name,
            current_endpoint=session.client.config.name,
        )

        response = HTMLResponse(html)
        # Set session cookie
        response.set_cookie(
            "pai_session",
            session.session_id,
            max_age=3600,  # 1 hour
            httponly=True,
            samesite="lax",
        )
        return response

    async def health(self, request: Request) -> Response:
        """Health check endpoint."""
        return Response(
            content='{"status": "ok"}',
            media_type="application/json",
        )

    async def new_session(self, request: Request) -> HTMLResponse:
        """Create a new session (clears current)."""
        session = await self.session_manager.create_session()

        # Return redirect script to reload with new session
        response = HTMLResponse(f"""
        <script>
            document.cookie = "pai_session={session.session_id}; path=/; max-age=3600; samesite=lax";
            window.location.reload();
        </script>
        """)
        response.set_cookie(
            "pai_session",
            session.session_id,
            max_age=3600,
            httponly=True,
            samesite="lax",
        )
        return response


def create_app(
    toml_config: PolyglotConfig,
    pricing_service: PricingService,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> Starlette:
    """Create the web application.

    Factory function for creating the Starlette app with all components.
    """
    server = WebServer(toml_config, pricing_service, host, port)
    return server.app


async def run_server(
    toml_config: PolyglotConfig,
    pricing_service: PricingService,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run the web server.

    Main entry point for starting the web interface.
    """
    import uvicorn

    server = WebServer(toml_config, pricing_service, host, port)

    config = uvicorn.Config(
        server.app,
        host=host,
        port=port,
        log_level="info",
    )
    server_instance = uvicorn.Server(config)

    print(f"\n{'=' * 50}")
    print("  pai web interface")
    print(f"  Running at: http://{host}:{port}")
    print(f"{'=' * 50}\n")

    await server_instance.serve()
