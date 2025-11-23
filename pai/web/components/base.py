"""Base component class for HTMX components.

Provides common functionality for all UI components:
- HTML rendering with Jinja2
- Route registration
- Session access
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from starlette.routing import Route

    from ..session import SessionManager

# Template directory
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


class Component(ABC):
    """Base class for self-managing HTMX components.

    Each component:
    - Manages its own routes
    - Renders its own templates
    - Has single responsibility
    """

    def __init__(self, session_manager: "SessionManager"):
        self.session_manager = session_manager
        self._jinja = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "xml"]),
            enable_async=True,
        )

    def render(self, template_name: str, **context) -> str:
        """Render a template with the given context."""
        template = self._jinja.get_template(template_name)
        return template.render(**context)

    async def render_async(self, template_name: str, **context) -> str:
        """Render a template asynchronously."""
        template = self._jinja.get_template(template_name)
        return await template.render_async(**context)

    @abstractmethod
    def get_routes(self) -> list["Route"]:
        """Return the routes this component handles."""
        ...

    @property
    @abstractmethod
    def component_id(self) -> str:
        """Unique identifier for this component (used in HTML IDs)."""
        ...
