# This file serves as a registry for all protocol adapters.

import importlib.metadata
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base_adapter import BaseProtocolAdapter

# This map is now populated dynamically at runtime.
ADAPTER_MAP: dict[str, "BaseProtocolAdapter"] = {}


def load_protocol_adapters(printer: Callable[..., Any] = print):
    """
    Discovers and loads protocol adapters using package metadata entry points.
    """
    group = "polyglot_ai.protocols"
    printer(f"🔎 Loading protocol adapters from entry point group '{group}'...")
    found_adapters = 0
    try:
        entry_points = importlib.metadata.entry_points(group=group)
        for ep in entry_points:
            try:
                adapter_class = ep.load()
                # The name is what's used in the config file
                adapter_name = ep.name
                ADAPTER_MAP[adapter_name] = adapter_class()
                printer(f"  ✅ Loaded adapter '{adapter_name}' from '{ep.module}'")
                found_adapters += 1
            except Exception as e:
                printer(
                    f"  ❌ Failed to load adapter '{ep.name}' from '{ep.module}': {e}"
                )
    except Exception as e:
        printer(f"  ⚠️ Could not load entry points for '{group}': {e}")

    if not found_adapters:
        printer("  (No external protocol adapters found)")
