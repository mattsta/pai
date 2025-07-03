"""
Utilities for session logging, statistics printing, and persistence.
"""

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from .models import ArenaState, Conversation, SessionStats


def print_stats(stats: "SessionStats", printer: Callable = print):
    """Prints a formatted summary of session statistics."""
    stat_dict = stats.get_stats()
    printer("\n📊 SESSION STATISTICS\n" + "=" * 50)
    last_req_stats = stat_dict.pop("last_request", None)
    for key, value in stat_dict.items():
        printer(f"{key.replace('_', ' ').title():<22}{value}")

    if last_req_stats:
        printer("-" * 50)
        printer("Last Request:")
        jitter_stats = last_req_stats.pop("jitter_stats", None)
        for key, value in last_req_stats.items():
            printer(f"  - {key.replace('_', ' ').title():<20}{value}")

        if jitter_stats:
            printer("\n  Stream Jitter Stats:")
            aborted = jitter_stats.get("aborted", False)
            if aborted:
                printer(f"    - {'Smoothing Aborted':<18}Yes")
            arrivals = jitter_stats.get("arrivals", "N/A")
            gaps = jitter_stats.get("gaps", 0)
            bursts = jitter_stats.get("bursts", 0)
            printer(f"    - {'Arrivals':<18}{arrivals}")
            printer(f"    - {'Gaps/Bursts':<18}{gaps}/{bursts}")
            printer(
                f"    - {'Min Delta (ms)':<18}{jitter_stats.get('min_delta', 'N/A')}"
            )
            printer(
                f"    - {'Mean Delta (ms)':<18}{jitter_stats.get('mean_delta', 'N/A')}"
            )
            printer(
                f"    - {'Std Dev Delta (ms)':<18}{jitter_stats.get('stdev_delta', 'N/A')}"
            )
            printer(
                f"    - {'Max Delta (ms)':<18}{jitter_stats.get('max_delta', 'N/A')}"
            )

    printer("=" * 50)


def closing(stats: "SessionStats", printer: Callable = print):
    """Prints the final statistics at the end of a session."""
    printer("\n\n📊 Final Statistics:")
    print_stats(stats, printer=printer)
    printer("\n👋 Session ended.")


def save_conversation_formats(
    conversation: "Conversation",
    log_dir: pathlib.Path,
    printer: Callable = print,
    arena_state: Optional["ArenaState"] = None,
):
    """Serializes a conversation to multiple HTML formats using Jinja2 templates."""
    try:
        # Assumes this file is in `pai/` and templates are in `pai/templates/`
        script_dir = pathlib.Path(__file__).parent
        template_dir = script_dir / "templates"

        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )
        env.filters["prettyjson"] = lambda v: json.dumps(v, indent=2)
        # Use the new method to get a richer history for logging.
        history = conversation.get_rich_history_for_template()

        # If in an arena, prepend the setup details to the history for logging.
        if arena_state:
            # Create a dict from the arena config for logging
            arena_config_dict = arena_state.arena_config.to_log_dict()
            arena_setup_yaml = yaml.dump(
                arena_config_dict, sort_keys=False, default_flow_style=False
            )
            history.insert(
                0,
                {
                    "role": "system",
                    "content": f"Arena Setup:\n\n---\n{arena_setup_yaml}---",
                },
            )
            if arena_state.initial_prompt:
                history.insert(
                    1, {"role": "user", "content": arena_state.initial_prompt}
                )

        # Defines the templates to render and their output filenames.
        formats = {
            "conversation.html": "conversation.html",
            "gptwink_format.html": "gptwink_conversation.html",
        }
        template_context = {
            "conversation_id": conversation.conversation_id,
            "history": history,
        }

        for template_name, output_filename in formats.items():
            try:
                template = env.get_template(template_name)
                final_html = template.render(**template_context)
                output_path = log_dir / output_filename
                output_path.write_text(final_html, encoding="utf-8")
            except Exception as e:
                printer(
                    f"\n⚠️ Warning: Could not render template '{template_name}': {e}"
                )
                # Don't fallback here, just try the next template

    except Exception as e:
        printer(f"\n⚠️ Warning: Could not initialize template environment: {e}")
        # As a global fallback, just write the raw turn data as JSON.
        try:
            all_turns = [turn.to_dict() for turn in conversation.turns]
            fallback_path = log_dir / "conversation_fallback.json"
            fallback_path.write_text(json.dumps(all_turns, indent=2), encoding="utf-8")
            printer(f"  -> Fallback data saved to {fallback_path}")
        except Exception as fallback_e:
            printer(f"  -> Could not even save fallback JSON: {fallback_e}")
