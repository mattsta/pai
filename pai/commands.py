"""
Command parsing and execution for the interactive UI. This module uses a
class-based approach where each command is a self-contained object.
"""

import asyncio
import inspect
import json
import shlex
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import yaml
from jinja2 import Environment
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .models import (
    Arena,
    ArenaConfigFile,
    ArenaConfigJudge,
    ArenaConfigParticipant,
    ArenaConversationStyle,
    ArenaParticipant,
    ArenaState,
    ArenaTurnOrder,
    Conversation,
    TomlJudge,
    TomlParticipant,
    UIMode,
)
from .tools import get_tool_schemas

if TYPE_CHECKING:
    from prompt_toolkit.application import Application

    from .pai import InteractiveUI


# --- Base Command Class ---
class Command(ABC):
    """Abstract base class for all slash commands."""

    def __init__(self, ui: "InteractiveUI"):
        self.ui = ui

    @property
    @abstractmethod
    def name(self) -> str:
        """The primary name of the command (e.g., 'switch')."""
        pass

    @property
    def aliases(self) -> list[str]:
        """A list of alternative names for the command."""
        return []

    @property
    def requires_param(self) -> bool:
        """Whether the command requires a parameter to function."""
        return False

    @property
    def help_text(self) -> str:
        """The detailed help text for the command, shown for incomplete commands."""
        return f"Usage: /{self.name} <parameter>"

    @abstractmethod
    def execute(self, app: "Application", param: str | None = None):
        """The logic to execute when the command is called."""
        pass


# --- Command Implementations ---
class QuitCommand(Command):
    @property
    def name(self):
        return "quit"

    @property
    def aliases(self):
        return ["exit", "q"]

    def execute(self, app: "Application", param: str | None = None):
        app.exit()


class StatsCommand(Command):
    @property
    def name(self):
        return "stats"

    def execute(self, app: "Application", param: str | None = None):
        from .log_utils import print_stats

        print_stats(self.ui.client.stats, printer=self.ui.pt_printer)


class HelpCommand(Command):
    @property
    def name(self):
        return "help"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.pt_printer(
            """
--- GENERAL COMMANDS ---
  /help                  - Show this help message
  /stats                 - Show session statistics
  /quit, /exit, /q       - Exit the program

--- PROVIDER & MODEL ---
  /endpoints             - List available provider endpoints
  /switch <name>         - Switch to a different provider endpoint
  /model <name>          - Set the model for the current session
  /models [term] [refresh] - List & filter models (add 'refresh' to bypass cache)
  /info [model_id]       - Show detailed info for a model (defaults to current)
  /temp <value>          - Set the generation temperature (0.0-2.0)
  /tokens <num>          - Set the max tokens for the response
  /timeout <seconds>     - Set the network request timeout

--- AGENT & TOOLS ---
  /agent                 - Enable native agent mode (requires --tools)
  /legacy_agent          - Enable agent mode for models without tool-calling
  /template              - Enable chat template completion for debugging
  /tools                 - Toggle tool-use capability on/off for the session
  /confirm on|off        - Toggle user confirmation for tool execution

--- CHAT & HISTORY ---
  /mode                  - Toggle between chat and completion modes
  /system <text>         - Replace the system prompt stack with new text
  /system add <text>     - Add a prompt to the system stack
  /system pop            - Remove the last prompt from the system stack
  /system show           - Show the current system prompt stack
  /system clear          - Clear all system prompts
  /prompts               - List available prompts from the 'prompts/' directory
  /prompt <name>         - Add a prompt from file to the system stack
  /clear                 - Clear the current conversation history
  /history               - Show the raw message history for the conversation
  /save <name>           - Save the chat session to a file
  /load <name>           - Load a chat session from a file

--- MULTI-MODEL ARENA ---
  /arena <name> [turns]  - Start a pre-configured multi-model arena from pai.toml
  /arena new <name>      - Start building a new arena configuration interactively
  /arena list              - List all saved arena configurations
  /arena load <name>       - Load an arena configuration from a file
  /arena save <name>       - Save the current interactive arena config to a file
  /arena run [prompt]      - Run the interactively built or loaded arena
  /arena show              - Show the current arena configuration being built
  /arena reset             - Discard the current arena configuration
  /arena participant add <id> --name "Name" --endpoint <ep> --model <m>
                         - Add a participant to the arena being built
  /arena participant prompt <id> [<prompt> | file:<path>]
                         - Set a participant's system prompt (from text or file)
  /arena set initiator <id> - Set the participant who speaks first
  /arena set turns <number> - Set the max number of turns for the debate
  /arena set order <seq..|rand..> - Set turn order (sequential, random)
  /arena set style <pair..|chat..> - Set conversation style (pairwise, chatroom)
  /arena set wildcards on|off - Toggle random wildcard events per turn
  /arena set judge <id> --name "Name" --endpoint <ep> --model <m> --prompt <p>
                         - Define the judge participant
  /pause                   - Pause the arena after the current model's turn
  /resume                  - Resume a paused arena
  /say <message>           - Interject with a message while arena is paused

--- UI & DEBUGGING ---
  /multiline             - Toggle multi-line input mode (use Esc+Enter)
  /stream                - Toggle response streaming on/off
  /rich                  - Toggle rich Markdown rendering for output
  /smooth                - Toggle adaptive smooth streaming
  /verbose               - Toggle verbose logging of request parameters
  /debug                 - Toggle raw protocol-level debugging
"""
        )


class EndpointsCommand(Command):
    @property
    def name(self):
        return "endpoints"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.pt_printer("Available Endpoints:")
        for ep in self.ui.client.toml_config.endpoints:
            self.ui.pt_printer(f" - {ep.name}")


class SwitchCommand(Command):
    @property
    def name(self):
        return "switch"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        if not param:
            self.ui.pt_printer(self.help_text)
            return
        self.ui.client.switch_endpoint(param.strip())


class ModelsCommand(Command):
    @property
    def name(self):
        return "models"

    def execute(self, app: "Application", param: str | None = None):
        """Fetches and displays available models for the current endpoint."""
        force_refresh = False
        search_term: str | None = None
        if param:
            parts = param.strip().split()
            if "refresh" in [p.lower() for p in parts]:
                force_refresh = True
                parts = [p for p in parts if p.lower() != "refresh"]
            if parts:
                search_term = parts[0]

        async def _fetch_and_print_models():
            """The async part of the command's execution."""
            action_desc = "Loading"
            source_desc = "from cache or API"
            if force_refresh:
                action_desc = "Fetching"
                source_desc = "from API"

            search_desc = f" matching '{search_term}'" if search_term else ""
            self.ui.pt_printer(
                f"⏳ {action_desc} models{search_desc} for '{self.ui.client.config.name}' {source_desc}..."
            )

            models = await self.ui.client.list_models(
                force_refresh=force_refresh, search_term=search_term
            )
            if models:
                self.ui.pt_printer("\nAvailable Models:")
                for m in models:
                    self.ui.pt_printer(f"  - {m}")
            else:
                if search_term:
                    self.ui.pt_printer(
                        f"\nNo available models found matching '{search_term}'."
                    )
                else:
                    self.ui.pt_printer("\nNo available models found.")

        # Create a task to run the async code without blocking the UI's event loop.
        asyncio.create_task(_fetch_and_print_models())


class InfoCommand(Command):
    @property
    def name(self):
        return "info"

    @property
    def requires_param(self):
        return False

    def execute(self, app: "Application", param: str | None = None):
        """Fetches and displays detailed model information from provider and Hugging Face."""
        if param:
            model_id = param.strip()
        else:
            model_id = self.ui.client.config.model_name

        async def _fetch_and_print_info():
            self.ui.pt_printer(
                f"⏳ Fetching info for '{model_id}' from provider cache and Hugging Face API..."
            )
            provider_info = await self.ui.client.get_cached_provider_model_info(
                model_id
            )
            hf_info = await self.ui.client.get_model_info(model_id)

            info = {}
            if provider_info:
                info.update(provider_info)
            if hf_info:
                info.update(hf_info)

            if not info:
                self.ui.pt_printer(
                    f"❌ No information found for model '{model_id}' from any source."
                )
                return

            def format_bytes(byte_count: int | None) -> str:
                if byte_count is None:
                    return "N/A"
                if byte_count == 0:
                    return "0 B"
                if byte_count < 1024:
                    return f"{byte_count} B"
                if byte_count < 1024**2:
                    return f"{byte_count / 1024:.2f} KB"
                if byte_count < 1024**3:
                    return f"{byte_count / 1024**2:.2f} MB"
                return f"{byte_count / 1024**3:.2f} GB"

            def format_large_number(n: int | None) -> str:
                if n is None:
                    return "N/A"
                if n < 1_000_000:
                    return f"{n:,}"
                if n < 1_000_000_000:
                    return f"{n / 1_000_000:.2f}M"
                return f"{n / 1_000_000_000:.2f}B"

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold cyan", no_wrap=True)
            table.add_column()

            table.add_row("ID", info.get("id"))
            table.add_row("Type", info.get("pipeline_tag"))
            table.add_row("Library", info.get("library_name"))
            table.add_row("Downloads", f"{info.get('downloads', 0):,}")
            table.add_row("Likes", f"{info.get('likes', 0):,}")

            if inference_state := info.get("inference"):
                table.add_row("Inference Status", str(inference_state))

            if used_storage := info.get("usedStorage"):
                table.add_row("Storage Used", format_bytes(used_storage))

            if st_info := info.get("safetensors"):
                if params := st_info.get("parameters"):
                    param_str = ", ".join(
                        f"{format_large_number(v)} ({k})" for k, v in params.items()
                    )
                    table.add_row("Parameters", param_str)

                    total_params = sum(params.values())
                    if total_params > 0:
                        # 16-bit (2 bytes), 8-bit (1 byte), 4-bit (0.5 bytes)
                        size_16bit = total_params * 2
                        size_8bit = total_params * 1
                        size_4bit = int(total_params * 0.5)

                        size_str = (
                            f"{format_bytes(size_16bit)} (16-bit) │ "
                            f"{format_bytes(size_8bit)} (8-bit) │ "
                            f"{format_bytes(size_4bit)} (4-bit)"
                        )
                        table.add_row("Est. Memory", size_str)

            def format_relative_time(dt: datetime) -> str:
                now = datetime.now(dt.tzinfo)
                if dt > now:
                    return "from the future"

                # Calculate differences for date parts, then borrow if negative
                years = now.year - dt.year
                months = now.month - dt.month
                days = now.day - dt.day
                hours = now.hour - dt.hour
                minutes = now.minute - dt.minute
                seconds = now.second - dt.second

                if seconds < 0:
                    seconds += 60
                    minutes -= 1
                if minutes < 0:
                    minutes += 60
                    hours -= 1
                if hours < 0:
                    hours += 24
                    days -= 1
                if days < 0:
                    # Get days in previous month relative to 'now'
                    days_in_prev_month = (now.replace(day=1) - timedelta(days=1)).day
                    days += days_in_prev_month
                    months -= 1
                if months < 0:
                    months += 12
                    years -= 1

                parts = []
                if years > 0:
                    parts.append(f"{years} year{'s' if years > 1 else ''}")
                if months > 0:
                    parts.append(f"{months} month{'s' if months > 1 else ''}")
                if days > 0:
                    parts.append(f"{days} day{'s' if days > 1 else ''}")
                if hours > 0:
                    parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
                if minutes > 0:
                    parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
                if seconds > 0:
                    parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

                if not parts:
                    # This case handles sub-second differences.
                    return "just now"

                return ", ".join(parts) + " ago"

            created_at_str = info.get("createdAt")
            if created_at_str:
                try:
                    dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    relative_str = format_relative_time(dt)
                    dt_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                    table.add_row("Created At", f"{dt_str} ({relative_str})")
                except (ValueError, TypeError):
                    table.add_row("Created At", created_at_str)

            if last_mod := info.get("lastModified"):
                # Avoid duplicating info if createdAt and lastModified are identical
                if not created_at_str or created_at_str != last_mod:
                    try:
                        dt = datetime.fromisoformat(last_mod.replace("Z", "+00:00"))
                        relative_str = format_relative_time(dt)
                        dt_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                        table.add_row("Last Modified", f"{dt_str} ({relative_str})")
                    except (ValueError, TypeError):
                        table.add_row("Last Modified", last_mod)

            if tags := info.get("tags"):
                table.add_row("Tags", Text(", ".join(tags), overflow="fold"))

            console = self.ui.client.display.rich_console
            panel = Panel(
                table,
                title=f"Model Info: {info.get('id', model_id)}",
                title_align="left",
                border_style="dim",
            )
            # Must capture rich output to print via prompt_toolkit's safe printer
            with console.capture() as capture:
                console.print(panel)

            from prompt_toolkit.formatted_text import ANSI

            self.ui.pt_printer(ANSI(capture.get()))

        asyncio.create_task(_fetch_and_print_info())


class ModelCommand(Command):
    @property
    def name(self):
        return "model"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        if not param:
            self.ui.pt_printer(self.help_text)
            return
        self.ui.client.set_model(param.strip())


class TempCommand(Command):
    @property
    def name(self):
        return "temp"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        if not param:
            self.ui.pt_printer(self.help_text)
            return
        self.ui.set_temperature(param.strip())


class TokensCommand(Command):
    @property
    def name(self):
        return "tokens"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        if not param:
            self.ui.pt_printer(self.help_text)
            return
        self.ui.set_max_tokens(param.strip())


class TimeoutCommand(Command):
    @property
    def name(self):
        return "timeout"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        if not param:
            self.ui.pt_printer(self.help_text)
            return
        try:
            timeout_val = int(param.strip())
            self.ui.client.set_timeout(timeout_val)
        except (ValueError, TypeError):
            self.ui.pt_printer("❌ Invalid value. Please provide an integer.")


class ToggleStreamCommand(Command):
    @property
    def name(self):
        return "stream"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.toggle_stream()


class ToggleVerboseCommand(Command):
    @property
    def name(self):
        return "verbose"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.toggle_verbose()


class ToggleDebugCommand(Command):
    @property
    def name(self):
        return "debug"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.toggle_debug()


class ToggleRichTextCommand(Command):
    @property
    def name(self):
        return "rich"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.toggle_rich_text()


class ToggleSmoothStreamCommand(Command):
    @property
    def name(self):
        return "smooth"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.toggle_smooth_stream()


class ToggleConfirmCommand(Command):
    @property
    def name(self):
        return "confirm"

    def execute(self, app: "Application", param: str | None = None):
        if not param or param.strip().lower() not in ["on", "off"]:
            self.ui.pt_printer("❌ Usage: /confirm on|off")
            return
        self.ui.set_confirm_tool_use(param.strip().lower() == "on")


class ToggleToolsCommand(Command):
    @property
    def name(self):
        return "tools"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.toggle_tools()


class HistoryCommand(Command):
    @property
    def name(self):
        return "history"

    def execute(self, app: "Application", param: str | None = None):
        if self.ui.state.mode in [UIMode.ARENA, UIMode.ARENA_SETUP]:
            if not self.ui.state.arena:
                self.ui.pt_printer("❌ Arena state not found, but in arena mode.")
                return

            state = self.ui.state.arena
            console = self.ui.client.display.rich_console
            self.ui.pt_printer("--- Arena Participant Histories ---")

            for p_id, participant in state.arena_config.participants.items():
                title_style = "cyan"
                if p_id == "judge":
                    title_style = "yellow"

                history_json = json.dumps(
                    participant.conversation.get_history(), indent=2
                )
                history_syntax = Syntax(
                    history_json, "json", theme="monokai", word_wrap=True
                )
                panel = Panel(
                    history_syntax,
                    title=f"History for {participant.name} ({p_id})",
                    border_style=title_style,
                    title_align="left",
                )
                with console.capture() as capture:
                    console.print(panel)
                from prompt_toolkit.formatted_text import ANSI

                self.ui.pt_printer(ANSI(capture.get()))
        else:
            self.ui.pt_printer(json.dumps(self.ui.conversation.get_history(), indent=2))


class ClearCommand(Command):
    @property
    def name(self):
        return "clear"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.conversation.clear()
        self.ui.pt_printer("🧹 History cleared.")


class SystemCommand(Command):
    @property
    def name(self):
        return "system"

    @property
    def requires_param(self):
        return True

    @property
    def help_text(self) -> str:
        return """
⚙️ System Prompt Command Help
Usage: /system <subcommand> [text]

Subcommands:
  add <text>     - Add a new prompt to the system prompt stack.
  pop            - Remove the last prompt from the stack.
  show           - Show all prompts in the current stack.
  clear          - Clear all system prompts from the stack.
  <text>         - If no subcommand is given, the entire text replaces the
                   current system prompt stack and clears chat history.
"""

    def _execute_add(self, text: str | None):
        if not text:
            self.ui.pt_printer("❌ Usage: /system add <prompt text>")
            return
        self.ui.conversation.add_system_prompt(text)
        self.ui.pt_printer("🤖 System prompt added to stack.")

    def _execute_pop(self, text: str | None):
        popped = self.ui.conversation.pop_system_prompt()
        if popped:
            self.ui.pt_printer(
                f"🤖 Popped system prompt: '{popped[:60].strip()}...'"
            )
        else:
            self.ui.pt_printer("🤖 System prompt stack is empty.")

    def _execute_show(self, text: str | None):
        prompts = self.ui.conversation.get_system_prompts()
        if not prompts:
            self.ui.pt_printer("🤖 System prompt stack is empty.")
        else:
            self.ui.pt_printer("--- System Prompt Stack ---")
            for i, p in enumerate(prompts):
                self.ui.pt_printer(f"[{i + 1}]> {p}")
            self.ui.pt_printer("---------------------------")

    def _execute_clear(self, text: str | None):
        self.ui.conversation.clear_system_prompts()
        self.ui.pt_printer("🤖 All system prompts have been cleared.")

    def execute(self, app: "Application", param: str | None = None):
        if self.ui.state.mode == UIMode.COMPLETION:
            self.ui.pt_printer("❌ /system is only available in chat mode.")
            return

        if not param:
            self.ui.pt_printer(self.help_text)
            return

        parts = param.lstrip().split(" ", 1)
        subcommand_prefix = parts[0].lower()
        text = parts[1] if len(parts) > 1 else None

        subcommands = {
            "add": self._execute_add,
            "pop": self._execute_pop,
            "show": self._execute_show,
            "clear": self._execute_clear,
        }

        matching_keys = [
            key for key in subcommands if key.startswith(subcommand_prefix)
        ]

        if len(matching_keys) == 1:
            command_name = matching_keys[0]
            subcommands[command_name](text)
            return

        if len(matching_keys) > 1:
            self.ui.pt_printer(
                f"❌ Ambiguous subcommand. Possibilities: {', '.join(sorted(matching_keys))}"
            )
            return

        # If no subcommand matches, the entire param is the prompt.
        self.ui.enter_mode(UIMode.CHAT, clear_history=True)
        self.ui.conversation.set_system_prompt(param)
        self.ui.pt_printer(
            "🤖 System prompt stack replaced. Conversation history cleared."
        )


class PromptsCommand(Command):
    @property
    def name(self):
        return "prompts"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.pt_printer("Available prompts:")
        found = False
        for p in sorted(self.ui.prompts_dir.glob("*.md")):
            self.ui.pt_printer(f"  - {p.stem}")
            found = True
        for p in sorted(self.ui.prompts_dir.glob("*.txt")):
            self.ui.pt_printer(f"  - {p.stem}")
            found = True
        if not found:
            self.ui.pt_printer(
                "  (None found. Add .md or .txt files to 'prompts/' directory)"
            )


class PromptCommand(Command):
    @property
    def name(self):
        return "prompt"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        if not param:
            self.ui.pt_printer(f"Usage: /{self.name} <name>")
            return
        if self.ui.state.mode == UIMode.COMPLETION:
            self.ui.pt_printer("❌ /prompt is only available in chat mode.")
            return

        clean_param = param.strip()
        prompt_path = self.ui.prompts_dir / f"{clean_param}.md"
        if not prompt_path.exists():
            prompt_path = self.ui.prompts_dir / f"{clean_param}.txt"

        if prompt_path.exists() and prompt_path.is_file():
            content = prompt_path.read_text(encoding="utf-8")
            # Loading a prompt now *adds* it to the stack instead of replacing.
            self.ui.conversation.add_system_prompt(content)
            self.ui.pt_printer(f"🤖 Added system prompt from '{clean_param}' to stack.")
        else:
            self.ui.pt_printer(
                f"❌ Prompt '{clean_param}' not found in '{self.ui.prompts_dir}'."
            )


class AgentCommand(Command):
    @property
    def name(self):
        return "agent"

    def execute(self, app: "Application", param: str | None = None):
        """Loads the special 'code_editor' prompt to start an agent session."""
        if self.ui.state.mode == UIMode.COMPLETION:
            self.ui.pt_printer("❌ /agent is only available in chat mode.")
            return

        if not self.ui.runtime_config.tools:
            self.ui.pt_printer(
                "⚠️  Warning: Native agent mode works best with the --tools flag enabled at startup."
            )

        self.ui.enter_mode(UIMode.NATIVE_AGENT)
        self.ui.client.tools_enabled = True

        param = "code_editor"
        prompt_path = self.ui.prompts_dir / f"{param}.md"

        if prompt_path.exists() and prompt_path.is_file():
            content = prompt_path.read_text(encoding="utf-8")
            self.ui.conversation.set_system_prompt(content)
            self.ui.pt_printer(
                f"🤖 Native Agent mode enabled (loaded '{param}' prompt)."
            )
        else:
            self.ui.pt_printer(
                f"❌ Agent prompt '{param}.md' not found in '{self.ui.prompts_dir}'."
            )


class LegacyAgentCommand(Command):
    @property
    def name(self):
        return "legacy_agent"

    def execute(self, app: "Application", param: str | None = None):
        """Loads a prompt that teaches a generic model to use tools via text."""
        if self.ui.state.mode == UIMode.COMPLETION:
            self.ui.pt_printer("❌ /legacy_agent is only available in chat mode.")
            return

        if not get_tool_schemas():
            self.ui.pt_printer(
                "❌ No tools loaded. To use agent mode, please restart and add the `--tools` flag."
            )
            return

        self.ui.enter_mode(UIMode.LEGACY_AGENT)
        self.ui.client.tools_enabled = False  # Disable native tools

        param = "legacy_agent"
        prompt_path = self.ui.prompts_dir / f"{param}.md"

        if prompt_path.exists() and prompt_path.is_file():
            from .tools import get_tool_manifest

            template = prompt_path.read_text(encoding="utf-8")
            manifest = get_tool_manifest()
            content = template.replace("{{ tool_manifest }}", manifest)

            self.ui.conversation.set_system_prompt(content)
            self.ui.pt_printer(
                "🤖 Legacy Agent mode enabled. Tools will be used via text prompt."
            )
        else:
            self.ui.pt_printer(
                f"❌ Legacy agent prompt '{param}.md' not found in '{self.ui.prompts_dir}'."
            )


class TemplateCommand(Command):
    @property
    def name(self):
        return "template"

    def execute(self, app: "Application", param: str | None = None):
        """Switches to template completion mode if the model has a chat template."""
        if self.ui.state.mode == UIMode.COMPLETION:
            self.ui.pt_printer(
                "❌ /template requires a chat-like history. Use /mode to switch."
            )
            return

        async def _load_and_set_template():
            model_id = self.ui.client.config.model_name
            self.ui.pt_printer(
                f"⏳ Fetching model info for '{model_id}' to find chat template..."
            )

            info = await self.ui.client.get_model_info(model_id)

            if not info:
                self.ui.pt_printer(
                    f"❌ Could not retrieve info for model '{model_id}'."
                )
                return

            # Look for the chat template in the correct nested location.
            template_str = None
            if isinstance(config := info.get("config"), dict):
                if isinstance(tokenizer_config := config.get("tokenizer_config"), dict):
                    template_str = tokenizer_config.get("chat_template")

            if not template_str:
                checked_paths = [
                    "'config.tokenizer_config.chat_template'",
                ]
                self.ui.pt_printer(
                    f"❌ No chat_template found in metadata for model '{model_id}'."
                )
                self.ui.pt_printer(f"   Checked path: {', '.join(checked_paths)}.")
                self.ui.pt_printer(
                    f"   Use `/info {model_id}` to inspect the full cached object."
                )
                if self.ui.runtime_config.verbose:
                    # Dump the full metadata structure if verbose mode is on.
                    self.ui.pt_printer("\n   --- Verbose: full metadata dump ---")
                    pretty_info = json.dumps(info, indent=2)
                    self.ui.pt_printer(pretty_info)
                    self.ui.pt_printer("   ------------------------------------")

                return

            try:
                # Use a specific environment for safety and consistency.
                env = Environment(
                    trim_blocks=True, lstrip_blocks=True, autoescape=False
                )
                template_obj = env.from_string(template_str)
                self.ui.state.chat_template = template_str
                self.ui.state.chat_template_obj = template_obj
                self.ui.enter_mode(UIMode.TEMPLATE_COMPLETION, clear_history=True)
                self.ui.pt_printer(
                    f"✅ Switched to Template Completion mode for '{model_id}'."
                )
                self.ui.pt_printer("ℹ️  Template:")
                from rich.markdown import Markdown
                from rich.panel import Panel

                console = self.ui.client.display.rich_console
                panel = Panel(
                    Markdown(f"```jinja\n{template_str}\n```"),
                    title="Jinja2 Chat Template",
                    title_align="left",
                    border_style="dim",
                )
                with console.capture() as capture:
                    console.print(panel)
                from prompt_toolkit.formatted_text import ANSI

                self.ui.pt_printer(ANSI(capture.get()))
            except Exception as e:
                self.ui.pt_printer(f"❌ Error compiling Jinja2 template: {e}")

        asyncio.create_task(_load_and_set_template())


class ArenaCommand(Command):
    @property
    def name(self):
        return "arena"

    @property
    def aliases(self) -> list[str]:
        return ["ar"]

    @property
    def requires_param(self):
        return True

    @property
    def help_text(self) -> str:
        return """
⚔️ Arena Command Help
Manages multi-model conversations. Can be run from a configuration in pai.toml
or built interactively.

--- Pre-configured (from pai.toml) ---
Usage: /arena <name> [turns]
  <name>         - The name of an arena defined in pai.toml.
  [turns]        - (Optional) Number of turns to run. Default: 10.

--- Interactive Builder ---
Usage: /arena <subcommand> [options...]
  new <name>       - Start building a new arena config named <name>.
  list             - List all saved arena configurations from the 'arenas/' dir.
  load <name>      - Load a saved arena configuration file.
  save <name>      - Save the current interactive configuration to a file.
  run [prompt]     - Run the currently loaded interactive arena.
  show             - Show the current interactive arena configuration.
  reset            - Discard the current arena configuration.
  participant ...  - Manage participants. Use '/arena p' for more help.
  set ...          - Configure arena settings. Use '/arena s' for more help.
"""

    def _parse_kv_args(self, args_list: list[str]) -> dict[str, str | bool]:
        """Parses a list of arguments into a dictionary of key-value pairs."""
        args = {}
        i = 0
        while i < len(args_list):
            arg = args_list[i]
            if arg.startswith("--"):
                key = arg[2:]
                # Check for next item being a value or another flag
                if i + 1 < len(args_list) and not args_list[i + 1].startswith("--"):
                    args[key] = args_list[i + 1]
                    i += 1  # Skip the value
                else:
                    args[key] = True  # It's a boolean flag
            i += 1
        return args

    def execute(self, app: "Application", param: str | None = None):
        if not param:
            self.ui.pt_printer(self.help_text)
            return
        if self.ui.state.mode == UIMode.COMPLETION:
            self.ui.pt_printer("❌ /arena is only available in chat mode.")
            return

        command_parts = param.strip().split(" ", 1)
        subcommand_prefix = command_parts[0]

        # The 'run' command takes a free-form string that should not be parsed by shlex.
        # We check for it first by its unambiguous prefix.
        if "run".startswith(subcommand_prefix):
            prompt = command_parts[1] if len(command_parts) > 1 else ""
            self._execute_run([prompt] if prompt else [])
            return

        # For all other commands, parse arguments using shlex for correctness.
        try:
            parts = shlex.split(param)
        except ValueError as e:
            if "No closing quotation" in str(e):
                self.ui.pt_printer("❌ Error: Unmatched quote in command arguments.")
            else:
                self.ui.pt_printer(f"❌ Error parsing arguments: {e}")
            return

        subcommand_prefix = parts[0]
        args = parts[1:]

        # --- Subcommand Dispatcher ---
        subcommands = {
            "new": self._execute_new,
            "n": self._execute_new,
            "list": self._execute_list,
            "ls": self._execute_list,
            "load": self._execute_load,
            "ld": self._execute_load,
            "save": self._execute_save,
            "sv": self._execute_save,
            "reset": self._execute_reset,
            "show": self._execute_show,
            "sh": self._execute_show,
            "participant": self._execute_participant,
            "p": self._execute_participant,
            "set": self._execute_set,
            "s": self._execute_set,
        }
        canonical_names = {
            self._execute_new: "new",
            self._execute_list: "list",
            self._execute_load: "load",
            self._execute_save: "save",
            self._execute_reset: "reset",
            self._execute_show: "show",
            self._execute_participant: "participant",
            self._execute_set: "set",
        }

        matching_keys = [
            key for key in subcommands if key.startswith(subcommand_prefix)
        ]

        if not matching_keys:
            # Fallback to legacy behavior: /arena <name> [turns]
            self._execute_from_config(parts)
            return

        unique_handlers = {subcommands[key] for key in matching_keys}
        if len(unique_handlers) > 1:
            possible_cmds = sorted([canonical_names[h] for h in unique_handlers])
            self.ui.pt_printer(
                f"❌ Ambiguous command. Possibilities: {', '.join(possible_cmds)}"
            )
            return

        handler = unique_handlers.pop()
        handler(args)

    def _execute_new(self, args: list[str]):
        if not args:
            self.ui.pt_printer("❌ Usage: /arena new <name>")
            return
        arena_name = args[0]
        arena_config = Arena(
            name=arena_name,
            participants={},
            initiator_id="",
            turn_order=ArenaTurnOrder.SEQUENTIAL,
            wildcards_enabled=False,
            conversation_style=ArenaConversationStyle.PAIRWISE,
        )
        arena_state = ArenaState(
            arena_config=arena_config, turn_order_ids=[], max_turns=10
        )
        self.ui.state.arena = arena_state
        self.ui.enter_mode(UIMode.ARENA_SETUP, clear_history=False)
        self.ui.pt_printer(
            f"🚧 New arena '{arena_name}' created. Add participants and set options."
        )

    def _execute_list(self):
        """Lists available, saved arena configuration files."""
        self.ui.pt_printer("💾 Available saved arenas:")
        found = False
        for p in sorted(self.ui.arenas_dir.glob("*.yaml")):
            self.ui.pt_printer(f"  - {p.stem}")
            found = True
        for p in sorted(self.ui.arenas_dir.glob("*.yml")):
            self.ui.pt_printer(f"  - {p.stem}")
            found = True
        if not found:
            self.ui.pt_printer(
                "  (None found. Use '/arena save <name>' to save a configuration.)"
            )

    def _execute_load(self, args: list[str]):
        """Loads an arena state from a YAML configuration file."""
        if not args:
            self.ui.pt_printer("❌ Usage: /arena load <filename>")
            return
        filename = args[0]
        if not filename.endswith((".yml", ".yaml")):
            filename += ".yaml"

        load_path = self.ui.arenas_dir / filename
        if not load_path.is_file():
            self.ui.pt_printer(f"❌ Arena configuration file not found: '{load_path}'")
            return

        try:
            with open(load_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            arena_file_data = ArenaConfigFile.model_validate(data)

            # --- Build ArenaState from loaded config ---
            participants = {}
            for p_id, p_config in arena_file_data.participants.items():
                system_prompt = self._get_system_prompt_from_config(p_config)
                conversation = Conversation()
                conversation.set_system_prompt(system_prompt)
                participants[p_id] = ArenaParticipant(
                    id=p_id,
                    name=p_config.name,
                    endpoint=p_config.endpoint,
                    model=p_config.model,
                    system_prompt=system_prompt,
                    conversation=conversation,
                )

            judge_participant = None
            if arena_file_data.judge:
                j_config = arena_file_data.judge
                system_prompt = self._get_system_prompt_from_config(j_config)
                judge_conversation = Conversation()
                judge_conversation.set_system_prompt(system_prompt)
                judge_participant = ArenaParticipant(
                    id="judge",
                    name=j_config.name,
                    endpoint=j_config.endpoint,
                    model=j_config.model,
                    system_prompt=system_prompt,
                    conversation=judge_conversation,
                )
                participants["judge"] = judge_participant

            arena_config_obj = Arena(
                name=arena_file_data.name,
                participants=participants,
                initiator_id=arena_file_data.initiator,
                judge=judge_participant,
                turn_order=arena_file_data.turn_order,
                wildcards_enabled=arena_file_data.wildcards_enabled,
                conversation_style=arena_file_data.conversation_style,
            )

            arena_state = ArenaState(
                arena_config=arena_config_obj,
                turn_order_ids=[],  # Will be set by /arena run
                max_turns=arena_file_data.max_turns,
            )

            self.ui.state.arena = arena_state
            self.ui.enter_mode(UIMode.ARENA_SETUP, clear_history=False)
            self.ui.pt_printer(
                f"✅ Loaded arena '{arena_file_data.name}' from '{load_path}'. Use `/arena run` to start."
            )

        except (yaml.YAMLError, Exception) as e:
            self.ui.pt_printer(f"❌ Error loading arena configuration: {e}")

    def _execute_save(self, args: list[str]):
        """Saves the current arena configuration to a YAML file."""
        if not self.ui.state.arena:
            self.ui.pt_printer("❌ No active arena configuration to save.")
            return
        if not args:
            self.ui.pt_printer("❌ Usage: /arena save <filename>")
            return
        filename = args[0]
        if not filename.endswith((".yml", ".yaml")):
            filename += ".yaml"

        state = self.ui.state.arena
        config = state.arena_config

        participants_dict = {}
        for p_id, p in config.participants.items():
            if p_id == "judge":
                continue
            participants_dict[p_id] = {
                "name": p.name,
                "endpoint": p.endpoint,
                "model": p.model,
                "system_prompt": p.system_prompt,
            }

        judge_dict = None
        if config.judge:
            judge_dict = {
                "name": config.judge.name,
                "endpoint": config.judge.endpoint,
                "model": config.judge.model,
                "system_prompt": config.judge.system_prompt,
            }

        arena_file_content = {
            "name": config.name,
            "initiator": config.initiator_id,
            "max_turns": state.max_turns,
            "turn_order": config.turn_order.value,
            "wildcards_enabled": config.wildcards_enabled,
            "conversation_style": config.conversation_style.value,
            "participants": participants_dict,
        }
        if judge_dict:
            arena_file_content["judge"] = judge_dict

        save_path = self.ui.arenas_dir / filename
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    arena_file_content, f, sort_keys=False, default_flow_style=False
                )
            self.ui.pt_printer(f"💾 Arena configuration saved to '{save_path}'.")
        except Exception as e:
            self.ui.pt_printer(f"❌ Error saving arena configuration: {e}")

    def _execute_reset(self):
        self.ui.state.arena = None
        self.ui.enter_mode(UIMode.CHAT, clear_history=False)
        self.ui.pt_printer("🗑️ Arena configuration reset.")

    def _execute_show(self):
        if not self.ui.state.arena:
            self.ui.pt_printer("❌ No active arena configuration to show.")
            return

        state = self.ui.state.arena
        config = state.arena_config
        console = self.ui.client.display.rich_console
        content = f"[bold]Arena:[/bold] {config.name}\n"
        content += f"[bold]Turns:[/bold] {state.max_turns}\n"
        content += f"[bold]Turn Order:[/bold] {config.turn_order.value}\n"
        content += f"[bold]Conversation Style:[/bold] {config.conversation_style.value}\n"
        content += f"[bold]Wildcards:[/bold] {'Enabled' if config.wildcards_enabled else 'Disabled'}\n"
        content += f"[bold]Initiator:[/bold] {config.initiator_id or '[Not Set]'}\n"
        content += f"\n[bold]Participants ({len(config.participants)}):[/bold]"

        self.ui.pt_printer(content)

        for p_id, p in config.participants.items():
            if p_id == "judge":
                continue
            p_content = f"  [bold cyan]ID:[/bold] {p.id}\n"
            p_content += f"  [bold]Name:[/bold] {p.name}\n"
            p_content += f"  [bold]Endpoint:[/bold] {p.endpoint}\n"
            p_content += f"  [bold]Model:[/bold] {p.model}\n"
            p_content += Syntax(
                p.system_prompt, "markdown", theme="monokai", word_wrap=True
            )
            panel = Panel(
                p_content,
                title=f"Participant: {p.name}",
                border_style="cyan",
                title_align="left",
            )
            with console.capture() as capture:
                console.print(panel)
            from prompt_toolkit.formatted_text import ANSI

            self.ui.pt_printer(ANSI(capture.get()))

        if config.judge:
            j = config.judge
            p_content = f"  [bold cyan]ID:[/bold] {j.id}\n"
            p_content += f"  [bold]Name:[/bold] {j.name}\n"
            p_content += f"  [bold]Endpoint:[/bold] {j.endpoint}\n"
            p_content += f"  [bold]Model:[/bold] {j.model}\n"
            p_content += Syntax(
                j.system_prompt, "markdown", theme="monokai", word_wrap=True
            )
            panel = Panel(
                p_content,
                title="Judge",
                border_style="yellow",
                title_align="left",
            )
            with console.capture() as capture:
                console.print(panel)
            from prompt_toolkit.formatted_text import ANSI

            self.ui.pt_printer(ANSI(capture.get()))

    def _get_participant_help(self) -> str:
        return """
👥 Arena Participant Command Help
Usage: /arena participant <add|prompt> [options...]

Subcommands:
  add <id> --name "Display Name" --endpoint <ep> --model <m>
    - Adds a new participant to the interactive arena.
    - <id>: A short, unique identifier (e.g., 'p1', 'critic').
    - --name: The full display name.
    - --endpoint: An endpoint name from pai.toml.
    - --model: The model to use.

  prompt <id> <text | file:path/to/prompt.md>
    - Sets the system prompt for a participant.
    - <id>: The ID of the participant to modify.
    - <text>: The raw text of the system prompt.
    - file:<path>: The path to a file in the 'prompts/' directory.
"""

    def _execute_participant_add(self, args: list[str]):
        if not args:
            self.ui.pt_printer("❌ Usage: /arena participant add <id> --name ...")
            return
        p_id = args[0]
        kv_args = self._parse_kv_args(args[1:])
        for key in ["name", "endpoint", "model"]:
            if key not in kv_args:
                self.ui.pt_printer(f"❌ Missing required argument: --{key}")
                return
        participant = ArenaParticipant(
            id=p_id,
            name=str(kv_args["name"]),
            endpoint=str(kv_args["endpoint"]),
            model=str(kv_args["model"]),
            system_prompt="You are a helpful assistant.",
            conversation=Conversation(),
        )
        self.ui.state.arena.arena_config.participants[p_id] = participant
        self.ui.pt_printer(f"✅ Added participant '{p_id}' to the arena.")

    def _execute_participant_prompt(self, args: list[str]):
        if len(args) < 2:
            self.ui.pt_printer(
                "❌ Usage: /arena participant prompt <id> <text|file:path>"
            )
            return
        p_id = args[0]
        prompt_str = " ".join(args[1:])
        participant = self.ui.state.arena.arena_config.participants.get(p_id)
        if not participant:
            self.ui.pt_printer(f"❌ Participant with ID '{p_id}' not found.")
            return

        if prompt_str.lower().startswith("file:"):
            path_str = prompt_str[5:]
            prompt_path = self.ui.prompts_dir / f"{path_str}.md"
            if not prompt_path.exists():
                prompt_path = self.ui.prompts_dir / f"{path_str}.txt"
            if not prompt_path.is_file():
                self.ui.pt_printer(f"❌ Prompt file not found: {prompt_path}")
                return
            prompt_content = prompt_path.read_text(encoding="utf-8")
            participant.system_prompt = prompt_content
            participant.conversation.set_system_prompt(prompt_content)
            self.ui.pt_printer(
                f"✅ Set system prompt for '{p_id}' from file '{path_str}'."
            )
        else:
            participant.system_prompt = prompt_str
            participant.conversation.set_system_prompt(prompt_str)
            self.ui.pt_printer(f"✅ Set system prompt for '{p_id}'.")

    def _execute_participant(self, args: list[str]):
        if self.ui.state.mode != UIMode.ARENA_SETUP:
            self.ui.pt_printer(
                "❌ Arena must be started with `/arena new <name>` first."
            )
            return
        if not args:
            self.ui.pt_printer(self._get_participant_help())
            return

        subcommand_prefix = args[0]
        p_args = args[1:]
        subcommands = {
            "add": self._execute_participant_add,
            "a": self._execute_participant_add,
            "prompt": self._execute_participant_prompt,
        }
        canonical_names = {
            self._execute_participant_add: "add",
            self._execute_participant_prompt: "prompt",
        }

        matching_keys = [
            key for key in subcommands if key.startswith(subcommand_prefix)
        ]
        if not matching_keys:
            self.ui.pt_printer(self._get_participant_help())
            return

        unique_handlers = {subcommands[key] for key in matching_keys}
        if len(unique_handlers) > 1:
            possible_cmds = sorted([canonical_names[h] for h in unique_handlers])
            self.ui.pt_printer(
                f"❌ Ambiguous participant command. Possibilities: {', '.join(possible_cmds)}"
            )
            return

        handler = unique_handlers.pop()
        handler(p_args)

    def _get_set_help(self) -> str:
        return """
🔧 Arena Set Command Help
Usage: /arena set <subcommand> [options...]

Subcommands:
  initiator <id>
    - Sets which participant speaks first.

  turns <number>
    - Sets the maximum number of turns for the arena.

  order <sequential|random>
    - Sets the turn order strategy.

  style <pairwise|chatroom>
    - Sets the conversation style.
    - pairwise: Each model sees a 1-on-1 chat history.
    - chatroom: All models see a shared, interleaved transcript.

  wildcards <on|off>
    - Toggles random events like temperature spikes.

  judge <id> --name "Name" --endpoint <ep> --model <m> --prompt <p>
    - Configures the judge participant for the arena.
"""

    def _execute_set(self, args: list[str]):
        if self.ui.state.mode != UIMode.ARENA_SETUP:
            self.ui.pt_printer(
                "❌ Arena must be started with `/arena new <name>` first."
            )
            return
        if not args:
            self.ui.pt_printer(self._get_set_help())
            return

        sub_sub_command = args[0]
        s_args = args[1:]

        if sub_sub_command in ["initiator", "i"]:
            if not s_args:
                self.ui.pt_printer("❌ Usage: /arena set initiator <participant_id>")
                return
            p_id = s_args[0]
            if p_id not in self.ui.state.arena.arena_config.participants:
                self.ui.pt_printer(f"❌ Participant '{p_id}' not found.")
                return
            self.ui.state.arena.arena_config.initiator_id = p_id
            self.ui.pt_printer(f"✅ Set arena initiator to '{p_id}'.")
        elif sub_sub_command in ["turns", "t"]:
            if not s_args:
                self.ui.pt_printer("❌ Usage: /arena set turns <number>")
                return
            try:
                self.ui.state.arena.max_turns = int(s_args[0])
                self.ui.pt_printer(
                    f"✅ Set max turns to {self.ui.state.arena.max_turns}."
                )
            except (ValueError, TypeError):
                self.ui.pt_printer("❌ Invalid value for turns. Must be an integer.")
        elif sub_sub_command in ["order", "o"]:
            if not s_args:
                self.ui.pt_printer("❌ Usage: /arena set order <sequential|random>")
                return
            strategy_str = s_args[0].lower()
            try:
                strategy_enum = ArenaTurnOrder(strategy_str)
                self.ui.state.arena.arena_config.turn_order = strategy_enum
                self.ui.pt_printer(f"✅ Set turn order strategy to '{strategy_str}'.")
            except ValueError:
                self.ui.pt_printer("❌ Invalid strategy. Use 'sequential' or 'random'.")
        elif sub_sub_command in ["style", "st"]:
            if not s_args:
                self.ui.pt_printer("❌ Usage: /arena set style <pairwise|chatroom>")
                return
            style_str = s_args[0].lower()
            try:
                style_enum = ArenaConversationStyle(style_str)
                self.ui.state.arena.arena_config.conversation_style = style_enum
                self.ui.pt_printer(f"✅ Set conversation style to '{style_str}'.")
            except ValueError:
                self.ui.pt_printer("❌ Invalid style. Use 'pairwise' or 'chatroom'.")
        elif sub_sub_command in ["wildcards", "w"]:
            if not s_args or s_args[0].lower() not in ["on", "off"]:
                self.ui.pt_printer("❌ Usage: /arena set wildcards <on|off>")
                return
            enabled = s_args[0].lower() == "on"
            self.ui.state.arena.arena_config.wildcards_enabled = enabled
            self.ui.pt_printer(f"✅ Wildcards {'enabled' if enabled else 'disabled'}.")
        elif sub_sub_command in ["judge", "j"]:
            if not s_args:
                self.ui.pt_printer("❌ Usage: /arena set judge <id> [options]")
                return
            p_id = s_args[0]
            kv_args = self._parse_kv_args(s_args[1:])
            for key in ["name", "endpoint", "model", "prompt"]:
                if key not in kv_args:
                    self.ui.pt_printer(f"❌ Missing required argument: --{key}")
                    return
            system_prompt = str(kv_args["prompt"])
            judge = ArenaParticipant(
                id=p_id,
                name=str(kv_args["name"]),
                endpoint=str(kv_args["endpoint"]),
                model=str(kv_args["model"]),
                system_prompt=system_prompt,
                conversation=Conversation(
                    _messages=[{"role": "system", "content": system_prompt}]
                ),
            )
            self.ui.state.arena.arena_config.judge = judge
            self.ui.state.arena.arena_config.participants[p_id] = (
                judge  # Also add to main dict
            )
            self.ui.pt_printer(f"✅ Set judge to '{p_id}'.")
        else:
            self.ui.pt_printer(
                "❌ Unknown set command. Use 'initiator', 'turns', or 'judge'."
            )

    def _get_system_prompt_from_config(
        self,
        p_config: TomlParticipant
        | TomlJudge
        | ArenaConfigParticipant
        | ArenaConfigJudge,
    ) -> str:
        """Determines the system prompt from an arena participant config object."""
        # Inline prompt takes precedence
        if p_config.system_prompt:
            return p_config.system_prompt
        # Fallback to a key for a file
        if p_config.system_prompt_key:
            prompt_key = p_config.system_prompt_key
            prompt_path = self.ui.prompts_dir / f"{prompt_key}.md"
            if not prompt_path.exists():
                prompt_path = self.ui.prompts_dir / f"{prompt_key}.txt"

            if not prompt_path.is_file():
                # Let the caller handle the error.
                raise FileNotFoundError(f"System prompt file '{prompt_key}' not found.")
            return prompt_path.read_text(encoding="utf-8")

        # Fallback to a generic prompt if neither is specified.
        return "You are a helpful assistant."

    def _execute_run(self, args: list[str]):
        if not self.ui.state.arena:
            self.ui.pt_printer(
                "❌ No active arena configuration. Use `/arena new` first."
            )
            return

        state = self.ui.state.arena
        # --- Validation ---
        p_ids = [p for p in state.arena_config.participants.keys() if p != "judge"]
        if len(p_ids) < 2:
            self.ui.pt_printer("❌ Arena must have at least 2 participants.")
            return
        if (
            state.arena_config.conversation_style == ArenaConversationStyle.PAIRWISE
            and len(p_ids) != 2
        ):
            self.ui.pt_printer(
                "❌ Pairwise mode requires exactly 2 participants (excluding the judge)."
            )
            return
        if not state.arena_config.initiator_id:
            self.ui.pt_printer("❌ Arena initiator has not been set.")
            return
        if state.arena_config.initiator_id not in p_ids:
            self.ui.pt_printer("❌ Initiator must be a valid participant ID.")
            return

        # --- Setup turn order ---
        initiator_idx = p_ids.index(state.arena_config.initiator_id)
        state.turn_order_ids = p_ids[initiator_idx:] + p_ids[:initiator_idx]

        # --- Start the orchestrator ---
        self.ui.enter_mode(UIMode.ARENA, clear_history=True)
        self.ui.state.arena = state
        self.ui.arena_paused_event.clear()

        initiator_name = state.arena_config.get_initiator().name
        self.ui.pt_printer(
            f"⚔️  Arena '{state.arena_config.name}' activated for {state.max_turns} turns."
        )
        self.ui.pt_printer(
            "   The arena will auto-run. Use /pause, /resume, and /say to control it."
        )

        initial_prompt = " ".join(args)
        orchestrator = self.ui._get_orchestrator()
        if initial_prompt:
            self.ui.pt_printer(f"▶️  Starting with prompt: {initial_prompt}")
            self.ui.generation_in_progress.set()
            self.ui.generation_task = asyncio.create_task(
                orchestrator.run(initial_prompt)
            )
        else:
            self.ui.pt_printer(
                f"   Your next prompt will be given to '{initiator_name}' to start the conversation."
            )

    def _execute_from_config(self, parts: list[str]):
        """Runs an arena from a static configuration in pai.toml."""
        arena_name, max_turns_str = (parts + [None])[:2]
        try:
            max_turns = int(max_turns_str) if max_turns_str else 10
        except (ValueError, TypeError):
            self.ui.pt_printer("❌ Invalid value for turns. Must be an integer.")
            return

        arenas_config = self.ui.client.toml_config.arenas
        arena_config = arenas_config.get(arena_name)
        if not arena_config:
            self.ui.pt_printer(f"❌ Arena '{arena_name}' not found in pai.toml.")
            return

        participant_configs = arena_config.participants
        judge_config = arena_config.judge
        if len(participant_configs) < 2:
            self.ui.pt_printer(
                f"❌ Arena '{arena_name}' must have at least 2 participants."
            )
            return

        try:
            participants = {}
            for p_id, p_config in participant_configs.items():
                system_prompt = self._get_system_prompt_from_config(p_config)
                conversation = Conversation()
                conversation.set_system_prompt(system_prompt)
                participants[p_id] = ArenaParticipant(
                    id=p_id,
                    name=p_config.name,
                    endpoint=p_config.endpoint,
                    model=p_config.model,
                    system_prompt=system_prompt,
                    conversation=conversation,
                )

            judge_participant = None
            if judge_config:
                system_prompt = self._get_system_prompt_from_config(judge_config)
                judge_participant = ArenaParticipant(
                    id="judge",
                    name=judge_config.name,
                    endpoint=judge_config.endpoint,
                    model=judge_config.model,
                    system_prompt=system_prompt,
                    conversation=Conversation(
                        _messages=[{"role": "system", "content": system_prompt}]
                    ),
                )
                participants["judge"] = judge_participant

            turn_order = ArenaTurnOrder.SEQUENTIAL
            if arena_config.turn_order:
                try:
                    turn_order = ArenaTurnOrder(arena_config.turn_order.lower())
                except ValueError:
                    self.ui.pt_printer(
                        f"⚠️ Invalid turn_order '{arena_config.turn_order}' in pai.toml, defaulting to sequential."
                    )

            conversation_style = ArenaConversationStyle.PAIRWISE
            if arena_config.conversation_style:
                try:
                    conversation_style = ArenaConversationStyle(
                        arena_config.conversation_style.lower()
                    )
                except ValueError:
                    self.ui.pt_printer(
                        f"⚠️ Invalid conversation_style '{arena_config.conversation_style}' in pai.toml, defaulting to pairwise."
                    )
            arena_config_obj = Arena(
                name=arena_name,
                participants=participants,
                initiator_id=arena_config.initiator,
                judge=judge_participant,
                turn_order=turn_order,
                wildcards_enabled=arena_config.wildcards_enabled or False,
                conversation_style=conversation_style,
            )
            p_ids = [p for p in participants.keys() if p != "judge"]
            initiator_idx = p_ids.index(arena_config_obj.initiator_id)
            turn_order_ids = p_ids[initiator_idx:] + p_ids[:initiator_idx]
            arena_state = ArenaState(
                arena_config=arena_config_obj,
                turn_order_ids=turn_order_ids,
                max_turns=max_turns,
            )
            self.ui.enter_mode(UIMode.ARENA, clear_history=True)
            self.ui.state.arena = arena_state
            self.ui.arena_paused_event.clear()

            initiator_name = arena_state.arena_config.get_initiator().name
            self.ui.pt_printer(
                f"⚔️  Arena '{arena_name}' activated for {max_turns} turns."
            )
            self.ui.pt_printer(
                f"   Your next prompt will be given to '{initiator_name}' to start the conversation."
            )
            self.ui.pt_printer(
                "   The arena will auto-run. Use /pause, /resume, and /say to control it."
            )
        except (KeyError, FileNotFoundError) as e:
            self.ui.pt_printer(f"❌ Error setting up arena: {e}")


class PauseCommand(Command):
    @property
    def name(self):
        return "pause"

    def execute(self, app: "Application", param: str | None = None):
        if self.ui.state.mode != UIMode.ARENA:
            self.ui.pt_printer("❌ /pause is only available in arena mode.")
            return
        if not self.ui.arena_paused_event.is_set():
            self.ui.pt_printer("ℹ️  Arena is already paused.")
            return

        self.ui.arena_paused_event.clear()
        self.ui.pt_printer(
            "⏸️  Arena will pause after the current participant finishes."
        )


class ResumeCommand(Command):
    @property
    def name(self):
        return "resume"

    def execute(self, app: "Application", param: str | None = None):
        if self.ui.state.mode != UIMode.ARENA:
            self.ui.pt_printer("❌ /resume is only available in arena mode.")
            return
        if self.ui.arena_paused_event.is_set():
            self.ui.pt_printer("ℹ️  Arena is already running.")
            return

        self.ui.arena_paused_event.set()
        self.ui.pt_printer("▶️  Resuming arena conversation.")


class SayCommand(Command):
    @property
    def name(self):
        return "say"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        if not param:
            self.ui.pt_printer(f"Usage: /{self.name} <message>")
            return
        if self.ui.state.mode != UIMode.ARENA or not self.ui.state.arena:
            self.ui.pt_printer("❌ /say is only available in arena mode.")
            return

        # If arena is running, pause it first
        if self.ui.arena_paused_event.is_set():
            self.ui.arena_paused_event.clear()  # Clear the event to pause
            self.ui.pt_printer("⏸️  Arena paused to interject.")

        self.ui.state.arena.last_message = param
        self.ui.pt_printer("💬 Interjecting with message. Resuming arena...")
        self.ui.arena_paused_event.set()


class SaveCommand(Command):
    @property
    def name(self):
        return "save"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        if not param:
            self.ui.pt_printer(f"Usage: /{self.name} <filename>")
            return
        if self.ui.state.mode == UIMode.COMPLETION:
            self.ui.pt_printer("❌ /save is only available in chat mode.")
            return

        snapshot_path = self.ui.snapshots_dir / f"{param.strip()}.json"
        try:
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(self.ui.conversation.to_json(), f, indent=2)
            self.ui.pt_printer(f"💾 Session snapshot saved to '{snapshot_path}'")
        except Exception as e:
            self.ui.pt_printer(f"❌ Error saving session snapshot: {e}")


class LoadCommand(Command):
    @property
    def name(self):
        return "load"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        if not param:
            self.ui.pt_printer(f"Usage: /{self.name} <filename>")
            return
        clean_param = param.strip()
        json_path = self.ui.snapshots_dir / f"{clean_param}.json"
        pickle_path = self.ui.snapshots_dir / f"{clean_param}.pkl"

        if not json_path.exists() and not pickle_path.exists():
            self.ui.pt_printer(f"❌ Session snapshot not found for '{clean_param}'.")
            return

        try:
            loaded_conversation = None
            if json_path.exists():
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                    loaded_conversation = Conversation.from_json(data)
                self.ui.pt_printer(
                    f"🔄 Session snapshot loaded from JSON file: '{json_path}'."
                )
            elif pickle_path.exists():
                import pickle

                self.ui.pt_printer(
                    "Legacy .pkl snapshot found. Loading and converting to JSON."
                )
                with open(pickle_path, "rb") as f:
                    loaded_conversation = pickle.load(f)
                # Save back as JSON for future use
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(loaded_conversation.to_json(), f, indent=2)
                pickle_path.unlink()  # Remove old pickle file
                self.ui.pt_printer(
                    f"✅ Converted to '{json_path}' and removed old file."
                )

            if not isinstance(loaded_conversation, Conversation):
                raise TypeError("File does not contain a valid Conversation object.")

            # Reset state before loading
            self.ui.enter_mode(UIMode.CHAT, clear_history=False)
            self.ui.conversation = loaded_conversation

            # Print last few messages to give context
            history = self.ui.conversation.get_history()
            if history:
                self.ui.pt_printer("\n--- Context from loaded history ---")
                for msg in history[-2:]:
                    content = msg.get("content") or ""
                    if content:
                        self.ui.pt_printer(f"[{msg['role']}]> {str(content)[:200]}...")
                self.ui.pt_printer("---------------------------------")

        except (TypeError, Exception) as e:
            self.ui.pt_printer(f"❌ Error loading session: {e}")


class ToggleModeCommand(Command):
    @property
    def name(self):
        return "mode"

    def execute(self, app: "Application", param: str | None = None):
        new_mode = (
            UIMode.COMPLETION
            if self.ui.state.mode != UIMode.COMPLETION
            else UIMode.CHAT
        )
        self.ui.enter_mode(new_mode, clear_history=True)
        mode_name = new_mode.name.replace("_", " ").title()
        self.ui.pt_printer(f"✅ Switched to {mode_name} mode.")


class ToggleMultilineCommand(Command):
    @property
    def name(self):
        return "multiline"

    def execute(self, app: "Application", param: str | None = None):
        self.ui.state.multiline_input = not self.ui.state.multiline_input
        if self.ui.state.multiline_input:
            self.ui.pt_printer(
                "✅ Multiline input enabled. Use Esc+Enter or Alt+Enter to submit."
            )
        else:
            self.ui.pt_printer("✅ Multiline input disabled.")


# --- Command Handler ---
class CommandHandler:
    """Parses and executes slash commands by dispatching to Command objects."""

    def __init__(self, ui: "InteractiveUI"):
        self.ui = ui
        self.commands: dict[str, Command] = {}
        self._register_commands()

    @property
    def completion_list(self) -> list[str]:
        """Returns a sorted list of all command names with a '/' prefix for completion."""
        return sorted([f"/{name}" for name in self.commands.keys()])

    def _register_commands(self):
        """Initializes and registers all Command subclasses automatically."""
        # This will find all concrete subclasses of Command defined in this module.
        for cmd_class in Command.__subclasses__():
            # Skip abstract base classes if any were defined to inherit from Command
            if not inspect.isabstract(cmd_class):
                instance = cmd_class(self.ui)
                self.commands[instance.name] = instance
                for alias in instance.aliases:
                    self.commands[alias] = instance

    def handle(self, text: str, app: "Application"):
        """Parses the command text and calls the appropriate handler."""
        command_body = text.lstrip("/")
        parts = command_body.split(" ", 1)
        cmd_name, param = parts[0].lower(), parts[1] if len(parts) > 1 else None

        # Find all commands that match the prefix
        matching_keys = [key for key in self.commands if key.startswith(cmd_name)]

        if not matching_keys:
            self.ui.pt_printer("❌ Unknown command.")
            return

        # Check for ambiguity among the matched commands
        unique_commands = {self.commands[key] for key in matching_keys}

        if len(unique_commands) > 1:
            # It's ambiguous, list the possible full commands.
            # We use a set to avoid listing the same command multiple times if aliases matched.
            possible_cmds = sorted({cmd.name for cmd in unique_commands})
            self.ui.pt_printer(
                f"❌ Ambiguous command. Possibilities: {', '.join(f'/{c}' for c in possible_cmds)}"
            )
            return

        # Unambiguous match, get the single command instance
        command = unique_commands.pop()

        # The command.requires_param check is now handled within each
        # command's `execute` method to allow for more detailed, command-specific
        # help text when a command is incomplete.
        command.execute(app, param)
