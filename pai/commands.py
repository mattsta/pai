"""
Command parsing and execution for the interactive UI. This module uses a
class-based approach where each command is a self-contained object.
"""

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import Arena, ArenaParticipant, ArenaState, Conversation, UIMode
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
  /info <model_id>       - Show detailed info for a model from Hugging Face
  /temp <value>          - Set the generation temperature (0.0-2.0)
  /tokens <num>          - Set the max tokens for the response
  /timeout <seconds>     - Set the network request timeout

--- AGENT & TOOLS ---
  /agent                 - Enable native agent mode (requires --tools)
  /legacy_agent          - Enable agent mode for models without tool-calling
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
  /arena <name> [turns]  - Start a multi-model arena conversation
  /pause                 - Pause the arena after the current model's turn
  /resume                - Resume a paused arena
  /say <message>         - Interject with a message while arena is paused

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
        return True

    def execute(self, app: "Application", param: str | None = None):
        """Fetches and displays detailed model information from Hugging Face."""
        model_id = param.strip()

        async def _fetch_and_print_info():
            self.ui.pt_printer(
                f"⏳ Fetching info for '{model_id}' from Hugging Face API or cache..."
            )
            info = await self.ui.client.get_model_info(model_id)

            if not info:
                # The client method prints a more specific error.
                return

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold cyan", no_wrap=True)
            table.add_column()

            table.add_row("ID", info.get("id"))
            table.add_row("Type", info.get("pipeline_tag"))
            table.add_row("Library", info.get("library_name"))
            table.add_row("Downloads", f"{info.get('downloads', 0):,}")
            table.add_row("Likes", f"{info.get('likes', 0):,}")
            table.add_row("SHA", info.get("sha"))

            if last_modified_str := info.get("lastModified"):
                try:
                    dt = datetime.fromisoformat(last_modified_str.replace("Z", "+00:00"))
                    table.add_row("Last Modified", dt.strftime("%Y-%m-%d %H:%M UTC"))
                except (ValueError, TypeError):
                    table.add_row("Last Modified", last_modified_str)

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
        self.ui.client.set_model(param.strip())


class TempCommand(Command):
    @property
    def name(self):
        return "temp"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        self.ui.set_temperature(param.strip())


class TokensCommand(Command):
    @property
    def name(self):
        return "tokens"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        self.ui.set_max_tokens(param.strip())


class TimeoutCommand(Command):
    @property
    def name(self):
        return "timeout"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
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

    def execute(self, app: "Application", param: str | None = None):
        if self.ui.state.mode == UIMode.COMPLETION:
            self.ui.pt_printer("❌ /system is only available in chat mode.")
            return

        parts = param.lstrip().split(" ", 1)
        subcommand = parts[0].lower()
        text = parts[1] if len(parts) > 1 else None

        if subcommand == "add":
            if not text:
                self.ui.pt_printer("❌ Usage: /system add <prompt text>")
                return
            self.ui.conversation.add_system_prompt(text)
            self.ui.pt_printer("🤖 System prompt added to stack.")
        elif subcommand == "pop":
            popped = self.ui.conversation.pop_system_prompt()
            if popped:
                self.ui.pt_printer(
                    f"🤖 Popped system prompt: '{popped[:60].strip()}...'"
                )
            else:
                self.ui.pt_printer("🤖 System prompt stack is empty.")
        elif subcommand == "show":
            prompts = self.ui.conversation.get_system_prompts()
            if not prompts:
                self.ui.pt_printer("🤖 System prompt stack is empty.")
            else:
                self.ui.pt_printer("--- System Prompt Stack ---")
                for i, p in enumerate(prompts):
                    self.ui.pt_printer(f"[{i + 1}]> {p}")
                self.ui.pt_printer("---------------------------")
        elif subcommand == "clear":
            self.ui.conversation.clear_system_prompts()
            self.ui.pt_printer("🤖 All system prompts have been cleared.")
        else:
            # If no subcommand matches, the entire param is the prompt.
            # This replaces the entire stack.
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


class ArenaCommand(Command):
    @property
    def name(self):
        return "arena"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        if self.ui.state.mode == UIMode.COMPLETION:
            self.ui.pt_printer("❌ /arena is only available in chat mode.")
            return

        parts = param.strip().split(" ", 1)
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
                prompt_key = p_config.system_prompt_key
                prompt_path = self.ui.prompts_dir / f"{prompt_key}.md"
                if not prompt_path.exists():
                    prompt_path = self.ui.prompts_dir / f"{prompt_key}.txt"

                if not prompt_path.is_file():
                    raise FileNotFoundError(
                        f"System prompt file for '{prompt_key}' not found."
                    )

                system_prompt = prompt_path.read_text(encoding="utf-8")

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
                prompt_key = judge_config.system_prompt_key
                prompt_path = self.ui.prompts_dir / f"{prompt_key}.md"
                if not prompt_path.exists():
                    prompt_path = self.ui.prompts_dir / f"{prompt_key}.txt"

                if not prompt_path.is_file():
                    raise FileNotFoundError(
                        f"System prompt file for judge '{prompt_key}' not found."
                    )

                system_prompt = prompt_path.read_text(encoding="utf-8")
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

            arena_config_obj = Arena(
                name=arena_name,
                participants=participants,
                initiator_id=arena_config.initiator,
                judge=judge_participant,
            )

            # Setup turn order (without the judge), starting with the initiator
            p_ids = [p for p in participants.keys() if p != "judge"]
            initiator_idx = p_ids.index(arena_config_obj.initiator_id)
            turn_order_ids = p_ids[initiator_idx:] + p_ids[:initiator_idx]

            # Create the dynamic state object for the session
            arena_state = ArenaState(
                arena_config=arena_config_obj,
                turn_order_ids=turn_order_ids,
                max_turns=max_turns,
            )

            # Enter arena mode
            self.ui.enter_mode(UIMode.ARENA, clear_history=True)
            self.ui.state.arena = arena_state
            self.ui.arena_paused_event.clear()  # Start in a paused state

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

        command = self.commands.get(cmd_name)

        if not command:
            self.ui.pt_printer("❌ Unknown command.")
            return

        if command.requires_param and not param:
            self.ui.pt_printer(f"❌ Command '/{command.name}' requires a parameter.")
            return

        command.execute(app, param)
