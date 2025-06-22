"""
Command parsing and execution for the interactive UI. This module uses a
class-based approach where each command is a self-contained object.
"""

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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
🔧 AVAILABLE COMMANDS:
  /help                  - Show this help message
  /stats                 - Show session statistics
  /quit, /exit, /q       - Exit the program
  /endpoints             - List available endpoints from config file
  /switch <name>         - Switch to a different endpoint
  /model <name>          - Change the default model for the session
  /temp <value>          - Change temperature (e.g., /temp 0.9)
  /tokens <num>          - Change max_tokens (e.g., /tokens 100)
  /timeout <seconds>     - Change request timeout (e.g., /timeout 120)
  /mode                  - Toggle between chat and completion mode (clears history)
  /multiline             - Toggle multi-line input mode (use Esc+Enter to submit)
  /stream, /verbose, /debug, /rich, /smooth, /tools, /confirm - Toggle flags on/off
  --- Chat Mode Only ---
  /system <text>         - Replace system prompt stack with this text
  /system show           - Show the current system prompt stack
  /system add <text>     - Add a new prompt to the stack
  /system pop            - Remove the last prompt from the stack
  /system clear          - Clear all system prompts from the stack
  /history               - Show conversation history
  /clear                 - Clear conversation history
  /save <name>           - Save the current chat session to a file
  /load <name>           - Load a chat session from a file
  /prompts               - List available, loadable system prompts
  /prompt <name>         - Load a system prompt from file (clears history)
  /agent                 - Start agent mode for OpenAI-compatible tool-use
  /legacy_agent          - Start agent mode for models without native tool-use
  --- Arena Mode Commands ---
  /arena <name> [turns]  - Start a multi-model arena conversation (clears history)
  /pause                 - Pause the arena conversation after the current turn
  /resume                - Resume a paused arena conversation
  /say <message>         - Interject with a message in a paused arena
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
        self.ui.client.switch_endpoint(param)


class ModelCommand(Command):
    @property
    def name(self):
        return "model"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        self.ui.client.set_model(param)


class TempCommand(Command):
    @property
    def name(self):
        return "temp"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        self.ui.set_temperature(param)


class TokensCommand(Command):
    @property
    def name(self):
        return "tokens"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        self.ui.set_max_tokens(param)


class TimeoutCommand(Command):
    @property
    def name(self):
        return "timeout"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        try:
            timeout_val = int(param)
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
        if param not in ["on", "off"]:
            self.ui.pt_printer("❌ Usage: /confirm on|off")
            return
        self.ui.set_confirm_tool_use(param == "on")


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
        if self.ui.state.mode != UIMode.COMPLETION:
            self.ui.pt_printer(json.dumps(self.ui.conversation.get_history(), indent=2))
        else:
            self.ui.pt_printer("❌ /history is only available in chat mode.")


class ClearCommand(Command):
    @property
    def name(self):
        return "clear"

    def execute(self, app: "Application", param: str | None = None):
        if self.ui.state.mode != UIMode.COMPLETION:
            self.ui.conversation.clear()
            self.ui.pt_printer("🧹 History cleared.")
        else:
            self.ui.pt_printer("❌ /clear is only available in chat mode.")


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

        parts = param.strip().split(" ", 1)
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

        prompt_path = self.ui.prompts_dir / f"{param}.md"
        if not prompt_path.exists():
            prompt_path = self.ui.prompts_dir / f"{param}.txt"

        if prompt_path.exists() and prompt_path.is_file():
            content = prompt_path.read_text(encoding="utf-8")
            # Loading a prompt now *adds* it to the stack instead of replacing.
            self.ui.conversation.add_system_prompt(content)
            self.ui.pt_printer(f"🤖 Added system prompt from '{param}' to stack.")
        else:
            self.ui.pt_printer(
                f"❌ Prompt '{param}' not found in '{self.ui.prompts_dir}'."
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
            self.ui.pt_printer(f"❌ Arena '{arena_name}' not found in polyglot.toml.")
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
        if self.ui.arena_paused_event.is_set():
            self.ui.pt_printer(
                "❌ Arena must be paused to interject. Use /pause first."
            )
            return

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
        import pickle

        if self.ui.state.mode == UIMode.COMPLETION:
            self.ui.pt_printer("❌ /save is only available in chat mode.")
            return

        session_path = self.ui.saved_sessions_dir / f"{param}.pkl"
        try:
            with open(session_path, "wb") as f:
                pickle.dump(self.ui.conversation, f)
            self.ui.pt_printer(f"💾 Session saved to '{session_path}'")
        except Exception as e:
            self.ui.pt_printer(f"❌ Error saving session: {e}")


class LoadCommand(Command):
    @property
    def name(self):
        return "load"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: str | None = None):
        import pickle

        session_path = self.ui.saved_sessions_dir / f"{param}.pkl"
        if not session_path.exists():
            self.ui.pt_printer(f"❌ Session file not found: '{session_path}'")
            return

        try:
            with open(session_path, "rb") as f:
                loaded_conversation = pickle.load(f)

            if not isinstance(loaded_conversation, Conversation):
                raise TypeError("File does not contain a valid Conversation object.")

            # Reset state before loading
            self.ui.enter_mode(UIMode.CHAT, clear_history=False)
            self.ui.conversation = loaded_conversation
            self.ui.pt_printer(f"🔄 Session loaded from '{session_path}'.")
            # Print last few messages to give context
            history = self.ui.conversation.get_history()
            if history:
                self.ui.pt_printer("\n--- Last message in loaded history ---")
                last_msg = history[-1]
                self.ui.pt_printer(
                    f"[{last_msg['role']}]> {last_msg['content'][:200]}..."
                )
                self.ui.pt_printer("------------------------------------")

        except (pickle.UnpicklingError, TypeError, Exception) as e:
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
        """Initializes and registers all Command subclasses."""
        command_classes = [
            HelpCommand,
            QuitCommand,
            StatsCommand,
            EndpointsCommand,
            SwitchCommand,
            ModelCommand,
            TempCommand,
            TokensCommand,
            TimeoutCommand,
            ToggleStreamCommand,
            ToggleVerboseCommand,
            ToggleDebugCommand,
            ToggleRichTextCommand,
            ToggleSmoothStreamCommand,
            ToggleConfirmCommand,
            ToggleToolsCommand,
            HistoryCommand,
            ClearCommand,
            SystemCommand,
            PromptsCommand,
            PromptCommand,
            AgentCommand,
            LegacyAgentCommand,
            ArenaCommand,
            PauseCommand,
            ResumeCommand,
            SayCommand,
            SaveCommand,
            LoadCommand,
            ToggleModeCommand,
            ToggleMultilineCommand,
        ]
        for cmd_class in command_classes:
            instance = cmd_class(self.ui)
            self.commands[instance.name] = instance
            for alias in instance.aliases:
                self.commands[alias] = instance

    def handle(self, text: str, app: "Application"):
        """Parses the command text and calls the appropriate handler."""
        parts = text[1:].split(" ", 1)
        cmd_name, param = parts[0].lower(), parts[1] if len(parts) > 1 else None

        command = self.commands.get(cmd_name)

        if not command:
            self.ui.pt_printer("❌ Unknown command.")
            return

        if command.requires_param and not param:
            self.ui.pt_printer(f"❌ Command '/{command.name}' requires a parameter.")
            return

        command.execute(app, param)
