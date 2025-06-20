"""
Command parsing and execution for the interactive UI. This module uses a
class-based approach where each command is a self-contained object.
"""

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional
from .tools import get_tool_schemas
from .models import Arena, ArenaParticipant, Conversation

if TYPE_CHECKING:
    from .pai import InteractiveUI
    from prompt_toolkit.application import Application


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
    def aliases(self) -> List[str]:
        """A list of alternative names for the command."""
        return []

    @property
    def requires_param(self) -> bool:
        """Whether the command requires a parameter to function."""
        return False

    @abstractmethod
    def execute(self, app: "Application", param: Optional[str] = None):
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

    def execute(self, app: "Application", param: Optional[str] = None):
        app.exit()


class StatsCommand(Command):
    @property
    def name(self):
        return "stats"

    def execute(self, app: "Application", param: Optional[str] = None):
        from .log_utils import print_stats

        print_stats(self.ui.client.stats, printer=self.ui.pt_printer)


class HelpCommand(Command):
    @property
    def name(self):
        return "help"

    def execute(self, app: "Application", param: Optional[str] = None):
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
  /stream, /verbose, /debug, /tools - Toggle flags on/off
  --- Chat Mode Only ---
  /system <text>         - Set a new system prompt (clears history)
  /history               - Show conversation history
  /clear                 - Clear conversation history
  /prompts               - List available, loadable system prompts
  /prompt <name>         - Load a system prompt from file (clears history)
  /agent                 - Start agent mode for OpenAI-compatible tool-use
  /legacy_agent          - Start agent mode for models without native tool-use
  /arena <name> [turns]  - Start a multi-model arena conversation (clears history)
    """
        )


class EndpointsCommand(Command):
    @property
    def name(self):
        return "endpoints"

    def execute(self, app: "Application", param: Optional[str] = None):
        self.ui.pt_printer("Available Endpoints:")
        for ep in self.ui.client.all_endpoints:
            self.ui.pt_printer(f" - {ep['name']}")


class SwitchCommand(Command):
    @property
    def name(self):
        return "switch"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: Optional[str] = None):
        self.ui.client.switch_endpoint(param)


class ModelCommand(Command):
    @property
    def name(self):
        return "model"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: Optional[str] = None):
        self.ui.client.config.model_name = param
        self.ui.pt_printer(f"✅ Model set to: {param}")


class TempCommand(Command):
    @property
    def name(self):
        return "temp"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: Optional[str] = None):
        try:
            self.ui.args.temperature = float(param)
            self.ui.pt_printer(f"✅ Temp set to: {self.ui.args.temperature}")
        except (ValueError, TypeError):
            self.ui.pt_printer("❌ Invalid value.")


class TokensCommand(Command):
    @property
    def name(self):
        return "tokens"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: Optional[str] = None):
        try:
            self.ui.args.max_tokens = int(param)
            self.ui.pt_printer(f"✅ Max tokens set to: {self.ui.args.max_tokens}")
        except (ValueError, TypeError):
            self.ui.pt_printer("❌ Invalid value.")


class TimeoutCommand(Command):
    @property
    def name(self):
        return "timeout"

    @property
    def requires_param(self):
        return True

    def execute(self, app: "Application", param: Optional[str] = None):
        try:
            timeout_val = int(param)
            if timeout_val <= 0:
                self.ui.pt_printer("❌ Timeout must be a positive integer.")
                return
            # This updates the timeout for the *current* endpoint config
            self.ui.client.config.timeout = timeout_val
            self.ui.pt_printer(f"✅ Request timeout set to: {timeout_val}s")
        except (ValueError, TypeError):
            self.ui.pt_printer("❌ Invalid value. Please provide an integer.")


class ToggleStreamCommand(Command):
    @property
    def name(self):
        return "stream"

    def execute(self, app: "Application", param: Optional[str] = None):
        self.ui.args.stream = not self.ui.args.stream
        self.ui.pt_printer(
            f"✅ Streaming {'enabled' if self.ui.args.stream else 'disabled'}."
        )


class ToggleVerboseCommand(Command):
    @property
    def name(self):
        return "verbose"

    def execute(self, app: "Application", param: Optional[str] = None):
        self.ui.args.verbose = not self.ui.args.verbose
        self.ui.pt_printer(
            f"✅ Verbose mode {'enabled' if self.ui.args.verbose else 'disabled'}."
        )


class ToggleDebugCommand(Command):
    @property
    def name(self):
        return "debug"

    def execute(self, app: "Application", param: Optional[str] = None):
        self.ui.client.display.debug_mode = not self.ui.client.display.debug_mode
        self.ui.pt_printer(
            f"✅ Debug mode {'enabled' if self.ui.client.display.debug_mode else 'disabled'}."
        )


class ToggleToolsCommand(Command):
    @property
    def name(self):
        return "tools"

    def execute(self, app: "Application", param: Optional[str] = None):
        # Prevent enabling tools if none were loaded at startup.
        if not self.ui.client.tools_enabled and not get_tool_schemas():
            self.ui.pt_printer(
                "❌ No tools loaded. To use tools, please restart and add the `--tools` flag."
            )
            return

        self.ui.client.tools_enabled = not self.ui.client.tools_enabled
        self.ui.pt_printer(
            f"✅ Tool calling {'enabled' if self.ui.client.tools_enabled else 'disabled'}."
        )


class HistoryCommand(Command):
    @property
    def name(self):
        return "history"

    def execute(self, app: "Application", param: Optional[str] = None):
        if self.ui.is_chat_mode:
            self.ui.pt_printer(json.dumps(self.ui.conversation.get_history(), indent=2))
        else:
            self.ui.pt_printer("❌ /history is only available in chat mode.")


class ClearCommand(Command):
    @property
    def name(self):
        return "clear"

    def execute(self, app: "Application", param: Optional[str] = None):
        if self.ui.is_chat_mode:
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

    def execute(self, app: "Application", param: Optional[str] = None):
        if self.ui.is_chat_mode:
            self.ui.native_agent_mode = False
            self.ui.legacy_agent_mode = False
            self.ui.conversation.set_system_prompt(param)
            self.ui.pt_printer("🤖 System prompt set.")
        else:
            self.ui.pt_printer("❌ /system is only available in chat mode.")


class PromptsCommand(Command):
    @property
    def name(self):
        return "prompts"

    def execute(self, app: "Application", param: Optional[str] = None):
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

    def execute(self, app: "Application", param: Optional[str] = None):
        if not self.ui.is_chat_mode:
            self.ui.pt_printer("❌ /prompt is only available in chat mode.")
            return

        self.ui.native_agent_mode = False
        self.ui.legacy_agent_mode = False

        prompt_path = self.ui.prompts_dir / f"{param}.md"
        if not prompt_path.exists():
            prompt_path = self.ui.prompts_dir / f"{param}.txt"

        if prompt_path.exists() and prompt_path.is_file():
            content = prompt_path.read_text(encoding="utf-8")
            self.ui.conversation.set_system_prompt(content)
            self.ui.pt_printer(
                f"🤖 System prompt loaded from '{param}'. History cleared."
            )
        else:
            self.ui.pt_printer(
                f"❌ Prompt '{param}' not found in '{self.ui.prompts_dir}'."
            )


class AgentCommand(Command):
    @property
    def name(self):
        return "agent"

    def execute(self, app: "Application", param: Optional[str] = None):
        """Loads the special 'code_editor' prompt to start an agent session."""
        if not self.ui.is_chat_mode:
            self.ui.pt_printer("❌ /agent is only available in chat mode.")
            return

        if not self.ui.args.tools:
            self.ui.pt_printer(
                "⚠️  Warning: Native agent mode works best with the --tools flag enabled at startup."
            )

        self.ui.native_agent_mode = True
        self.ui.legacy_agent_mode = False
        self.ui.client.tools_enabled = True

        param = "code_editor"
        prompt_path = self.ui.prompts_dir / f"{param}.md"

        if prompt_path.exists() and prompt_path.is_file():
            content = prompt_path.read_text(encoding="utf-8")
            self.ui.conversation.set_system_prompt(content)
            self.ui.pt_printer(
                f"🤖 Native Agent mode enabled (loaded '{param}' prompt). History cleared."
            )
        else:
            self.ui.pt_printer(
                f"❌ Agent prompt '{param}.md' not found in '{self.ui.prompts_dir}'."
            )


class LegacyAgentCommand(Command):
    @property
    def name(self):
        return "legacy_agent"

    def execute(self, app: "Application", param: Optional[str] = None):
        """Loads a prompt that teaches a generic model to use tools via text."""
        if not self.ui.is_chat_mode:
            self.ui.pt_printer("❌ /legacy_agent is only available in chat mode.")
            return

        if not get_tool_schemas():
            self.ui.pt_printer(
                "❌ No tools loaded. To use agent mode, please restart and add the `--tools` flag."
            )
            return

        self.ui.native_agent_mode = False
        self.ui.legacy_agent_mode = True
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
                f"🤖 Legacy Agent mode enabled. Tools will be used via text prompt."
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

    def execute(self, app: "Application", param: Optional[str] = None):
        if not self.ui.is_chat_mode:
            self.ui.pt_printer("❌ /arena is only available in chat mode.")
            return

        parts = param.strip().split(" ", 1)
        arena_name, max_turns_str = (parts + [None])[:2]

        try:
            max_turns = int(max_turns_str) if max_turns_str else 10
        except (ValueError, TypeError):
            self.ui.pt_printer("❌ Invalid value for turns. Must be an integer.")
            return

        arenas_config = self.ui.client.raw_config.get("arenas", {})
        arena_config = arenas_config.get(arena_name)

        if not arena_config:
            self.ui.pt_printer(f"❌ Arena '{arena_name}' not found in polyglot.toml.")
            return

        participant_configs = arena_config.get("participants", {})
        if len(participant_configs) != 2:
            self.ui.pt_printer(f"❌ Arena '{arena_name}' must have exactly 2 participants.")
            return

        try:
            participants = {}
            for p_id, p_config in participant_configs.items():
                prompt_key = p_config["system_prompt_key"]
                prompt_path = self.ui.prompts_dir / f"{prompt_key}.md"
                if not prompt_path.exists():
                    prompt_path = self.ui.prompts_dir / f"{prompt_key}.txt"

                if not prompt_path.is_file():
                    raise FileNotFoundError(f"System prompt file for '{prompt_key}' not found.")

                system_prompt = prompt_path.read_text(encoding="utf-8")

                conversation = Conversation()
                conversation.set_system_prompt(system_prompt)

                participants[p_id] = ArenaParticipant(
                    id=p_id,
                    name=p_config["name"],
                    endpoint=p_config["endpoint"],
                    model=p_config["model"],
                    system_prompt=system_prompt,
                    conversation=conversation,
                )

            current_endpoint = self.ui.client.config.name
            for p in participants.values():
                if p.endpoint != current_endpoint:
                    self.ui.pt_printer(
                        f"❌ Participant '{p.name}' uses endpoint '{p.endpoint}', but session is on '{current_endpoint}'."
                    )
                    self.ui.pt_printer(
                        "   (Cross-endpoint arenas are not yet supported. Please /switch first.)"
                    )
                    return

            arena = Arena(
                name=arena_name,
                participants=participants,
                initiator_id=arena_config["initiator"],
            )

            self.ui.conversation.clear()
            self.ui.native_agent_mode = False
            self.ui.legacy_agent_mode = False
            self.ui.arena_mode = True
            self.ui.active_arena = arena
            self.ui.arena_max_turns = max_turns

            initiator_name = arena.get_initiator().name
            self.ui.pt_printer(f"⚔️  Arena '{arena_name}' activated for {max_turns} turns.")
            self.ui.pt_printer(
                f"   Your next prompt will be given to '{initiator_name}' to start the conversation."
            )

        except (KeyError, FileNotFoundError) as e:
            self.ui.pt_printer(f"❌ Error setting up arena: {e}")


class ToggleModeCommand(Command):
    @property
    def name(self):
        return "mode"

    def execute(self, app: "Application", param: Optional[str] = None):
        self.ui.is_chat_mode = not self.ui.is_chat_mode
        self.ui.native_agent_mode = False
        self.ui.legacy_agent_mode = False
        mode_name = "Chat" if self.ui.is_chat_mode else "Completion"
        self.ui.conversation.clear()
        self.ui.pt_printer(f"✅ Switched to {mode_name} mode. History cleared.")


# --- Command Handler ---
class CommandHandler:
    """Parses and executes slash commands by dispatching to Command objects."""

    def __init__(self, ui: "InteractiveUI"):
        self.ui = ui
        self.commands: Dict[str, Command] = {}
        self._register_commands()

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
            ToggleToolsCommand,
            HistoryCommand,
            ClearCommand,
            SystemCommand,
            PromptsCommand,
            PromptCommand,
            AgentCommand,
            LegacyAgentCommand,
            ArenaCommand,
            ToggleModeCommand,
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
