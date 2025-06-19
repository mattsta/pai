"""
Command parsing and execution for the interactive UI. This module uses a
class-based approach where each command is a self-contained object.
"""

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

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
  /mode                  - Toggle between chat and completion mode (clears history)
  /stream, /verbose, /debug, /tools - Toggle flags on/off
  --- Chat Mode Only ---
  /system <text>         - Set a new system prompt (clears history)
  /history               - Show conversation history
  /clear                 - Clear conversation history
  /prompts               - List available, loadable system prompts
  /prompt <name>         - Load a system prompt from file (clears history)
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

        prompt_path = self.ui.prompts_dir / f"{param}.md"
        if not prompt_path.exists():
            prompt_path = self.ui.prompts_dir / f"{param}.txt"

        if prompt_path.exists() and prompt_path.is_file():
            content = prompt_path.read_text(encoding="utf-8")
            self.ui.conversation.set_system_prompt(content)
            self.ui.pt_printer(f"🤖 System prompt loaded from '{param}'. History cleared.")
        else:
            self.ui.pt_printer(
                f"❌ Prompt '{param}' not found in '{self.ui.prompts_dir}'."
            )


class ToggleModeCommand(Command):
    @property
    def name(self):
        return "mode"

    def execute(self, app: "Application", param: Optional[str] = None):
        self.ui.is_chat_mode = not self.ui.is_chat_mode
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
            ToggleStreamCommand,
            ToggleVerboseCommand,
            ToggleDebugCommand,
            ToggleToolsCommand,
            HistoryCommand,
            ClearCommand,
            SystemCommand,
            PromptsCommand,
            PromptCommand,
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
