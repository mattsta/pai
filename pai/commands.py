"""
Command parsing and execution for the interactive UI.
"""
import json
from typing import TYPE_CHECKING, Dict, Callable

if TYPE_CHECKING:
    from .pai import InteractiveUI
    from prompt_toolkit.application import Application


class CommandHandler:
    """Parses and executes slash commands from the user."""

    def __init__(self, ui: "InteractiveUI"):
        self.ui = ui

    def handle(self, text: str, app: "Application"):
        """Parses the command text and calls the appropriate handler."""
        # To avoid circular imports at runtime, we import necessary functions here.
        from .log_utils import print_stats

        parts = text[1:].lower().split(" ", 1)
        cmd, params = parts[0], parts[1] if len(parts) > 1 else ""

        # This dictionary maps command strings to their handler methods.
        # It's defined here inside the handle method to capture the `app` instance
        # for commands that need it, like 'quit'.
        COMMANDS: Dict[str, Callable] = {
            "quit": app.exit,
            "exit": app.exit,
            "q": app.exit,
            "stats": lambda: print_stats(self.ui.client.stats, printer=self.ui.pt_printer),
            "help": self.cmd_help,
            "endpoints": self.cmd_endpoints,
            "switch": self.cmd_switch,
            "model": self.cmd_model,
            "temp": self.cmd_temp,
            "tokens": self.cmd_tokens,
            "stream": self.cmd_toggle_stream,
            "verbose": self.cmd_toggle_verbose,
            "debug": self.cmd_toggle_debug,
            "tools": self.cmd_toggle_tools,
            "history": self.cmd_history,
            "clear": self.cmd_clear,
            "system": self.cmd_system,
            "prompts": self.cmd_prompts,
            "prompt": self.cmd_prompt,
            "mode": self.cmd_toggle_mode,
        }

        if cmd in COMMANDS:
            if cmd in ["switch", "model", "temp", "tokens", "system", "prompt"]:
                if params:
                    COMMANDS[cmd](params)
                else:
                    self.ui.pt_printer(f"❌ Command '/{cmd}' requires a parameter.")
            else:
                COMMANDS[cmd]()
        else:
            self.ui.pt_printer("❌ Unknown command.")

    def cmd_endpoints(self):
        """Lists available endpoints."""
        self.ui.pt_printer("Available Endpoints:")
        for ep in self.ui.client.all_endpoints:
            self.ui.pt_printer(f" - {ep['name']}")

    def cmd_switch(self, params: str):
        """Switches to a different endpoint."""
        self.ui.client.switch_endpoint(params)

    def cmd_model(self, params: str):
        """Changes the model for the session."""
        self.ui.client.config.model_name = params
        self.ui.pt_printer(f"✅ Model set to: {params}")

    def cmd_temp(self, params: str):
        """Changes the temperature for requests."""
        try:
            self.ui.args.temperature = float(params)
            self.ui.pt_printer(f"✅ Temp set to: {self.ui.args.temperature}")
        except ValueError:
            self.ui.pt_printer("❌ Invalid value.")

    def cmd_tokens(self, params: str):
        """Changes max_tokens for requests."""
        try:
            self.ui.args.max_tokens = int(params)
            self.ui.pt_printer(f"✅ Max tokens set to: {self.ui.args.max_tokens}")
        except ValueError:
            self.ui.pt_printer("❌ Invalid value.")

    def cmd_toggle_stream(self):
        """Toggles streaming on or off."""
        self.ui.args.stream = not self.ui.args.stream
        self.ui.pt_printer(
            f"✅ Streaming {'enabled' if self.ui.args.stream else 'disabled'}."
        )

    def cmd_toggle_verbose(self):
        """Toggles verbose mode on or off."""
        self.ui.args.verbose = not self.ui.args.verbose
        self.ui.pt_printer(
            f"✅ Verbose mode {'enabled' if self.ui.args.verbose else 'disabled'}."
        )

    def cmd_toggle_debug(self):
        """Toggles debug mode on or off."""
        self.ui.client.display.debug_mode = not self.ui.client.display.debug_mode
        self.ui.pt_printer(
            f"✅ Debug mode {'enabled' if self.ui.client.display.debug_mode else 'disabled'}."
        )

    def cmd_toggle_tools(self):
        """Toggles tool calling on or off."""
        self.ui.client.tools_enabled = not self.ui.client.tools_enabled
        self.ui.pt_printer(
            f"✅ Tool calling {'enabled' if self.ui.client.tools_enabled else 'disabled'}."
        )

    def cmd_history(self):
        """Shows the conversation history as JSON."""
        if self.ui.is_chat_mode:
            self.ui.pt_printer(json.dumps(self.ui.conversation.get_history(), indent=2))

    def cmd_clear(self):
        """Clears the conversation history."""
        if self.ui.is_chat_mode:
            self.ui.conversation.clear()
            self.ui.pt_printer("🧹 History cleared.")

    def cmd_system(self, params: str):
        """Sets a new system prompt."""
        if self.ui.is_chat_mode:
            self.ui.conversation.set_system_prompt(params)
            self.ui.pt_printer(f"🤖 System prompt set.")

    def cmd_prompts(self):
        """Lists available prompt files."""
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

    def cmd_prompt(self, params: str):
        """Loads a prompt file as the new system prompt."""
        if not self.ui.is_chat_mode:
            self.ui.pt_printer("❌ /prompt command only available in chat mode.")
            return

        prompt_path = self.ui.prompts_dir / f"{params}.md"
        if not prompt_path.exists():
            prompt_path = self.ui.prompts_dir / f"{params}.txt"

        if prompt_path.exists() and prompt_path.is_file():
            content = prompt_path.read_text(encoding="utf-8")
            self.ui.conversation.set_system_prompt(content)
            self.ui.pt_printer(f"🤖 System prompt loaded from '{params}'. History cleared.")
        else:
            self.ui.pt_printer(
                f"❌ Prompt '{params}' not found in '{self.ui.prompts_dir}'."
            )

    def cmd_toggle_mode(self):
        """Toggles between chat and completion mode."""
        self.ui.is_chat_mode = not self.ui.is_chat_mode
        mode_name = "Chat" if self.ui.is_chat_mode else "Completion"
        # Switching modes should start a fresh context, so we clear the history.
        self.ui.conversation.clear()
        self.ui.pt_printer(f"✅ Switched to {mode_name} mode. History cleared.")

    def cmd_help(self):
        """Prints the help message."""
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
