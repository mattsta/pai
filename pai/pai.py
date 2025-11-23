"""
Polyglot AI: An Interactive, Multi-Provider CLI for the OpenAI API Format.
This version is a direct refactoring of the original code, preserving all features
and fixing the circular import error.
"""

import asyncio
import importlib.metadata
import json
import logging
import os
import pathlib
import re
import sys
import tomllib
from copy import deepcopy
from datetime import datetime
from html import escape

import httpx
import typer
import yaml
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, FuzzyCompleter, WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    DynamicContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import Frame, SearchToolbar

from .client import APIError, PolyglotClient
from .commands import CommandHandler
from .display import StreamingDisplay
from .log_utils import closing, print_stats
from .models import (
    ChatRequest,
    CompletionRequest,
    Conversation,
    LogManifest,
    PolyglotConfig,
    RuntimeConfig,
    UIMode,
    UIState,
)
from .orchestration import (
    ArenaOrchestrator,
    BaseOrchestrator,
    DefaultOrchestrator,
    LegacyAgentOrchestrator,
)
from .pricing import PricingService

# --- Protocol Adapter Imports ---
from .protocols import load_protocol_adapters
from .tools import get_tool_schemas, set_mcp_manager

# --- Global Definitions ---
session = PromptSession()

# Precompiled regex for stripping HTML tags from formatted text
_HTML_TAG_RE = re.compile(r"<[^<]+?>")


def _load_env_file(env_file: str | None = None) -> None:
    """Load environment variables from a .env file.

    If env_file is specified, loads from that path.
    Otherwise, auto-detects .env in current directory or parent directories.
    """
    env_paths = []
    if env_file:
        env_paths = [pathlib.Path(env_file)]
    else:
        # Auto-detect .env file
        cwd = pathlib.Path.cwd()
        env_paths = [cwd / ".env", cwd.parent / ".env"]

    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes if present
                            if (value.startswith('"') and value.endswith('"')) or \
                               (value.startswith("'") and value.endswith("'")):
                                value = value[1:-1]
                            # Only set if not already in environment
                            if key not in os.environ:
                                os.environ[key] = value
                print(f"📁 Loaded environment from: {env_path}")
                return
            except Exception as e:
                print(f"⚠️  Warning: Could not load {env_path}: {e}")


def _load_batch_prompts(batch_file: str) -> list[str]:
    """Load prompts from a batch file.

    Supports:
    - Plain text: One prompt per line
    - JSON: Array of strings or array of objects with 'prompt' key
    - YAML: List of strings or list of objects with 'prompt' key
    """
    path = pathlib.Path(batch_file)
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

    content = path.read_text(encoding="utf-8")

    # Try JSON first
    if path.suffix.lower() == ".json" or content.strip().startswith("["):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                prompts = []
                for item in data:
                    if isinstance(item, str):
                        prompts.append(item)
                    elif isinstance(item, dict) and "prompt" in item:
                        prompts.append(item["prompt"])
                return prompts
        except json.JSONDecodeError:
            pass

    # Try YAML
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            data = yaml.safe_load(content)
            if isinstance(data, list):
                prompts = []
                for item in data:
                    if isinstance(item, str):
                        prompts.append(item)
                    elif isinstance(item, dict) and "prompt" in item:
                        prompts.append(item["prompt"])
                return prompts
        except yaml.YAMLError:
            pass

    # Fall back to plain text (one prompt per line)
    lines = content.strip().split("\n")
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def print_banner():
    print("🪶 Polyglot AI: A Universal CLI for the OpenAI API Format 🪶")


class CommandCompleter(Completer):
    """
    A custom completer for slash commands.
    It only provides completions if the text starts with '/' and contains no spaces.
    """

    def __init__(self, command_list: list[str]):
        # Use a FuzzyCompleter for a better user experience.
        self.fuzzy_completer = FuzzyCompleter(
            WordCompleter(command_list, ignore_case=True)
        )

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        # Only complete if it's a command at the start of the line.
        if text.startswith("/") and " " not in text:
            yield from self.fuzzy_completer.get_completions(document, complete_event)


# Logging and statistics functions have been moved to `pai/log_utils.py`.


class DisplayAccessor:
    """
    Provides lazy, index-based access to StreamingDisplay objects.
    Displays are created on-demand when accessed, saving memory when
    concurrent operations aren't being used.
    """

    def __init__(self, ui: "InteractiveUI"):
        self._ui = ui

    def __getitem__(self, index: int) -> StreamingDisplay:
        """Returns the display at the given index, creating it if necessary."""
        if index < 0 or index >= self._ui.MAX_CONCURRENT:
            raise IndexError(
                f"Display index {index} out of range (0-{self._ui.MAX_CONCURRENT - 1})"
            )
        if index not in self._ui._displays:
            self._ui._create_display(index)
        return self._ui._displays[index]

    def __iter__(self):
        """Iterates over existing displays only (does not create new ones)."""
        for i in sorted(self._ui._displays.keys()):
            yield self._ui._displays[i]

    def __len__(self):
        """Returns the number of currently allocated displays."""
        return len(self._ui._displays)


class InteractiveUI:
    """Encapsulates all logic for the text user interface."""

    _ORCHESTRATOR_MAP = {
        UIMode.ARENA: ArenaOrchestrator,
        UIMode.LEGACY_AGENT: LegacyAgentOrchestrator,
        UIMode.CHAT: DefaultOrchestrator,
        UIMode.NATIVE_AGENT: DefaultOrchestrator,
        UIMode.COMPLETION: DefaultOrchestrator,
        UIMode.TEMPLATE_COMPLETION: DefaultOrchestrator,
    }
    MAX_CONCURRENT = 50

    def __init__(self, client: "PolyglotClient", runtime_config: RuntimeConfig):
        self.client = client
        self.runtime_config = runtime_config

        # State management (replaces is_chat_mode, native_agent_mode, etc.)
        initial_mode = UIMode.CHAT if runtime_config.chat else UIMode.COMPLETION
        self.state = UIState(mode=initial_mode)

        if self.state.mode != UIMode.COMPLETION and self.runtime_config.system:
            self.conversation.set_system_prompt(self.runtime_config.system)

        # Setup directories
        pai_user_dir = pathlib.Path.home() / ".pai"
        pai_user_dir.mkdir(exist_ok=True)
        self.prompts_dir = pathlib.Path("prompts")
        self.prompts_dir.mkdir(exist_ok=True)
        self.arenas_dir = pathlib.Path("arenas")
        self.arenas_dir.mkdir(exist_ok=True)
        self.snapshots_dir = pathlib.Path("session_snapshots")
        self.snapshots_dir.mkdir(exist_ok=True)

        # Setup UI components
        self.pt_printer = print_formatted_text
        self.history = FileHistory(str(pai_user_dir / "history.txt"))

        # Command handler must be created before the input buffer that uses it.
        self.command_handler = CommandHandler(self)
        command_completer = CommandCompleter(self.command_handler.completion_list)

        self.active_concurrent_count = 0
        # Lazy-allocated display cache: only create displays when needed
        # This saves ~14MB of memory by not pre-allocating 50 displays
        self._displays: dict[int, StreamingDisplay] = {}
        # Primary display (index 0) is always needed
        primary_display = self._create_display(0)
        self.client.display = primary_display

        self.streaming_output_buffer = self.client.display.output_buffer
        self.reasoning_output_buffer = self.client.display.reasoning_output_buffer

        self.log_manifest = None  # Initialize before use by start_new_log_session
        self.start_new_log_session()

        self.input_buffer = Buffer(
            name="input_buffer",
            multiline=Condition(lambda: self.state.multiline_input),
            history=self.history,
            completer=command_completer,
            enable_history_search=True,
            accept_handler=self._on_buffer_accepted,
        )

        # State management
        self.arena_paused_event = asyncio.Event()
        self.generation_in_progress = asyncio.Event()
        self.generation_task: asyncio.Task | None = None
        self.spinner_chars = ["|", "/", "-", "\\"]
        self.spinner_idx = 0

        self.confirm_session = PromptSession()

        # Build the application
        self.app = self._create_application()

    def _sanitize_for_path(self, title: str) -> str:
        """Sanitizes a string to be safe for use in a file path."""
        # Remove invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", title)
        # Replace spaces with underscores
        sanitized = sanitized.replace(" ", "_")
        # Truncate to a reasonable length
        return sanitized[:100]

    def _create_display(self, index: int) -> StreamingDisplay:
        """Creates and configures a new StreamingDisplay for the given index."""
        display = StreamingDisplay(
            debug_mode=self.runtime_config.debug,
            rich_text_mode=self.runtime_config.rich_text,
            smooth_stream_mode=self.runtime_config.smooth_stream,
            enhanced_debug_mode=self.runtime_config.enhanced_debug,
            clone_id=index,
        )
        display.set_printer(self.pt_printer, is_interactive=True)
        display.ui = self
        self._displays[index] = display
        return display

    def _ensure_displays(self, count: int) -> None:
        """Ensures that displays 0 through count-1 exist, creating them lazily."""
        for i in range(count):
            if i not in self._displays:
                self._create_display(i)

    @property
    def displays(self) -> "DisplayAccessor":
        """Returns a display accessor that creates displays on-demand."""
        return DisplayAccessor(self)

    def start_new_log_session(self, title: str | None = None):
        """Starts a new conversation and a new log directory."""
        # Save the manifest for the previous session if it exists.
        if self.log_manifest:
            self.save_log_manifest("new_session")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = f"{timestamp}-interactive"
        if title:
            sanitized_title = self._sanitize_for_path(title)
            dir_name = f"{timestamp}-{sanitized_title}"

        self.log_dir = pathlib.Path("logs") / dir_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Reset conversation and UI state for the new session
        self.conversation = Conversation()
        if self.reasoning_output_buffer:
            self.reasoning_output_buffer.reset()
        if self.streaming_output_buffer:
            self.streaming_output_buffer.reset()

        self._create_log_manifest()
        self.pt_printer(
            f"\n✨ New session started. Logs will be saved to: {self.log_dir}"
        )

    def rename_log_session(self, title: str):
        """Renames the current log directory by appending a title."""
        if not hasattr(self, "log_dir") or not self.log_dir.exists():
            self.pt_printer("❌ No active log session to rename.")
            return

        sanitized_title = self._sanitize_for_path(title)
        current_name = self.log_dir.name
        match = re.match(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", current_name)
        if not match:
            self.pt_printer(
                "❌ Could not find timestamp prefix in log directory name. Cannot rename."
            )
            return

        timestamp_prefix = match.group(1)
        new_dir_name = f"{timestamp_prefix}-{sanitized_title}"
        new_log_dir = self.log_dir.parent / new_dir_name

        if new_log_dir.exists():
            self.pt_printer(f"❌ A directory named '{new_dir_name}' already exists.")
            return

        try:
            self.log_dir.rename(new_log_dir)
            self.log_dir = new_log_dir
            self.pt_printer(
                f"✅ Session renamed. Logs are now being saved to: {self.log_dir}"
            )
        except Exception as e:
            self.pt_printer(f"❌ Failed to rename log directory: {e}")

    def _create_log_manifest(self):
        """Initializes the log manifest for the session."""
        self.log_manifest = LogManifest(
            session_id=str(self.conversation.conversation_id),
            start_time=self.client.stats.start_time,
        )

    def save_log_manifest(self, finish_reason: str):
        """Populates and saves the final log manifest for the session."""
        manifest = self.log_manifest
        stats = self.client.stats

        manifest.end_time = datetime.now()
        manifest.duration_seconds = (
            manifest.end_time - manifest.start_time
        ).total_seconds()
        manifest.finish_reason = finish_reason
        manifest.total_turns = len(self.conversation.turns)
        manifest.total_cost = stats.total_cost
        manifest.total_requests = stats.requests_sent
        manifest.successful_requests = stats.requests_sent - stats.errors
        manifest.failed_requests = stats.errors
        manifest.total_tokens_sent = stats.total_tokens_sent
        manifest.total_tokens_received = stats.total_tokens_received

        endpoints = set()
        models = set()
        for turn in self.conversation.turns:
            if turn.endpoint_name:
                endpoints.add(turn.endpoint_name)
            if turn.model_name:
                models.add(turn.model_name)
        manifest.endpoints_used = sorted(list(endpoints))
        manifest.models_used = sorted(list(models))

        # Get initial prompt from the first turn if it exists
        if self.conversation.turns:
            first_turn = self.conversation.turns[0]
            if "messages" in first_turn.request_data:
                user_msg = next(
                    (
                        m["content"]
                        for m in first_turn.request_data["messages"]
                        if m["role"] == "user"
                    ),
                    None,
                )
                manifest.initial_prompt = user_msg
            elif "prompt" in first_turn.request_data:
                manifest.initial_prompt = first_turn.request_data["prompt"]

        # Save to file
        manifest_path = self.log_dir / "manifest.yaml"
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                # Use Pydantic's model_dump to get a dict, then use yaml.dump
                manifest_dict = manifest.model_dump(
                    mode="json"
                )  # mode='json' converts datetimes to iso strings
                yaml.dump(manifest_dict, f, sort_keys=False, default_flow_style=False)
        except Exception as e:
            self.pt_printer(f"⚠️  Warning: Could not save log manifest: {e}")

    def _get_mode_display_name(self) -> str:
        """Returns the string name of the current interaction mode."""
        if self.state.mode == UIMode.ARENA and self.state.arena:
            status = "Running"
            if not self.generation_in_progress.is_set():
                status = "Finished"
            elif not self.arena_paused_event.is_set():
                status = "Paused"

            judge_str = " w/ Judge" if self.state.arena.arena_config.judge else ""
            return f"Arena: {self.state.arena.arena_config.name}{judge_str} ({status})"
        elif self.state.mode == UIMode.ARENA_SETUP:
            return "Arena Setup"
        elif self.state.mode == UIMode.NATIVE_AGENT:
            return "Agent"
        elif self.state.mode == UIMode.LEGACY_AGENT:
            return "Legacy Agent"
        elif self.state.mode == UIMode.TEMPLATE_COMPLETION:
            return "Template"
        elif self.state.mode == UIMode.CHAT:
            return "Chat"
        return "Completion"

    def _get_prompt_text(self) -> HTML:
        """Generates the HTML for the input prompt."""
        # If we are in arena mode, but there is no active generation task,
        # it means we are waiting for the very first prompt from the user.
        if (
            self.state.mode == UIMode.ARENA
            and self.state.arena
            and not self.generation_in_progress.is_set()
        ):
            initiator = self.state.arena.arena_config.get_initiator()
            # Use a specific prompt for the arena's first turn
            return HTML(
                f"<style fg='ansigreen'>⚔️  Prompt for {initiator.name}:</style> "
            )
        return HTML(
            f"<style fg='ansigreen'>👤 ({self._get_mode_display_name()}) User:</style> "
        )

    def _create_application(self) -> Application:
        """Constructs the prompt_toolkit Application object."""

        prompt_ui = VSplit(
            [
                Window(
                    FormattedTextControl(self._get_prompt_text),
                    # The width must calculate the length of the *unformatted* string.
                    width=lambda: len(
                        _HTML_TAG_RE.sub("", self._get_prompt_text().value)
                    )
                    + 1,
                ),
                Window(BufferControl(buffer=self.input_buffer)),
            ]
        )

        def get_status_text():
            if self.state.mode == UIMode.ARENA and not self.arena_paused_event.is_set():
                return HTML(
                    "<style fg='ansiyellow'>[--- Arena Paused --- Use /resume or /say &lt;...&gt; ---]</style>"
                )

            display = self.client.display
            status = display.status
            live_stats = display.current_request_stats

            if status == "Streaming" and live_stats:
                if display._smoothing_aborted:
                    return HTML(
                        "<style fg='ansiyellow'>[Smooth streaming disabled. Press Ctrl+C again to cancel.]</style>"
                    )
                spinner = self.spinner_chars[self.spinner_idx]
                self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
                time_since_last = display.time_since_last_token
                color = "ansigreen"
                if time_since_last > 2.0:
                    color = "ansiyellow"
                if time_since_last > 5.0:
                    color = "ansired"
                duration = live_stats.live_stream_duration
                return HTML(
                    f"<style fg='{color}'>[{spinner}] Streaming... ({duration:.1f}s duration, {time_since_last:.1f}s since last token)</style>"
                )
            elif status == "Waiting..." and live_stats:
                duration = live_stats.current_duration
                if duration > 5.0:
                    # After 5 seconds, show a timer to give feedback on long waits.
                    return HTML(
                        f"<style fg='ansiyellow'>[Sent. Waiting for response... ({duration:.0f}s)]</style>"
                    )
                return HTML("<style fg='ansiyellow'>[Waiting for response...]</style>")
            # Fallback for idle or other states where this component might be briefly visible.
            return HTML("")

        waiting_ui = Window(FormattedTextControl(get_status_text), height=1)

        reasoning_window = ConditionalContainer(
            Window(
                content=BufferControl(buffer=self.reasoning_output_buffer),
                wrap_lines=True,
                style="fg:grey",
            ),
            # Show the reasoning window whenever it has content.
            # It's cleared at the start of the next user prompt.
            filter=Condition(lambda: bool(self.reasoning_output_buffer.text)),
        )

        live_output_window = ConditionalContainer(
            Window(
                content=BufferControl(buffer=self.streaming_output_buffer),
                wrap_lines=True,
            ),
            filter=Condition(
                lambda: self.generation_in_progress.is_set()
                and self.active_concurrent_count <= 1
                and self.streaming_output_buffer.text
            ),
        )

        def get_concurrent_output_windows():
            """Dynamically creates concurrent output windows only when needed."""
            if (
                not self.generation_in_progress.is_set()
                or self.active_concurrent_count <= 1
            ):
                return Window()  # Empty placeholder
            # Only create windows for active displays (lazy allocation)
            windows = []
            for i in range(self.active_concurrent_count):
                display = self.displays[i]  # Creates display on-demand via accessor
                if display.output_buffer.text:
                    windows.append(
                        Frame(
                            body=Window(
                                content=BufferControl(buffer=display.output_buffer),
                                wrap_lines=True,
                            ),
                            title=f"Response {i + 1} ({display.status})",
                        )
                    )
            return HSplit(windows) if windows else Window()

        concurrent_output_container = ConditionalContainer(
            DynamicContainer(get_concurrent_output_windows),
            filter=Condition(
                lambda: self.generation_in_progress.is_set()
                and self.active_concurrent_count > 1
            ),
        )

        search_toolbar = SearchToolbar()
        toolbar_window = Window(
            content=FormattedTextControl(self._get_toolbar_text),
            height=4,  # Use a fixed height for a stable layout
            style="reverse",
        )

        layout = Layout(
            HSplit(
                [
                    reasoning_window,
                    live_output_window,
                    concurrent_output_container,
                    ConditionalContainer(
                        prompt_ui,
                        filter=Condition(
                            lambda: not self.generation_in_progress.is_set()
                            or self.state.mode in [UIMode.ARENA, UIMode.ARENA_SETUP]
                        ),
                    ),
                    ConditionalContainer(
                        waiting_ui,
                        filter=Condition(lambda: self.generation_in_progress.is_set()),
                    ),
                    search_toolbar,
                    toolbar_window,
                ]
            ),
            focused_element=self.input_buffer,
        )

        return Application(
            layout=layout,
            key_bindings=self._create_key_bindings(),
            refresh_interval=0.2,
            full_screen=False,
        )

    def _create_key_bindings(self) -> KeyBindings:
        """Creates key bindings, including custom Ctrl+C and Ctrl+D handlers."""
        kb = KeyBindings()

        @kb.add("escape", "enter", filter=Condition(lambda: self.state.multiline_input))
        def _(event):
            """
            Custom binding for submitting in multiline mode.
            The default 'enter' key will just insert a newline.
            """
            event.app.current_buffer.validate_and_handle()

        @kb.add("c-c", eager=True)
        def _(event):
            """
            Custom Ctrl+C handler.
            - If a generation is in progress, cancel it.
            - If the input buffer has text, clear it.
            - If the input buffer is empty, print a fake prompt to emulate a new line.
            """
            if self.generation_in_progress.is_set() and self.generation_task:
                display = self.client.display
                is_smooth_active = (
                    display.smooth_stream_mode and not display._smoothing_aborted
                )

                # First Ctrl+C in an active smooth stream: abort smoothing, not generation.
                if is_smooth_active:
                    # This must be run as a task as the handler itself is not async.
                    asyncio.create_task(display.abort_smoothing())
                else:
                    # Second Ctrl+C, or Ctrl+C in a non-smooth stream: cancel generation.
                    self.generation_task.cancel()
            else:
                if event.app.current_buffer.text:
                    event.app.current_buffer.reset()
                else:
                    # To create a "new line" effect, we print a line that looks
                    # like our prompt to the scrollback buffer. The application
                    # will then redraw the actual interactive prompt below it.
                    self.pt_printer(
                        HTML(
                            f"<style fg='ansigreen'>👤 ({self._get_mode_display_name()}) User:</style> "
                        )
                    )

        @kb.add("c-d", filter=Condition(lambda: not self.input_buffer.text), eager=True)
        def _(event):
            """Handle Ctrl+D on an empty buffer to exit cleanly."""
            event.app.exit(result=EOFError())

        # Merge our bindings with the defaults. Ours take precedence.
        return merge_key_bindings([kb, load_key_bindings()])

    def _on_buffer_accepted(self, buffer: Buffer):
        """Callback for when the user presses Enter on the input buffer."""
        user_input = buffer.text
        lstripped_input = user_input.lstrip()

        if not lstripped_input:
            buffer.reset()
            return

        # Check if we should block the input.
        if lstripped_input.startswith("/"):
            # Commands are always allowed to be processed.
            pass
        else:
            # It's a prompt. We should block it if a generation is active,
            # unless the arena is explicitly paused and waiting for input.
            is_paused_arena = (
                self.state.mode == UIMode.ARENA
                and not self.arena_paused_event.is_set()
                and self.generation_in_progress.is_set()
            )
            if self.generation_in_progress.is_set() and not is_paused_arena:
                self.pt_printer(
                    HTML(
                        "<style fg='ansiyellow'>ℹ️ A generation is in progress. Use a command or wait.</style>"
                    )
                )
                buffer.reset()
                return

        # If we've reached here, the input is valid to process.
        buffer.reset(append_to_history=True)

        self.pt_printer(
            HTML(
                f"\n<style fg='ansigreen'>👤 ({self._get_mode_display_name()}) User:</style> {escape(user_input)}"
            )
        )

        if lstripped_input.startswith("/"):
            self.command_handler.handle(lstripped_input, self.app)
        else:
            # A new prompt from the user means we should clear the previous turn's reasoning view.
            if self.reasoning_output_buffer:
                self.reasoning_output_buffer.reset()

            orchestrator = self._get_orchestrator()
            if orchestrator:
                if self.state.multiplier > 1:
                    from .orchestration.multiply import MultiplyOrchestrator

                    orchestrator = MultiplyOrchestrator(self)
                else:
                    self.active_concurrent_count = 1

                self.generation_in_progress.set()
                self.generation_task = asyncio.create_task(orchestrator.run(user_input))

    # Business logic for chat, agent, and arena modes has been extracted
    # to `pai/orchestration/` classes.

    def _get_orchestrator(self) -> BaseOrchestrator | None:
        """Selects the appropriate orchestrator based on the current UI mode."""
        orchestrator_class = self._ORCHESTRATOR_MAP.get(self.state.mode)
        if orchestrator_class:
            return orchestrator_class(self)
        return None

    async def _confirm_tool_call(self, tool_name: str, args: dict) -> bool:
        """Asks the user for confirmation to run a tool."""
        self.client.display.show_tool_call_request(tool_name, args)

        try:
            # This modal-like prompt will temporarily take over the input line
            result = await self.confirm_session.prompt_async(
                "Authorize this tool call? [y/N]: ",
            )
            approved = result.lower().strip() == "y"
            if not approved:
                self.pt_printer(HTML("  <style fg='ansiyellow'>-> Denied.</style>"))
            return approved
        except (EOFError, KeyboardInterrupt):
            self.pt_printer(HTML("\n<style fg='ansiyellow'>-> Denied by user.</style>"))
            return False

    def _get_toolbar_line1_text(self) -> str:
        """Generates the first line of the toolbar (context)."""
        prompt_count = len(self.conversation.get_system_prompts())
        mode_str = escape(self._get_mode_display_name())
        if prompt_count > 1:
            mode_str += f" <style fg='ansicyan'>Sys[{prompt_count}]</style>"
        if self.state.multiplier > 1:
            mode_str += (
                f" <style fg='ansired' bold='true'>x{self.state.multiplier}</style>"
            )

        if self.state.mode == UIMode.ARENA_SETUP and self.state.arena:
            arena_name = self.state.arena.arena_config.name
            num_participants = len(self.state.arena.arena_config.participants)
            return f"<style fg='ansiyellow'><b>🚧 Building Arena: {escape(arena_name)}</b></style> | {num_participants} Participants"
        elif self.state.mode == UIMode.ARENA and self.state.arena:
            p_configs = self.state.arena.arena_config.participants
            p_details = " vs ".join(
                [
                    f"{p.name} ({p.model})"
                    for p_id, p in p_configs.items()
                    if p_id != "judge"
                ]
            )
            judge_str = " w/ Judge" if self.state.arena.arena_config.judge else ""
            arena_name_esc = escape(self.state.arena.arena_config.name)
            p_details_esc = escape(p_details)
            return f"<style fg='ansiyellow'><b>⚔️ ARENA: {arena_name_esc}{judge_str}</b></style> | {p_details_esc}"
        else:
            endpoint_esc = escape(self.client.config.name)
            model_esc = escape(self.client.config.model_name)
            return (
                f"<b>{endpoint_esc.upper()}:{model_esc}</b> | <b>Mode:</b> {mode_str}"
            )

    def _get_toolbar_line2_text(self) -> str:
        """Generates the second line of the toolbar (performance stats)."""
        display = self.client.display
        session_stats = self.client.stats
        live_stats = display.current_request_stats
        parts = []

        status_esc = escape(display.status)
        parts.append(f"<style fg='ansimagenta'><b>Status: {status_esc}</b></style>")

        if live_stats and display.status in ["Waiting...", "Streaming"]:
            live_tps = live_stats.live_tok_per_sec
            live_tokens = live_stats.tokens_received
            status_color = (
                "ansigreen" if display.status == "Streaming" else "ansiyellow"
            )
            live_stats_str = f"<b>Live:</b> {live_tokens:4d} tk @ {live_tps:5.1f} tk/s"
            parts.append(f"<style fg='{status_color}'>{live_stats_str}</style>")
        else:
            last_req = session_stats.last_request_stats
            if last_req:
                last_tps = f"{last_req.final_tok_per_sec:5.1f} tk/s"
                last_tokens = f"{last_req.tokens_received:4d} tk"
                parts.append(f"<b>Last:</b> {last_tokens}, {last_tps}")
                reason = escape(last_req.finish_reason or "N/A")
                parts.append(f"<style fg='ansiyellow'><b>Stop:</b> {reason}</style>")

                mean_d_str = "--.-ms"
                if j_stats := last_req.jitter_stats:
                    if j_stats.mean_delta != "N/A":
                        try:
                            mean_d = float(j_stats.mean_delta)
                            mean_d_str = f"{mean_d:4.1f}ms"
                        except ValueError:
                            pass
                parts.append(f"<style fg='grey'><b>Avg Δ:</b> {mean_d_str}</style>")

        session_tokens = self.conversation.session_token_count
        total_stream_time = session_stats.total_stream_time
        total_received = session_stats.total_tokens_received
        total_cost = session_stats.total_cost

        if live_stats and display.status in ["Waiting...", "Streaming"]:
            session_tokens += live_stats.tokens_sent + live_stats.tokens_received
            total_stream_time += live_stats.live_stream_duration
            total_received += live_stats.tokens_received
            if live_stats.cost:
                total_cost += live_stats.cost.total_cost

        session_tps = total_received / max(total_stream_time, 1)

        # Budget-aware cost display
        budget = self.runtime_config.session_budget
        if budget is not None:
            percent_used = (total_cost / budget) * 100 if budget > 0 else 0
            if percent_used >= 100:
                cost_str = f"<style bg='ansired' fg='white'><b>OVER BUDGET:</b> ${total_cost:.4f}/${budget:.2f}</style>"
            elif percent_used >= 80:
                cost_str = f"<style fg='ansiyellow'><b>Cost:</b> ${total_cost:.4f}/${budget:.2f} ({percent_used:.0f}%)</style>"
            else:
                cost_str = f"<b>Cost:</b> ${total_cost:.4f}/${budget:.2f}"
        else:
            cost_str = f"<b>Cost:</b> ${total_cost:.4f}"

        parts.extend(
            [
                cost_str,
                f"<b>Total:</b> {session_tokens:5d} tk",
                f"<b>Avg:</b> {session_tps:5.1f} tk/s",
            ]
        )
        return " | ".join(parts)

    def _get_toolbar_line3_text(self) -> str:
        """Generates the third line of the toolbar (toggles)."""
        on = "<style fg='ansigreen'>ON</style>"
        off = "<style fg='default'>OFF</style>"
        core_toggles = [
            f"Stream: {on if self.runtime_config.stream else off}",
            f"Rich: {on if self.runtime_config.rich_text else off}",
            f"Smooth: {on if self.runtime_config.smooth_stream else off}",
            f"Multiline: {on if self.state.multiline_input else off}",
        ]
        return f"<b>Toggles</b> | {' | '.join(core_toggles)}"

    def _get_toolbar_line4_text(self) -> str:
        """Generates the fourth line of the toolbar (dynamic content)."""
        display = self.client.display
        is_agent_mode = self.state.mode in [
            UIMode.NATIVE_AGENT,
            UIMode.LEGACY_AGENT,
        ]
        is_smoothing_active = (
            self.runtime_config.smooth_stream and display.status == "Streaming"
        )

        if is_smoothing_active and (s_stats := display.smoothing_stats):
            parts = []
            if s_stats.stream_finished and s_stats.queue_size > 0:
                parts.append(
                    "<style fg='ansimagenta'><b>[Rendering Queue...]</b></style>"
                )
            parts.append(f"Queue: {s_stats.queue_size:4d}")
            if s_stats.smoothing_aborted:
                parts.append("Drain: LIVE")
            else:
                parts.append(f"Drain: {s_stats.buffer_drain_time_s:4.1f}s")

            if not s_stats.stream_finished:
                try:
                    min_d = float(s_stats.min_delta)
                    mean_d = float(s_stats.mean_delta)
                    stdev_d = float(s_stats.stdev_delta)
                    max_d = float(s_stats.max_delta)
                    parts.append(
                        f"Δ (min/mean/stdev/max ms): {min_d:4.1f}/{mean_d:4.1f}/{stdev_d:4.1f}/{max_d:4.1f}"
                    )
                except ValueError:
                    parts.append("Δ (min/mean/stdev/max ms): --.-/--.-/--.-/--.-")
                parts.append(f"G/B: {s_stats.gaps:2d}/{s_stats.bursts:3d}")
            else:
                parts.append("Δ (min/mean/stdev/max ms): --.-/--.-/--.-/--.-")
                parts.append("G/B: --/---")
            return f"<b>Smooth Stats</b> | {' | '.join(parts)}"

        if (
            self.state.mode == UIMode.ARENA
            and self.state.arena
            and self.generation_in_progress.is_set()
        ):
            state = self.state.arena
            if state.turn_order_ids:
                num_participants = len(state.turn_order_ids)
                turn_num = (state.current_speech // num_participants) + 1
                parts = [
                    f"Turn: {turn_num}/{state.max_turns}",
                    f"Speech: {state.current_speech + 1}/{state.max_speeches}",
                ]
                if state.current_speech < state.max_speeches:
                    current_participant_id = state.turn_order_ids[0]
                    participant = state.arena_config.get_participant(
                        current_participant_id
                    )
                    if participant:
                        parts.append(f"Speaking: <b>{escape(participant.name)}</b>")
                return f"<style fg='ansimagenta'><b>Arena Progress</b> | {' | '.join(parts)}</style>"

        if is_agent_mode:
            parts = [
                f"Loops: {self.state.agent_loops}",
                f"Tools Used: {self.state.tools_used}",
            ]
            return f"<style fg='ansimagenta'><b>Agent Stats</b> | {' | '.join(parts)}</style>"

        on = "<style fg='ansigreen'>ON</style>"
        off = "<style fg='default'>OFF</style>"
        yellow_on = "<style fg='ansiyellow'>ON</style>"
        # Use compact labels to save space
        agent_toggles = [
            f"T: {on if self.client.tools_enabled else off}",
            f"Cf: {yellow_on if self.runtime_config.confirm_tool_use else off}",
            f"KR: {yellow_on if self.runtime_config.keep_reasoning else off}",
            f"D: {yellow_on if display.debug_mode else off}",
            f"ED: {yellow_on if self.runtime_config.enhanced_debug else off}",
            f"V: {yellow_on if self.runtime_config.verbose else off}",
        ]
        log_part = f"Log: {escape(str(self.log_dir))}"
        agent_part = f"<b>Agent:</b> {' | '.join(agent_toggles)}"
        return f"{log_part}    {agent_part}"

    def _get_toolbar_text(self) -> HTML:
        """Generates the HTML for the multi-line bottom toolbar."""
        try:
            line1 = self._get_toolbar_line1_text()
            line2 = self._get_toolbar_line2_text()
            line3 = self._get_toolbar_line3_text()
            line4 = self._get_toolbar_line4_text()
            return HTML(f"{line1}\n{line2}\n{line3}\n{line4}")
        except Exception as e:
            # If any rendering fails, return a safe, minimal toolbar to prevent crashing.
            return HTML(
                f"<style bg='ansired' fg='white'>[Toolbar Error: {escape(str(e), quote=False)}]</style>"
            )

    def enter_mode(self, mode: UIMode, clear_history: bool = True):
        """Handles the logic of switching UI modes."""
        self.state.mode = mode
        # Preserve arena state only when in Arena or Arena Setup modes.
        if mode not in [UIMode.ARENA, UIMode.ARENA_SETUP]:
            self.state.arena = None

        # Reset agent-related flags and stats when switching modes.
        if mode not in [UIMode.NATIVE_AGENT, UIMode.LEGACY_AGENT]:
            self.state.tools_used = 0
            self.state.agent_loops = 0

        if mode != UIMode.TEMPLATE_COMPLETION:
            self.state.chat_template = None
            self.state.chat_template_obj = None

        if clear_history:
            self.conversation.clear()
            self.pt_printer("🧹 History cleared.")

    def set_temperature(self, temp_str: str | None):
        """Sets the temperature for generation."""
        try:
            self.runtime_config.temperature = float(temp_str)
            self.pt_printer(f"✅ Temp set to: {self.runtime_config.temperature}")
        except (ValueError, TypeError):
            self.pt_printer("❌ Invalid value for temperature.")

    def set_max_tokens(self, tokens_str: str | None):
        """Sets the max tokens for generation."""
        try:
            self.runtime_config.max_tokens = int(tokens_str)
            self.pt_printer(f"✅ Max tokens set to: {self.runtime_config.max_tokens}")
        except (ValueError, TypeError):
            self.pt_printer("❌ Invalid value for max tokens.")

    def set_confirm_tool_use(self, confirm: bool):
        """Toggles tool confirmation mode."""
        self.runtime_config.confirm_tool_use = confirm
        self.pt_printer(
            f"✅ Tool confirmation mode {'enabled' if self.runtime_config.confirm_tool_use else 'disabled'}."
        )

    def toggle_keep_reasoning(self):
        """Toggles keeping reasoning in history."""
        self.runtime_config.keep_reasoning = not self.runtime_config.keep_reasoning
        self.pt_printer(
            f"✅ Keep reasoning {'enabled' if self.runtime_config.keep_reasoning else 'disabled'}."
        )

    def toggle_stream(self):
        """Toggles streaming mode."""
        self.runtime_config.stream = not self.runtime_config.stream
        self.pt_printer(
            f"✅ Streaming {'enabled' if self.runtime_config.stream else 'disabled'}."
        )

    def toggle_verbose(self):
        """Toggles verbose logging."""
        self.runtime_config.verbose = not self.runtime_config.verbose
        self.pt_printer(
            f"✅ Verbose mode {'enabled' if self.runtime_config.verbose else 'disabled'}."
        )

    def toggle_debug(self):
        """Toggles protocol debug mode."""
        self.client.display.debug_mode = not self.client.display.debug_mode
        self.pt_printer(
            f"✅ Debug mode {'enabled' if self.client.display.debug_mode else 'disabled'}."
        )

    def toggle_enhanced_debug(self):
        """Toggles enhanced debug mode."""
        self.runtime_config.enhanced_debug = not self.runtime_config.enhanced_debug
        self.client.display.enhanced_debug_mode = self.runtime_config.enhanced_debug
        # Enhanced debug implies regular debug. This makes the command a master toggle for this view.
        self.client.display.debug_mode = self.runtime_config.enhanced_debug
        self.pt_printer(
            f"✅ Enhanced debug mode {'enabled' if self.runtime_config.enhanced_debug else 'disabled'}."
        )

    def toggle_rich_text(self):
        """Toggles rich text rendering."""
        self.runtime_config.rich_text = not self.runtime_config.rich_text
        self.client.display.rich_text_mode = self.runtime_config.rich_text
        self.pt_printer(
            f"✅ Rich text output {'enabled' if self.runtime_config.rich_text else 'disabled'}."
        )

    def toggle_smooth_stream(self):
        """Toggles smooth streaming mode."""
        self.runtime_config.smooth_stream = not self.runtime_config.smooth_stream
        self.client.display.smooth_stream_mode = self.runtime_config.smooth_stream
        self.pt_printer(
            f"✅ Smooth streaming {'enabled' if self.runtime_config.smooth_stream else 'disabled'}."
        )

    def toggle_tools(self):
        """Toggles tool usage for the session."""
        if not self.runtime_config.tools:
            self.pt_printer(
                "❌ To use tools, please restart and add the `--tools` flag."
            )
            return

        self.client.tools_enabled = not self.client.tools_enabled
        self.pt_printer(
            f"✅ Tool calling {'enabled' if self.client.tools_enabled else 'disabled'}."
        )

    async def run(self):
        """Starts the interactive UI."""
        self.pt_printer(
            f"🎯 {self._get_mode_display_name()} Mode | Endpoint: {self.client.config.name} | Model: {self.client.config.model_name}"
        )
        self.pt_printer(f"💾 Session logs will be saved to: {self.log_dir}")
        self.pt_printer("Type '/help' for commands, '/quit' to exit.")
        self.pt_printer("-" * 60)

        # The reason for the session ending is tracked so the manifest can be saved correctly.
        finish_reason = "crashed"
        try:
            result = await self.app.run_async()
            if isinstance(result, EOFError):
                finish_reason = "eof"
            else:
                finish_reason = "quit"
        except Exception:
            # The exception will be caught and handled by main(), but we set the
            # reason here so the 'finally' block can log it correctly.
            finish_reason = "error"
            raise  # Re-raise to allow main() to handle the exit.
        finally:
            self.save_log_manifest(finish_reason)
            closing(self.client.stats, printer=self.pt_printer)


# NEW: typer application replaces main(), async_main(), and argparse
app = typer.Typer(
    name="pai",
    help="🪶 Polyglot AI: A Universal CLI for the OpenAI API Format 🪶",
    add_completion=False,
    no_args_is_help=True,
)


def get_version_string() -> str:
    """Gets the version of the PAI package, or a dev string."""
    try:
        # This will work when the package is installed
        return importlib.metadata.version("pai")
    except importlib.metadata.PackageNotFoundError:
        # Fallback for when running from source without installation
        return f"dev-{datetime.now().strftime('%Y%m%d')}"


def _merge_configs(base: dict, new: dict) -> dict:
    """Performs a deep merge of two configuration dictionaries."""
    merged = deepcopy(base)

    # Simple key updates (last one wins)
    if "custom-pricing-file" in new:
        merged["custom-pricing-file"] = new["custom-pricing-file"]

    # Dict updates for profiles and arenas
    for key in ["profiles", "arenas"]:
        if key in new:
            if key not in merged or not isinstance(merged.get(key), dict):
                merged[key] = {}
            merged[key].update(new[key])

    # Tool config merge (list concatenation, no duplicates)
    if new_tool_config := new.get("tool_config"):
        if "directories" in new_tool_config:
            if "tool_config" not in merged:
                merged["tool_config"] = {"directories": []}
            elif "directories" not in merged["tool_config"]:
                merged["tool_config"]["directories"] = []

            base_dirs = merged["tool_config"]["directories"]
            for d in new_tool_config["directories"]:
                if d not in base_dirs:
                    base_dirs.append(d)

    # Endpoints merge (by name)
    if "endpoints" in new and new["endpoints"]:
        if "endpoints" not in merged:
            merged["endpoints"] = []

        base_endpoints = {e["name"]: e for e in merged["endpoints"]}
        new_endpoints = {e["name"]: e for e in new["endpoints"]}

        base_endpoints.update(new_endpoints)
        merged["endpoints"] = list(base_endpoints.values())

    return merged


def load_toml_config(path: str) -> PolyglotConfig:
    """Loads a base TOML config and merges any configs from the providers/ dir."""
    try:
        with open(path, "rb") as f:
            base_data = tomllib.load(f)
    except FileNotFoundError:
        # Provide a helpful message if the old config file name is found.
        if path == "pai.toml" and pathlib.Path("polyglot.toml").exists():
            sys.exit(
                "❌ FATAL: Config file 'polyglot.toml' found. Please rename it to 'pai.toml'."
            )
        sys.exit(f"❌ FATAL: Config file not found at '{path}'")
    except Exception as e:
        sys.exit(f"❌ FATAL: Could not parse '{path}': {e}")

    # Now merge provider configs from a 'providers' directory
    providers_dir = pathlib.Path("providers")
    merged_data = base_data

    if providers_dir.is_dir():
        provider_files = sorted(providers_dir.glob("*.toml"))
        if provider_files:
            typer.echo(
                f"🔎 Merging {len(provider_files)} provider configs from '{providers_dir}'..."
            )

        for provider_file in provider_files:
            try:
                with open(provider_file, "rb") as f:
                    provider_data = tomllib.load(f)
                    merged_data = _merge_configs(merged_data, provider_data)
            except Exception as e:
                typer.echo(
                    f"⚠️  Warning: Could not load or merge '{provider_file}': {e}",
                    err=True,
                )

    try:
        return PolyglotConfig.model_validate(merged_data)
    except Exception as e:
        sys.exit(f"❌ FATAL: Error in final merged config: {e}")


async def _run_batch_mode(client: "PolyglotClient", runtime_config: RuntimeConfig) -> None:
    """Run batch mode processing for multiple prompts.

    Reads prompts from a file, processes each one, and optionally writes results to a file.
    Supports text, JSON, and YAML input formats.
    """
    from datetime import datetime

    batch_file = runtime_config.batch_file
    if not batch_file:
        return

    typer.echo(f"📦 Batch Mode: Loading prompts from {batch_file}")

    try:
        prompts = _load_batch_prompts(batch_file)
    except FileNotFoundError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)

    if not prompts:
        typer.echo("⚠️  No prompts found in batch file.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"📝 Found {len(prompts)} prompt(s) to process")
    typer.echo(f"🔧 Using model: {client.config.model_name}")
    typer.echo("-" * 60)

    results = []
    start_time = datetime.now()

    for i, prompt_text in enumerate(prompts, 1):
        typer.echo(f"\n[{i}/{len(prompts)}] Processing prompt...")

        # Build messages
        messages = (
            [{"role": "system", "content": runtime_config.system}]
            if runtime_config.system
            else []
        )
        messages.append({"role": "user", "content": prompt_text})

        # Create request
        request = ChatRequest(
            messages=messages,
            model=client.config.model_name,
            max_tokens=runtime_config.max_tokens,
            temperature=runtime_config.temperature,
            stream=False,  # Disable streaming for batch mode
            tools=get_tool_schemas() if client.tools_enabled else [],
        )

        try:
            # Generate response
            response = await client.generate(request, runtime_config.verbose)

            result = {
                "index": i,
                "prompt": prompt_text,
                "response": response if response else "",
                "model": client.config.model_name,
                "success": True,
                "stats": {
                    "tokens_sent": client.stats.last_request_stats.tokens_sent if client.stats.last_request_stats else 0,
                    "tokens_received": client.stats.last_request_stats.tokens_received if client.stats.last_request_stats else 0,
                    "cost": client.stats.last_request_stats.cost.total_cost if client.stats.last_request_stats and client.stats.last_request_stats.cost else 0,
                },
            }
            typer.echo(f"   ✅ Completed ({result['stats']['tokens_received']} tokens)")

        except Exception as e:
            result = {
                "index": i,
                "prompt": prompt_text,
                "response": None,
                "model": client.config.model_name,
                "success": False,
                "error": str(e),
            }
            typer.echo(f"   ❌ Failed: {e}")

        results.append(result)

    # Summary
    elapsed = datetime.now() - start_time
    successful = sum(1 for r in results if r["success"])
    typer.echo("\n" + "=" * 60)
    typer.echo(f"📊 Batch Complete: {successful}/{len(prompts)} successful")
    typer.echo(f"⏱️  Total time: {elapsed.total_seconds():.1f}s")
    typer.echo(f"💰 Total cost: ${client.stats.total_cost:.5f}")

    # Write output file if specified
    if runtime_config.output_file:
        output_path = pathlib.Path(runtime_config.output_file)
        output_data = {
            "batch_file": batch_file,
            "model": client.config.model_name,
            "timestamp": datetime.now().isoformat(),
            "total_prompts": len(prompts),
            "successful": successful,
            "failed": len(prompts) - successful,
            "elapsed_seconds": elapsed.total_seconds(),
            "total_cost": client.stats.total_cost,
            "results": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        typer.echo(f"📄 Results written to: {output_path}")


async def _run(runtime_config: RuntimeConfig, toml_config: PolyglotConfig):
    """The core async logic of the application."""
    if runtime_config.log_file:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            filename=runtime_config.log_file,
            filemode="w",
        )
        logging.info("--- Log file initialized ---")

    # Load protocol adapters and custom tools from config
    load_protocol_adapters(printer=typer.echo)
    # Conditionally load tools only if the --tools flag is active.
    if runtime_config.tools:
        typer.echo("🛠️  --tools flag detected. Loading tools...")
        if toml_config.tool_config:
            if tool_dirs := toml_config.tool_config.directories:
                from .tools import load_tools_from_directory

                for tool_dir in tool_dirs:
                    load_tools_from_directory(tool_dir, printer=typer.echo)
        else:
            typer.echo("  (No 'tool_config' section in pai.toml)")

        # Initialize MCP if configured
        if toml_config.mcp and toml_config.mcp.enabled:
            typer.echo("🔌 MCP support enabled. Loading MCP servers...")
            from .mcp import MCPConfig, MCPManager, MCPServerConfig

            # Convert TomlMCPConfig to MCPConfig
            mcp_servers = []
            for name, server_toml in toml_config.mcp.servers.items():
                mcp_servers.append(
                    MCPServerConfig(
                        name=name,
                        command=server_toml.command,
                        env=server_toml.env,
                        enabled=server_toml.enabled,
                        auto_connect=server_toml.auto_connect,
                        timeout=server_toml.timeout,
                    )
                )
            mcp_config = MCPConfig(enabled=True, servers=mcp_servers)

            # Create and set the global MCP manager
            mcp_manager = MCPManager()
            mcp_manager.load_config(mcp_config)
            set_mcp_manager(mcp_manager)

            typer.echo(f"  📦 Configured {len(mcp_servers)} MCP server(s)")
            if mcp_servers:
                typer.echo("  Use /mcp status to see server status")
                typer.echo("  MCP servers will auto-connect when tools are requested")

    # Validate arena configurations
    for arena_name, arena_config in toml_config.arenas.items():
        for p_id, participant in arena_config.participants.items():
            if not any(ep.name == participant.endpoint for ep in toml_config.endpoints):
                typer.echo(
                    f"❌ FATAL: Arena '{arena_name}' participant '{p_id}' references non-existent endpoint '{participant.endpoint}'.",
                    err=True,
                )
                raise typer.Exit(code=1)
        if arena_config.judge:
            if not any(
                ep.name == arena_config.judge.endpoint for ep in toml_config.endpoints
            ):
                typer.echo(
                    f"❌ FATAL: Arena '{arena_name}' judge references non-existent endpoint '{arena_config.judge.endpoint}'.",
                    err=True,
                )
                raise typer.Exit(code=1)

    # Use a single httpx client session for the application's lifecycle
    transport = httpx.AsyncHTTPTransport(retries=3)
    async with httpx.AsyncClient(transport=transport, timeout=30.0) as http_session:
        pricing_service = PricingService()
        # The CLI flag takes precedence over the setting in the config file.
        custom_pricing_path = (
            runtime_config.custom_pricing_file or toml_config.custom_pricing_file
        )
        await pricing_service.load_pricing_data(custom_file_path=custom_pricing_path)

        try:
            version_str = get_version_string()
            client = PolyglotClient(
                runtime_config,
                toml_config,
                http_session,
                pricing_service,
                version=version_str,
            )
            if runtime_config.batch_file:
                # Batch mode - process multiple prompts from file
                await _run_batch_mode(client, runtime_config)
            elif runtime_config.prompt:
                # Non-interactive mode - single prompt
                if runtime_config.chat:
                    messages = (
                        [{"role": "system", "content": runtime_config.system}]
                        if runtime_config.system
                        else []
                    )
                    messages.append({"role": "user", "content": runtime_config.prompt})
                    request = ChatRequest(
                        messages=messages,
                        model=client.config.model_name,
                        max_tokens=runtime_config.max_tokens,
                        temperature=runtime_config.temperature,
                        stream=runtime_config.stream,
                        tools=get_tool_schemas() if client.tools_enabled else [],
                    )
                else:
                    request = CompletionRequest(
                        prompt=runtime_config.prompt,
                        model=client.config.model_name,
                        max_tokens=runtime_config.max_tokens,
                        temperature=runtime_config.temperature,
                        stream=runtime_config.stream,
                    )
                await client.generate(request, runtime_config.verbose)
                print_stats(client.stats)
            else:
                # Interactive mode
                from .tools import get_mcp_manager

                mcp_manager = get_mcp_manager()

                # Auto-connect MCP servers if tools are enabled
                if mcp_manager and runtime_config.tools:
                    await mcp_manager.connect_all(printer=typer.echo)

                try:
                    ui = InteractiveUI(client, runtime_config)
                    await ui.run()
                finally:
                    # Clean up MCP connections on exit
                    if mcp_manager:
                        await mcp_manager.disconnect_all()
        except (APIError, ValueError, httpx.RequestError, ConnectionError) as e:
            # Typer/rich will print a nice error message, no need to print it ourselves.
            raise typer.Exit(code=1) from e


@app.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)
def run(
    ctx: typer.Context,
    profile: str | None = typer.Option(
        None, "--profile", help="Use a named profile from the config file."
    ),
    prompt: str | None = typer.Option(
        None, "-p", "--prompt", help="Send a single prompt and exit."
    ),
    chat: bool = typer.Option(True, help="Enable chat mode. Required for tool use."),
    system: str | None = typer.Option(None, help="Set a system prompt for chat mode."),
    model: str | None = typer.Option(
        None, help="Override the default model for the session."
    ),
    endpoint: str = typer.Option(
        "openai", help="The name of the endpoint from the config file to use."
    ),
    max_tokens: int = typer.Option(2000, help="Set the max tokens for the response."),
    temperature: float = typer.Option(
        0.7, help="Set the temperature for the response."
    ),
    timeout: int | None = typer.Option(
        None, help="Set the request timeout in seconds for the session."
    ),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Enable/disable streaming."
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging of request parameters."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable raw protocol debug mode for streaming."
    ),
    enhanced_debug: bool = typer.Option(
        False,
        "--enhanced-debug",
        help="Enable enhanced, diff-based debug mode. Implies --debug.",
    ),
    tools: bool = typer.Option(
        False,
        "--tools",
        help="Enable tool-use capabilities by loading tools from directories in config.",
    ),
    rich_text: bool = typer.Option(
        True, "--rich/--no-rich", help="Enable/disable rich text formatting for output."
    ),
    confirm_tool_use: bool = typer.Option(
        False,
        "--confirm",
        help="Require user confirmation before executing a tool.",
    ),
    keep_reasoning: bool = typer.Option(
        True,
        "--keep-reasoning/--no-keep-reasoning",
        help="Include model's reasoning in the assistant's message history.",
    ),
    smooth_stream: bool = typer.Option(
        True,
        "--smooth-stream/--no-smooth-stream",
        help="Enable/disable smoothed streaming output. On by default.",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Path to a file for writing debug and verbose logs.",
        show_default=False,
    ),
    config: str = typer.Option("pai.toml", help="Path to the TOML configuration file."),
    custom_pricing_file: str | None = typer.Option(
        None,
        "--custom-pricing-file",
        help="Path to a custom TOML pricing file. Overrides 'custom-pricing-file' in config.",
        show_default=False,
    ),
    batch_file: str | None = typer.Option(
        None,
        "--batch",
        "-b",
        help="Path to a file containing prompts (one per line or JSON array). Runs non-interactively.",
        show_default=False,
    ),
    output_file: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for batch mode results (JSON format).",
        show_default=False,
    ),
    env_file: str | None = typer.Option(
        None,
        "--env-file",
        help="Path to a .env file to load environment variables from.",
        show_default=False,
    ),
):
    """Main application entrypoint."""
    print("🪶 Polyglot AI: A Universal CLI for Any AI Provider 🪶")

    # Load environment variables from .env file if specified or auto-detect
    _load_env_file(env_file)

    toml_config = load_toml_config(config)

    # Handle profile loading. This must happen before RuntimeConfig is instantiated
    # so that the context can provide the correct default values.
    if profile:
        profile_settings = toml_config.profiles.get(profile)
        if not profile_settings:
            typer.echo(
                f"❌ Error: Profile '{profile}' not found in '{config}'.", err=True
            )
            raise typer.Exit(code=1)

        # Let Typer use the profile's values as defaults for any unspecified CLI options.
        # `exclude_unset=True` ensures we only use values defined in the profile.
        ctx.default_map = profile_settings.model_dump(exclude_unset=True)
        typer.echo(f"✅ Loaded profile '{profile}'.")

    runtime_config = RuntimeConfig(
        profile=profile,
        prompt=prompt,
        chat=chat,
        system=system,
        model=model,
        endpoint=endpoint,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        stream=stream,
        verbose=verbose,
        debug=debug,
        enhanced_debug=enhanced_debug,
        tools=tools,
        rich_text=rich_text,
        confirm_tool_use=confirm_tool_use,
        keep_reasoning=keep_reasoning,
        smooth_stream=smooth_stream,
        log_file=log_file,
        config=config,
        custom_pricing_file=custom_pricing_file,
        batch_file=batch_file,
        output_file=output_file,
    )
    asyncio.run(_run(runtime_config, toml_config))


def main():
    try:
        app()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
