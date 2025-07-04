"""
Polyglot AI: An Interactive, Multi-Provider CLI for the OpenAI API Format.
This version is a direct refactoring of the original code, preserving all features
and fixing the circular import error.
"""

import asyncio
import importlib.metadata
import json
import logging
import pathlib
import re
import sys
from datetime import datetime
from html import escape

import httpx
import toml
import typer
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
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import SearchToolbar

from .client import APIError, PolyglotClient
from .commands import CommandHandler
from .log_utils import closing, print_stats
from .models import (
    ChatRequest,
    CompletionRequest,
    Conversation,
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
from .tools import get_tool_schemas

# --- Global Definitions ---
session = PromptSession()


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

    def __init__(self, client: "PolyglotClient", runtime_config: RuntimeConfig):
        self.client = client
        self.runtime_config = runtime_config

        # State management (replaces is_chat_mode, native_agent_mode, etc.)
        initial_mode = UIMode.CHAT if runtime_config.chat else UIMode.COMPLETION
        self.state = UIState(mode=initial_mode)

        self.conversation = Conversation()
        if self.state.mode != UIMode.COMPLETION and self.runtime_config.system:
            self.conversation.set_system_prompt(self.runtime_config.system)

        # Setup directories
        self.log_dir = pathlib.Path("logs") / datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S-interactive"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
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
        self.client.display.set_printer(self.pt_printer, is_interactive=True)
        self.client.display.ui = self  # Give display access to UI state
        self.history = FileHistory(str(pai_user_dir / "history.txt"))

        # Command handler must be created before the input buffer that uses it.
        self.command_handler = CommandHandler(self)
        command_completer = CommandCompleter(self.command_handler.completion_list)

        self.streaming_output_buffer = Buffer()
        self.client.display.output_buffer = self.streaming_output_buffer

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
                        re.sub("<[^<]+?>", "", self._get_prompt_text().value)
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

        live_output_window = ConditionalContainer(
            Window(
                content=BufferControl(buffer=self.streaming_output_buffer),
                wrap_lines=True,
            ),
            filter=Condition(
                lambda: self.generation_in_progress.is_set()
                and self.streaming_output_buffer.text
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
                    live_output_window,
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
            # It's a prompt, so dispatch to an orchestrator.
            orchestrator = self._get_orchestrator()
            if orchestrator:
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
        json_args = json.dumps(args, indent=2)
        self.pt_printer(
            HTML(
                f"\n<style fg='ansimagenta' bg='ansiblack'>🔧 Agent wants to execute: <b>{escape(tool_name)}</b></style>"
            )
        )
        self.pt_printer(
            HTML(
                f"<style fg='ansimagenta'>   with arguments:\n{escape(json_args)}</style>"
            )
        )

        try:
            # This modal-like prompt will temporarily take over the input line
            result = await self.confirm_session.prompt_async(
                "Authorize this tool call? [y/N]: ",
            )
            return result.lower().strip() == "y"
        except (EOFError, KeyboardInterrupt):
            return False

    def _get_toolbar_line1_text(self) -> str:
        """Generates the first line of the toolbar (context)."""
        prompt_count = len(self.conversation.get_system_prompts())
        mode_str = escape(self._get_mode_display_name())
        if prompt_count > 1:
            mode_str += f" <style fg='ansicyan'>Sys[{prompt_count}]</style>"

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
        total_time = session_stats.total_response_time
        total_received = session_stats.total_tokens_received
        total_cost = session_stats.total_cost

        if live_stats and display.status in ["Waiting...", "Streaming"]:
            session_tokens += live_stats.tokens_sent + live_stats.tokens_received
            total_time += live_stats.current_duration
            total_received += live_stats.tokens_received
            if live_stats.cost:
                total_cost += live_stats.cost.total_cost

        session_tps = total_received / max(total_time, 1)

        parts.extend(
            [
                f"<b>Cost:</b> ${total_cost:.4f}",
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
        agent_toggles = [
            f"Tools: {on if self.client.tools_enabled else off}",
            f"Confirm: {yellow_on if self.runtime_config.confirm_tool_use else off}",
            f"Debug: {yellow_on if display.debug_mode else off}",
            f"Verbose: {yellow_on if self.runtime_config.verbose else off}",
        ]
        log_part = f"Log: {escape(str(self.log_dir))}"
        agent_part = f"<b>Agent Toggles</b> | {' | '.join(agent_toggles)}"
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
                f"<style bg='ansired' fg='white'>[Toolbar Error: {escape(str(e))}]</style>"
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

        # This try/finally ensures that closing stats are printed even if the app
        # exits with an exception (e.g., Ctrl+C/Ctrl+D). The exceptions are
        # allowed to propagate up to main() where they are handled for a clean exit.
        try:
            # run_async() returns the value from app.exit(). The default Ctrl+D
            # handler passes an EOFError. We handle it here to prevent it from
            # propagating to Typer, which would cause an "Aborted." message.
            result = await self.app.run_async()
            if isinstance(result, EOFError):
                return  # This triggers the finally block and exits cleanly.
        finally:
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


def load_toml_config(path: str) -> PolyglotConfig:
    """Loads and validates the TOML configuration file."""
    try:
        with open(path, encoding="utf-8") as f:
            data = toml.load(f)
            return PolyglotConfig.model_validate(data)
    except FileNotFoundError:
        # Provide a helpful message if the old config file name is found.
        if path == "pai.toml" and pathlib.Path("polyglot.toml").exists():
            sys.exit(
                "❌ FATAL: Config file 'polyglot.toml' found. Please rename it to 'pai.toml'."
            )
        sys.exit(f"❌ FATAL: Config file not found at '{path}'")
    except Exception as e:
        sys.exit(f"❌ FATAL: Could not parse '{path}': {e}")


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
            if runtime_config.prompt:
                # Non-interactive mode
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
                ui = InteractiveUI(client, runtime_config)
                await ui.run()
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
    chat: bool = typer.Option(False, help="Enable chat mode. Required for tool use."),
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
):
    """Main application entrypoint."""
    print("🪶 Polyglot AI: A Universal CLI for Any AI Provider 🪶")

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
        tools=tools,
        rich_text=rich_text,
        confirm_tool_use=confirm_tool_use,
        smooth_stream=smooth_stream,
        log_file=log_file,
        config=config,
        custom_pricing_file=custom_pricing_file,
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
