"""
Polyglot AI: An Interactive, Multi-Provider CLI for the OpenAI API Format.
This version is a direct refactoring of the original code, preserving all features
and fixing the circular import error.
"""

import os
import sys
import json
import time
import argparse
import asyncio
import httpx
import ulid
import pathlib
from html import escape
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.widgets import SearchToolbar
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory

import toml

from .utils import estimate_tokens
from .log_utils import print_stats, closing, save_conversation_formats

from .models import (
    Turn,
    Conversation,
    RequestStats,
    TestSession,
    EndpointConfig,
    CompletionRequest,
    ChatRequest,
)

# --- Protocol Adapter Imports ---
from .protocols.base_adapter import BaseProtocolAdapter, ProtocolContext
from .protocols.openai_chat_adapter import OpenAIChatAdapter
from .protocols.legacy_completion_adapter import LegacyCompletionAdapter

# --- Global Definitions ---
session = PromptSession()
ADAPTER_MAP = {
    "openai_chat": OpenAIChatAdapter(),
    "legacy_completion": LegacyCompletionAdapter(),
}


# --- Data Models ---
# All data classes for state, history, and requests have been moved to `pai/models.py`
# to improve modularity and break circular dependencies.


# --- [REFACTORED] StreamingDisplay for Stable UI ---
class StreamingDisplay:
    """Manages all console output, ensuring prompt-toolkit UI is not corrupted."""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self._printer = print  # Default to standard print
        self._is_interactive = False
        self.output_buffer: Optional[Buffer] = None

        # State for UI
        self.status = "Idle"
        # Holds the stats for the *in-progress* request.
        self.current_request_stats: Optional[RequestStats] = None
        self._last_token_time: Optional[float] = None

        # Response tracking
        self.line_count: int = 0
        self.chunk_count: int = 0
        self.current_response: str = ""
        self.first_token_received = False

    def set_printer(self, printer: callable, is_interactive: bool):
        """Sets the function used for printing to the console."""
        self._printer = printer
        self._is_interactive = is_interactive

    def start_response(self):
        """Prepares for a new response stream."""
        self.current_response = ""
        if self.output_buffer:
            self.output_buffer.reset()

        self.current_request_stats = RequestStats()
        self._last_token_time = None
        self.line_count = 0
        self.chunk_count = 0
        self.first_token_received = False
        self.status = "Waiting..."

    @property
    def time_since_last_token(self) -> float:
        """Returns seconds since the last token was received, for spinner UI."""
        if self.status == "Streaming" and self._last_token_time:
            return time.time() - self._last_token_time
        return 0.0

    def _print(self, *args, **kwargs):
        """Internal print-router."""
        # In interactive mode, self._printer is prompt_toolkit's thread-safe
        # print_formatted_text function, which handles redrawing the prompt.
        self._printer(*args, **kwargs)

    def show_raw_line(self, line: str):
        if self.debug_mode:
            if not self.first_token_received:
                self._print("🔍 DEBUG MODE: Showing raw protocol traffic\n" + "=" * 60)
                self.first_token_received = (
                    True  # Prevent header from printing multiple times
                )
            self.line_count += 1
            duration = (
                self.current_request_stats.current_duration
                if self.current_request_stats
                else 0
            )
            prefix = f"⚪ [{duration:6.2f}s] L{self.line_count:03d}: "
            if line.startswith("data: "):
                prefix = f"🔵 [{duration:6.2f}s] L{self.line_count:03d}: "
            self._print(f"{prefix}{repr(line)}")

    def show_parsed_chunk(self, chunk_data: Dict, chunk_text: str):
        """Handles printing a parsed chunk of text from the stream."""
        # Don't do anything for empty chunks, which some providers send.
        if not chunk_text:
            return

        self._last_token_time = time.time()

        # Update state first
        if not self.first_token_received:
            self.status = "Streaming"
            if self.current_request_stats:
                self.current_request_stats.record_first_token()
            self.first_token_received = True

        self.current_response += chunk_text
        self.chunk_count += 1
        if self.current_request_stats:
            # To ensure the live count is consistent with the final count,
            # we recalculate from the full response string on every chunk.
            self.current_request_stats.tokens_received = estimate_tokens(
                self.current_response
            )

        # Handle rendering
        if self._is_interactive and self.output_buffer:
            # For interactive mode with a buffer, update the buffer's content.
            # This will be displayed live in the UI's live output window.
            self.output_buffer.text = f"🤖 Assistant: {self.current_response}"
        elif not self._is_interactive:
            # For non-interactive, print header once, then stream chunks.
            if self.chunk_count == 1:
                self._print("\n🤖 Assistant: ", end="")
            self._print(chunk_text, end="", flush=True)

        if self.debug_mode:
            duration = (
                self.current_request_stats.current_duration
                if self.current_request_stats
                else 0
            )
            self._print(
                f"🟢 [{duration:6.2f}s] C{self.chunk_count:03d} TEXT: {repr(chunk_text)}"
            )

    def finish_response(self, success: bool = True) -> Optional[RequestStats]:
        """Finalizes the response, prints stats, and resets the display state."""
        self.status = "Done"
        if self.current_request_stats:
            self.current_request_stats.finish(success=success)
            # The token count is now calculated live. This final assignment
            # ensures it's perfectly accurate based on the complete final string,
            # but the live value should already match this.
            self.current_request_stats.tokens_received = estimate_tokens(
                self.current_response
            )

        # In interactive mode, if we have a response, print it to the scrollback
        # history. This "finalizes" it, moving it from the temporary live
        # buffer to the main conversation transcript. This happens regardless
        # of success to ensure partial/cancelled outputs are preserved.
        if self._is_interactive and self.current_response:
            self._print(HTML(f"🤖 Assistant: {self.current_response}"))

        # On success, print final stats.
        if success and self.current_request_stats:
            stats = self.current_request_stats
            if self.debug_mode:
                self._print(
                    "=" * 60
                    + f"\n🔍 DEBUG SUMMARY: {self.line_count} lines, {self.chunk_count} chunks, {stats.response_time:.2f}s\n"
                    + "=" * 60
                )
            elif not self._is_interactive and self.first_token_received:
                # For non-interactive mode, print the final stats line.
                tok_per_sec = stats.final_tok_per_sec
                ttft_str = f" | TTFT: {stats.ttft:.2f}s" if stats.ttft is not None else ""
                self._print(
                    f"\n\n📊 Response in {stats.response_time:.2f}s ({stats.tokens_received} tokens, {tok_per_sec:.1f} tok/s{ttft_str})"
                )

        # Always clear the live buffer and reset state for the next command.
        if self.output_buffer:
            self.output_buffer.reset()
        self.status = "Idle"
        return self.current_request_stats


class APIError(Exception):
    pass


class PolyglotClient:
    def __init__(
        self,
        args: argparse.Namespace,
        loaded_config: Dict,
        http_session: httpx.AsyncClient,
    ):
        self.all_endpoints = loaded_config.get("endpoints", [])
        self.config = EndpointConfig()
        self.stats = TestSession()
        self.display = StreamingDisplay(args.debug)
        self.http_session = http_session
        self.tools_enabled = args.tools
        self.switch_endpoint(args.endpoint)
        if self.config and args.model:
            self.config.model_name = args.model

    def switch_endpoint(self, name: str):
        endpoint_data = next(
            (e for e in self.all_endpoints if e["name"].lower() == name.lower()), None
        )
        if not endpoint_data:
            self.display._print(
                f"❌ Error: Endpoint '{name}' not found in configuration file."
            )
            return
        api_key = os.getenv(endpoint_data["api_key_env"])
        if not api_key:
            self.display._print(
                f"❌ Error: API key for '{name}' not found. Set {endpoint_data['api_key_env']}."
            )
            return
        self.config.name = endpoint_data["name"]
        self.config.base_url = endpoint_data["base_url"]
        self.config.api_key = api_key
        self.config.chat_adapter = ADAPTER_MAP.get(endpoint_data.get("chat_adapter"))
        self.config.completion_adapter = ADAPTER_MAP.get(
            endpoint_data.get("completion_adapter")
        )
        # Configure the httpx client for the selected endpoint
        self.http_session.base_url = self.config.base_url
        self.http_session.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PolyglotAI/0.1.0",
        }
        self.display._print(f"✅ Switched to endpoint: {self.config.name}")

    async def generate(
        self, request: Union[CompletionRequest, ChatRequest], verbose: bool
    ) -> Dict[str, Any]:
        is_chat = isinstance(request, ChatRequest)
        adapter = (
            self.config.chat_adapter if is_chat else self.config.completion_adapter
        )
        if not adapter:
            raise APIError(
                f"Endpoint '{self.config.name}' does not support {'chat' if is_chat else 'completion'} mode."
            )
        if verbose:
            self.display._print(
                f"\n🚀 Sending request via endpoint '{self.config.name}' using adapter '{type(adapter).__name__}'"
            )
            self.display._print(f"📝 Model: {self.config.model_name}")
            self.display._print(
                f"🎛️ Parameters: temp={request.temperature}, max_tokens={request.max_tokens}, stream={request.stream}"
            )

        context = ProtocolContext(
            http_session=self.http_session,
            display=self.display,
            stats=self.stats,
            config=self.config,
            tools_enabled=self.tools_enabled,
        )
        return await adapter.generate(context, request, verbose)


def print_banner():
    print("🪶 Polyglot AI: A Universal CLI for the OpenAI API Format 🪶")


# Logging and statistics functions have been moved to `pai/log_utils.py`.


class InteractiveUI:
    """Encapsulates all logic for the text user interface."""

    def __init__(self, client: "PolyglotClient", args: argparse.Namespace):
        self.client = client
        self.args = args
        self.is_chat_mode = args.chat
        self.conversation = Conversation()
        if self.is_chat_mode and self.args.system:
            self.conversation.set_system_prompt(self.args.system)

        # Setup directories
        self.session_dir = pathlib.Path("sessions") / datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S-interactive"
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        pai_user_dir = pathlib.Path.home() / ".pai"
        pai_user_dir.mkdir(exist_ok=True)
        self.prompts_dir = pathlib.Path("prompts")
        self.prompts_dir.mkdir(exist_ok=True)

        # Setup UI components
        self.pt_printer = print_formatted_text
        self.client.display.set_printer(self.pt_printer, is_interactive=True)
        self.history = FileHistory(str(pai_user_dir / "history.txt"))
        self.streaming_output_buffer = Buffer()
        self.client.display.output_buffer = self.streaming_output_buffer

        self.input_buffer = Buffer(
            name="input_buffer",
            multiline=False,
            history=self.history,
            enable_history_search=True,
            accept_handler=self._on_buffer_accepted,
        )

        # State management
        self.generation_in_progress = asyncio.Event()
        self.generation_task: Optional[asyncio.Task] = None
        self.spinner_chars = ["|", "/", "-", "\\"]
        self.spinner_idx = 0

        # Build the application
        self.app = self._create_application()

    def _create_application(self) -> Application:
        """Constructs the prompt_toolkit Application object."""
        # This is the main input bar at the bottom of the screen.
        prompt_ui = VSplit(
            [
                Window(
                    FormattedTextControl(
                        lambda: HTML(
                            f"<style fg='ansigreen'>👤 ({self.client.config.name}) User:</style> "
                        )
                    ),
                    width=lambda: len(f"👤 ({self.client.config.name}) User: ") + 1,
                ),
                Window(BufferControl(buffer=self.input_buffer)),
            ]
        )

        def get_status_text():
            display = self.client.display
            if display.first_token_received:
                # Spinner logic
                spinner = self.spinner_chars[self.spinner_idx]
                self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)

                time_since_last = display.time_since_last_token
                color = "ansigreen"
                if time_since_last > 2.0:  # Becomes yellow after 2 seconds
                    color = "ansiyellow"
                if time_since_last > 5.0:  # Becomes red after 5 seconds
                    color = "ansired"
                return HTML(
                    f"<style fg='{color}'>[{spinner}] Streaming... ({time_since_last:.1f}s since last token)</style>"
                )
            else:
                return HTML("<style fg='ansiyellow'>[Waiting for response...]</style>")

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
            height=3,
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

        @kb.add("c-c", eager=True)
        def _(event):
            """
            Custom Ctrl+C handler.
            - If a generation is in progress, cancel it.
            - If the input buffer has text, clear it.
            - If the input buffer is empty, print a fake prompt to emulate a new line.
            """
            if self.generation_in_progress.is_set() and self.generation_task:
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
                            f"<style fg='ansigreen'>👤 ({self.client.config.name}) User:</style> "
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
        if self.generation_in_progress.is_set():
            return

        user_input = buffer.text
        # Only process if the input is not just whitespace
        if user_input.strip():
            # Store the raw string to history and echo it to the user.
            self.history.store_string(user_input)
            self.pt_printer(
                HTML(
                    f"\n<style fg='ansigreen'>👤 ({self.client.config.name}) User:</style> {user_input}"
                )
            )
            buffer.reset()

            stripped_input = user_input.strip()
            if stripped_input.startswith("/"):
                # Because the app is created after the buffer, we can safely
                # reference self.app here, as it will exist when this handler is called.
                self._handle_command(stripped_input, self.app)
            else:
                self.generation_task = asyncio.create_task(
                    self._process_and_generate(stripped_input)
                )
        else:
            # On empty or whitespace-only input, we clear the buffer and
            # print a "fake" prompt to give the user feedback of a new line.
            buffer.reset()
            self.pt_printer(
                HTML(
                    f"<style fg='ansigreen'>👤 ({self.client.config.name}) User:</style> "
                )
            )

    def _handle_command(self, text: str, app: Application):
        parts = text[1:].lower().split(" ", 1)
        cmd, params = parts[0], parts[1] if len(parts) > 1 else ""

        COMMANDS = {
            "quit": app.exit,
            "exit": app.exit,
            "q": app.exit,
            "stats": lambda: print_stats(self.client.stats, printer=self.pt_printer),
            "help": lambda: self._print_help(),
            "endpoints": self._cmd_endpoints,
            "switch": self._cmd_switch,
            "model": self._cmd_model,
            "temp": self._cmd_temp,
            "tokens": self._cmd_tokens,
            "stream": self._cmd_toggle_stream,
            "verbose": self._cmd_toggle_verbose,
            "debug": self._cmd_toggle_debug,
            "tools": self._cmd_toggle_tools,
            "history": self._cmd_history,
            "clear": self._cmd_clear,
            "system": self._cmd_system,
            "prompts": self._cmd_prompts,
            "prompt": self._cmd_prompt,
            "mode": self._cmd_toggle_mode,
        }

        if cmd in COMMANDS:
            if cmd in ["switch", "model", "temp", "tokens", "system"]:
                if params:
                    COMMANDS[cmd](params)
                else:
                    self.pt_printer(f"❌ Command '/{cmd}' requires a parameter.")
            else:
                COMMANDS[cmd]()
        else:
            self.pt_printer("❌ Unknown command.")

    # --- Command Implementations ---
    def _cmd_endpoints(self):
        self.pt_printer("Available Endpoints:")
        for ep in self.client.all_endpoints:
            self.pt_printer(f" - {ep['name']}")

    def _cmd_switch(self, params: str):
        self.client.switch_endpoint(params)

    def _cmd_model(self, params: str):
        self.client.config.model_name = params
        self.pt_printer(f"✅ Model set to: {params}")

    def _cmd_temp(self, params: str):
        try:
            self.args.temperature = float(params)
            self.pt_printer(f"✅ Temp set to: {self.args.temperature}")
        except ValueError:
            self.pt_printer("❌ Invalid value.")

    def _cmd_tokens(self, params: str):
        try:
            self.args.max_tokens = int(params)
            self.pt_printer(f"✅ Max tokens set to: {self.args.max_tokens}")
        except ValueError:
            self.pt_printer("❌ Invalid value.")

    def _cmd_toggle_stream(self):
        self.args.stream = not self.args.stream
        self.pt_printer(
            f"✅ Streaming {'enabled' if self.args.stream else 'disabled'}."
        )

    def _cmd_toggle_verbose(self):
        self.args.verbose = not self.args.verbose
        self.pt_printer(
            f"✅ Verbose mode {'enabled' if self.args.verbose else 'disabled'}."
        )

    def _cmd_toggle_debug(self):
        self.client.display.debug_mode = not self.client.display.debug_mode
        self.pt_printer(
            f"✅ Debug mode {'enabled' if self.client.display.debug_mode else 'disabled'}."
        )

    def _cmd_toggle_tools(self):
        self.client.tools_enabled = not self.client.tools_enabled
        self.pt_printer(
            f"✅ Tool calling {'enabled' if self.client.tools_enabled else 'disabled'}."
        )

    def _cmd_history(self):
        if self.is_chat_mode:
            self.pt_printer(json.dumps(self.conversation.get_history(), indent=2))

    def _cmd_clear(self):
        if self.is_chat_mode:
            self.conversation.clear()
            self.pt_printer("🧹 History cleared.")

    def _cmd_system(self, params: str):
        if self.is_chat_mode:
            self.conversation.set_system_prompt(params)
            self.pt_printer(f"🤖 System prompt set.")

    def _cmd_prompts(self):
        """Lists available prompt files."""
        self.pt_printer("Available prompts:")
        found = False
        for p in sorted(self.prompts_dir.glob("*.md")):
            self.pt_printer(f"  - {p.stem}")
            found = True
        for p in sorted(self.prompts_dir.glob("*.txt")):
            self.pt_printer(f"  - {p.stem}")
            found = True
        if not found:
            self.pt_printer("  (None found. Add .md or .txt files to 'prompts/' directory)")

    def _cmd_prompt(self, params: str):
        """Loads a prompt file as the new system prompt."""
        if not self.is_chat_mode:
            self.pt_printer("❌ /prompt command only available in chat mode.")
            return

        prompt_path = self.prompts_dir / f"{params}.md"
        if not prompt_path.exists():
            prompt_path = self.prompts_dir / f"{params}.txt"

        if prompt_path.exists() and prompt_path.is_file():
            content = prompt_path.read_text(encoding="utf-8")
            self.conversation.set_system_prompt(content)
            self.pt_printer(f"🤖 System prompt loaded from '{params}'. History cleared.")
        else:
            self.pt_printer(f"❌ Prompt '{params}' not found in '{self.prompts_dir}'.")

    def _cmd_toggle_mode(self):
        """Toggles between chat and completion mode."""
        self.is_chat_mode = not self.is_chat_mode
        mode_name = 'Chat' if self.is_chat_mode else 'Completion'
        # Switching modes should start a fresh context, so we clear the history.
        self.conversation.clear()
        self.pt_printer(f"✅ Switched to {mode_name} mode. History cleared.")

    async def _process_and_generate(self, user_input_str: str):
        self.generation_in_progress.set()
        try:
            request: Union[ChatRequest, CompletionRequest]
            if self.is_chat_mode:
                messages = self.conversation.get_messages_for_next_turn(user_input_str)
                request = ChatRequest(
                    messages=messages,
                    model=self.client.config.model_name,
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    stream=self.args.stream,
                )
            else:
                request = CompletionRequest(
                    prompt=user_input_str,
                    model=self.client.config.model_name,
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    stream=self.args.stream,
                )

            result = await self.client.generate(request, self.args.verbose)

            if self.is_chat_mode and result:
                turn = Turn(
                    request_data=result.get("request", {}),
                    response_data=result.get("response", {}),
                    assistant_message=result.get("text", ""),
                )
                self.conversation.add_turn(turn)
                try:
                    turn_file = self.session_dir / f"{turn.turn_id}-turn.json"
                    turn_file.write_text(
                        json.dumps(turn.to_dict(), indent=2), encoding="utf-8"
                    )
                    save_conversation_formats(
                        self.conversation, self.session_dir, printer=self.pt_printer
                    )
                except Exception as e:
                    self.pt_printer(f"\n⚠️  Warning: Could not save session turn: {e}")
        except asyncio.CancelledError:
            # When cancelled, we still want to save the partial response.
            # finish_response() sets success=False and returns the incomplete stats.
            request_stats = self.client.display.finish_response(success=False)
            partial_text = self.client.display.current_response

            # Add to session stats.
            if request_stats:
                # This is an approximation of tokens_sent as the final request
                # payload is constructed inside the adapter.
                if isinstance(request, ChatRequest):
                    request_stats.tokens_sent = sum(
                        estimate_tokens(m.get("content", "")) for m in request.messages
                    )
                elif isinstance(request, CompletionRequest):
                    request_stats.tokens_sent = estimate_tokens(request.prompt)
                self.client.stats.add_completed_request(request_stats)

            if self.is_chat_mode and partial_text:
                # Create a turn with the partial data for logging.
                request_data = request.to_dict(self.client.config.model_name)
                # Fabricate a partial response object for logging
                response_data = {
                    "pai_note": "This response was cancelled by the user.",
                    "choices": [
                        {"message": {"role": "assistant", "content": partial_text}}
                    ],
                }
                turn = Turn(
                    request_data=request_data,
                    response_data=response_data,
                    assistant_message=partial_text,
                )
                self.conversation.add_turn(turn)
                try:
                    turn_file = self.session_dir / f"{turn.turn_id}-turn.json"
                    turn_file.write_text(
                        json.dumps(turn.to_dict(), indent=2), encoding="utf-8"
                    )
                    save_conversation_formats(
                        self.conversation, self.session_dir, printer=self.pt_printer
                    )
                except Exception as e:
                    self.pt_printer(
                        f"\n⚠️  Warning: Could not save cancelled session turn: {e}"
                    )

            self.pt_printer(
                HTML("\n<style fg='ansiyellow'>🚫 Generation cancelled.</style>")
            )
        except Exception as e:
            self.client.display.finish_response(success=False)
            self.pt_printer(HTML(f"<style fg='ansired'>❌ ERROR: {e}</style>"))
        finally:
            self.generation_in_progress.clear()
            self.generation_task = None

    def _get_toolbar_text(self) -> HTML:
        """Generates the HTML for the multi-line bottom toolbar."""
        try:
            client, args, session_dir = self.client, self.args, self.session_dir
            endpoint, model = client.config.name, client.config.model_name
            session_stats, display = client.stats, client.display
            live_stats = display.current_request_stats

            # Escape any potentially problematic strings before embedding in HTML
            endpoint_esc = escape(endpoint)
            model_esc = escape(model)
            status_esc = escape(display.status)
            session_dir_esc = escape(str(session_dir))

            # --- Line 1: Overall Session Stats ---
            total_tokens = (
                session_stats.total_tokens_sent + session_stats.total_tokens_received
            )
            total_time = session_stats.total_response_time
            total_received = session_stats.total_tokens_received

            # Add live data during streaming for a real-time view
            if live_stats and display.status == "Streaming":
                total_tokens += live_stats.tokens_sent + live_stats.tokens_received
                total_time += live_stats.current_duration
                total_received += live_stats.tokens_received

            avg_tok_per_sec = total_received / max(total_time, 1)
            line1 = f"<b><style bg='ansiblack' fg='white'> {endpoint_esc.upper()}:{model_esc} </style></b> | <b>Total Tokens:</b> {total_tokens} | <b>Avg Tok/s:</b> {avg_tok_per_sec:.1f}"

            # --- Line 2: Live and Last Request Stats ---
            last_req = session_stats.last_request_stats

            if live_stats and display.status in ["Waiting...", "Streaming"]:
                # Display live stats for the in-progress request
                live_tps = live_stats.live_tok_per_sec
                live_tokens = live_stats.tokens_received
                status_color = (
                    "ansigreen" if display.status == "Streaming" else "ansiyellow"
                )
                live_tps_str = f"<b><style fg='{status_color}'>Live Tokens: {live_tokens}, Live Tok/s: {live_tps:.1f}</style></b>"
                line2_parts = [
                    f"<style fg='ansimagenta'><b>Status: {status_esc}</b></style>",
                    live_tps_str,
                ]
            else:
                # Display stats for the last completed request
                last_tokens_str = f"{last_req.tokens_received}" if last_req else "N/A"
                last_ttft_str = (
                    f"{last_req.ttft:.2f}s" if last_req and last_req.ttft else "N/A"
                )
                last_tps_str = (
                    f"{last_req.final_tok_per_sec:.1f}" if last_req else "N/A"
                )
                last_finish_reason_str = (
                    escape(last_req.finish_reason or "N/A") if last_req else "N/A"
                )
                line2_parts = [
                    f"<style fg='ansimagenta'><b>Status: {status_esc}</b></style>",
                    f"<b>Prev Tokens:</b> {last_tokens_str}",
                    f"<b>Prev TTFT:</b> {last_ttft_str}",
                    f"<b>Prev Tok/s:</b> {last_tps_str}",
                    f"<b>Finish:</b> {last_finish_reason_str}",
                ]

            tools_status = (
                f"<style fg='ansigreen'>ON</style>"
                if client.tools_enabled
                else f"<style fg='ansired'>OFF</style>"
            )
            debug_status = (
                f"<style fg='ansiyellow'>ON</style>" if display.debug_mode else "OFF"
            )
            line3_parts = [
                f"<b>Tools:</b> {tools_status}",
                f"<b>Debug:</b> {debug_status}",
                f"<b>Mode:</b> {'Chat' if self.is_chat_mode else 'Completion'}",
                f"<style fg='grey'>Log: {session_dir_esc}</style>",
            ]

            return HTML(
                f"{line1}\n{' | '.join(p for p in line2_parts if p)}\n{' | '.join(line3_parts)}"
            )
        except Exception as e:
            # If any rendering fails, return a safe, minimal toolbar to prevent crashing.
            return HTML(
                f"<style bg='ansired' fg='white'>[Toolbar Error: {escape(str(e))}]</style>"
            )

    def _print_help(self):
        self.pt_printer("""
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
    """)

    async def run(self):
        """Starts the interactive UI."""
        self.pt_printer(
            f"🎯 {'Chat' if self.is_chat_mode else 'Completion'} Mode | Endpoint: {self.client.config.name} | Model: {self.client.config.model_name}"
        )
        self.pt_printer(f"💾 Session logs will be saved to: {self.session_dir}")
        self.pt_printer("Type '/help' for commands, '/quit' to exit.")
        self.pt_printer("-" * 60)

        # This try/finally ensures that closing stats are printed even if the app
        # exits with an exception (e.g., Ctrl+C/Ctrl+D). The exceptions are
        # allowed to propagate up to main() where they are handled for a clean exit.
        try:
            # run_async() returns the value from app.exit(). The default Ctrl+D
            # handler passes an EOFError. We re-raise it to be caught in main().
            result = await self.app.run_async()
            if isinstance(result, EOFError):
                raise result
        finally:
            closing(self.client.stats, printer=self.pt_printer)


async def interactive_mode(client: PolyglotClient, args: argparse.Namespace):
    """The main interactive mode, which uses a prompt-toolkit Application for a stable UI."""
    ui = InteractiveUI(client, args)
    await ui.run()


async def async_main(args: argparse.Namespace):
    """The async entrypoint for the application."""
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            loaded_config = toml.load(f)
    except FileNotFoundError:
        sys.exit(f"❌ FATAL: Config file not found at '{args.config}'")
    except Exception as e:
        sys.exit(f"❌ FATAL: Could not parse '{args.config}': {e}")

    # Use a single httpx client session for the application's lifecycle
    transport = httpx.AsyncHTTPTransport(retries=3)
    # Default timeout is 5 seconds. Set a longer one for model generation.
    async with httpx.AsyncClient(transport=transport, timeout=30.0) as http_session:
        try:
            client = PolyglotClient(args, loaded_config, http_session)
            if args.prompt:
                # Non-interactive mode
                if args.chat:
                    messages = (
                        [{"role": "system", "content": args.system}]
                        if args.system
                        else []
                    )
                    messages.append({"role": "user", "content": args.prompt})
                    request = ChatRequest(
                        messages=messages,
                        model=client.config.model_name,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        stream=args.stream,
                    )
                else:
                    request = CompletionRequest(
                        prompt=args.prompt,
                        model=client.config.model_name,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        stream=args.stream,
                    )
                await client.generate(request, args.verbose)
                print_stats(client.stats)
            else:
                # Interactive mode
                await interactive_mode(client, args)
        except (APIError, ValueError, httpx.RequestError, ConnectionError) as e:
            sys.exit(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Polyglot AI: A Universal CLI for the OpenAI API Format",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", default="polyglot.toml", help="Path to the TOML configuration file."
    )
    parser.add_argument(
        "--endpoint",
        default="openai",
        help="The name of the endpoint from the config file to use.",
    )
    parser.add_argument("--model", help="Override the default model for the session.")
    parser.add_argument(
        "--chat", action="store_true", help="Enable chat mode. Required for tool use."
    )
    parser.add_argument("-p", "--prompt", help="Send a single prompt and exit.")
    parser.add_argument("--system", help="Set a system prompt for chat mode.")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no-stream", action="store_false", dest="stream")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging of request parameters.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable raw protocol debug mode for streaming.",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Enable tool calling feature (requires chat mode).",
    )
    args = parser.parse_args()

    print_banner()
    try:
        asyncio.run(async_main(args))
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
