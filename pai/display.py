import logging
import time
from html import escape

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import ANSI, HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .models import RequestStats
from .utils import estimate_tokens


class StreamingDisplay:
    """Manages all console output, ensuring prompt-toolkit UI is not corrupted."""

    def __init__(self, debug_mode: bool = False, rich_text_mode: bool = True):
        self.debug_mode = debug_mode
        self.rich_text_mode = rich_text_mode
        self._printer = print  # Default to standard print
        self._is_interactive = False
        self.output_buffer: Buffer | None = None
        self.actor_name = "🤖 Assistant"
        self.rich_console = Console()

        # State for UI
        self.status = "Idle"
        # Holds the stats for the *in-progress* request.
        self.current_request_stats: RequestStats | None = None
        self._last_token_time: float | None = None

        # Response tracking
        self.line_count: int = 0
        self.chunk_count: int = 0
        self.current_response: str = ""
        self.first_token_received = False

    def set_printer(self, printer: callable, is_interactive: bool):
        """Sets the function used for printing to the console."""
        self._printer = printer
        self._is_interactive = is_interactive

    def start_response(self, tokens_sent: int = 0, actor_name: str | None = None):
        """Prepares for a new response stream."""
        self.current_response = ""
        if self.output_buffer:
            self.output_buffer.reset()

        self.actor_name = actor_name or "🤖 Assistant"
        self.current_request_stats = RequestStats(tokens_sent=tokens_sent)
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
                header = "🔍 DEBUG MODE: Showing raw protocol traffic\n" + "=" * 60
                self._print(header)
                logging.info(header)
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
            log_line = f"{prefix}{repr(line)}"
            self._print(log_line)
            logging.info(log_line)

    def show_parsed_chunk(self, chunk_data: dict, chunk_text: str):
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
            self.output_buffer.text = f"{self.actor_name}: {self.current_response}"
            # Move the cursor to the end of the buffer to ensure scrolling.
            self.output_buffer.cursor_position = len(self.output_buffer.text)
        elif not self._is_interactive:
            # For non-interactive, print header once, then stream chunks.
            if self.chunk_count == 1:
                self._print(f"\n{self.actor_name}: ", end="")
            self._print(chunk_text, end="", flush=True)

        if self.debug_mode:
            duration = (
                self.current_request_stats.current_duration
                if self.current_request_stats
                else 0
            )
            log_line = f"🟢 [{duration:6.2f}s] C{self.chunk_count:03d} TEXT: {repr(chunk_text)}"
            self._print(log_line)
            logging.info(log_line)

    def finish_response(self, success: bool = True) -> RequestStats | None:
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
            if self.rich_text_mode:
                # Render final output as Markdown inside a panel for clarity
                panel_to_print = Panel(
                    Markdown(self.current_response.strip(), code_theme="monokai"),
                    title=self.actor_name,
                    title_align="left",
                    border_style="dim",
                )
                # Capture the rich output and print it as ANSI text for prompt-toolkit
                with self.rich_console.capture() as capture:
                    self.rich_console.print(panel_to_print)
                self._printer(ANSI(capture.get()))
            else:
                self._printer(
                    HTML(f"{escape(self.actor_name)}: {escape(self.current_response)}")
                )

        # On success, print final stats.
        if success and self.current_request_stats:
            stats = self.current_request_stats
            if self.debug_mode:
                summary = (
                    "=" * 60
                    + f"\n🔍 DEBUG SUMMARY: {self.line_count} lines, {self.chunk_count} chunks, {stats.response_time:.2f}s\n"
                    + "=" * 60
                )
                self._print(summary)
                logging.info(summary)
            elif not self._is_interactive and self.first_token_received:
                # For non-interactive mode, print the final stats line.
                tok_per_sec = stats.final_tok_per_sec
                ttft_str = (
                    f" | TTFT: {stats.ttft:.2f}s" if stats.ttft is not None else ""
                )
                self._print(
                    f"\n\n📊 Response in {stats.response_time:.2f}s ({stats.tokens_received} tokens, {tok_per_sec:.1f} tok/s{ttft_str})"
                )

        # The buffer is reset by start_response(), so we just reset status here.
        # Keeping the buffer allows the final response to remain visible.
        self.status = "Idle"
        return self.current_request_stats
