import asyncio
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

    def __init__(
        self,
        debug_mode: bool = False,
        rich_text_mode: bool = True,
        smooth_stream_mode: bool = False,
    ):
        self.debug_mode = debug_mode
        self.rich_text_mode = rich_text_mode
        self.smooth_stream_mode = smooth_stream_mode
        self._printer = print  # Default to standard print
        self._is_interactive = False
        self.output_buffer: Buffer | None = None
        self.actor_name = "🤖 Assistant"
        self.rich_console = Console()
        self.char_queue = asyncio.Queue()
        self._renderer_task: asyncio.Task | None = None

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
        # It's critical to cancel any lingering tasks *before* creating a new one.
        if self._renderer_task and not self._renderer_task.done():
            self._renderer_task.cancel()

        self.current_response = ""
        if self.output_buffer:
            self.output_buffer.reset()

        # Create a new queue, discarding the old one, to prevent processing stale data.
        self.char_queue = asyncio.Queue()

        self.actor_name = actor_name or "🤖 Assistant"
        self.current_request_stats = RequestStats(tokens_sent=tokens_sent)
        self._last_token_time = None
        self.line_count = 0
        self.chunk_count = 0
        self.first_token_received = False
        self.status = "Waiting..."

        # Start the new renderer task *after* all state is reset.
        if self.smooth_stream_mode:
            self._renderer_task = asyncio.create_task(self._smooth_renderer())

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

    def _render_text(self, text: str):
        """Internal helper to render a piece of text, updating buffer or printing."""
        self.current_response += text
        if self._is_interactive and self.output_buffer:
            # We must update the full buffer text each time for prompt_toolkit.
            self.output_buffer.text = f"{self.actor_name}: {self.current_response}"
            # Move the cursor to the end of the buffer to ensure scrolling.
            self.output_buffer.cursor_position = len(self.output_buffer.text)
        elif not self._is_interactive:
            # For non-interactive, we can just print the character.
            self._print(text, end="", flush=True)

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

    async def _smooth_renderer(self):
        """A background task that renders characters from a queue at a smoothed pace."""
        try:
            # Render the first chunk instantly to establish a baseline TPS
            # without an artificial delay.
            first_chunk = await self.char_queue.get()
            if first_chunk is None:
                return
            self._render_text(first_chunk)

            while True:
                chunk_text = await self.char_queue.get()
                if chunk_text is None:  # Sentinel to stop
                    break

                # The core smoothing logic starts from the second chunk.
                target_tps = (
                    self.current_request_stats.live_tok_per_sec
                    if self.current_request_stats
                    else 0
                )

                # If TPS is zero or very low, don't introduce huge delays.
                # Render at a minimum readable speed (e.g., 15 CPS, ~5 TPS).
                if target_tps < 5:
                    delay_per_char = 1.0 / 15.0
                else:
                    target_cps = target_tps * 3.5  # Heuristic
                    delay_per_char = 1.0 / target_cps

                for char in chunk_text:
                    self._render_text(char)
                    await asyncio.sleep(delay_per_char)
        except asyncio.CancelledError:
            # When cancelled, `finish_response` is responsible for draining the queue.
            pass

    async def show_parsed_chunk(self, chunk_data: dict, chunk_text: str):
        """Handles a parsed chunk of text from the stream."""
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

            # Print header once for non-interactive mode.
            if not self._is_interactive:
                self._print(f"\n{self.actor_name}: ", end="")

        self.chunk_count += 1
        if self.smooth_stream_mode:
            await self.char_queue.put(chunk_text)
        else:
            self._render_text(chunk_text)

        # To ensure the live count is consistent with the final count,
        # we recalculate from the full response string after every chunk.
        if self.current_request_stats:
            self.current_request_stats.tokens_received = estimate_tokens(
                self.current_response
            )

        if self.debug_mode:
            duration = (
                self.current_request_stats.current_duration
                if self.current_request_stats
                else 0
            )
            log_line = f"🟢 [{duration:6.2f}s] C{self.chunk_count:03d} TEXT: {repr(chunk_text)}"
            self._print(log_line)
            logging.info(log_line)

    async def finish_response(self, success: bool = True) -> RequestStats | None:
        """Finalizes the response, prints stats, and resets the display state."""
        # In smooth mode, stop the renderer and drain the queue to ensure all text is printed.
        if self._renderer_task and not self._renderer_task.done():
            self._renderer_task.cancel()
            try:
                # Wait for the task to acknowledge cancellation.
                await self._renderer_task
            except asyncio.CancelledError:
                pass  # This is the expected outcome.
            finally:
                self._renderer_task = None

        # Instantly render any text remaining in the queue.
        while not self.char_queue.empty():
            try:
                chunk = self.char_queue.get_nowait()
                if chunk is not None:
                    self._render_text(chunk)
            except asyncio.QueueEmpty:
                break

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
