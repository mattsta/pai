from __future__ import annotations

import asyncio
import logging
import re
import statistics
import time
from html import escape
from typing import Any

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import ANSI, HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .models import RequestStats, SmoothingStats
from .utils import estimate_tokens


class StreamSmoother:
    """
    Calculates adaptive delays for smooth stream rendering.
    Its goal is to provide a consistent rendering speed by maintaining a small
    buffer to hide network jitter.
    """

    def __init__(self):
        # --- Configuration ---
        self.target_buffer_s = 1.5
        self.min_wps = 8.5  # Minimum readable words per second
        self.max_wps = 100.0  # Cap to prevent runaway speeds
        self._urgency_gain = 0.5  # How aggressively we correct buffer errors.

        # --- State ---
        self.current_drain_time_s = 0.0

    def get_render_delay(self, live_tps: float, queue_size: int) -> float:
        """Calculates the next sleep delay to manage the buffer."""
        # 1. Calculate a base render speed directly from the live token rate.
        #    This makes the controller much more responsive to current conditions.
        live_wps = max(live_tps * 1.7, self.min_wps)
        live_wps = min(live_wps, self.max_wps)

        # 2. Calculate the base delay and how long it would take to drain the queue.
        items_per_second_base = live_wps * 2
        base_delay = 1.0 / items_per_second_base if items_per_second_base > 0 else 0.1
        self.current_drain_time_s = (
            queue_size * base_delay if items_per_second_base > 0 else 0
        )
        buffer_error_s = self.current_drain_time_s - self.target_buffer_s

        # 3. Use a P-controller to adjust the render *delay*.
        #    A large buffer (error > 0) should decrease delay (factor < 1).
        #    A small buffer (error < 0) should increase delay (factor > 1).
        urgency_factor = 1.0 - (buffer_error_s * self._urgency_gain)
        urgency_factor = max(
            0.25, min(1.75, urgency_factor)
        )  # Clamp for stability, being more aggressive on speed-up.

        # 4. The final delay is the base delay adjusted by our urgency factor.
        return base_delay * urgency_factor


class StreamingDisplay:
    """Manages all console output, ensuring prompt-toolkit UI is not corrupted."""

    def __init__(
        self,
        debug_mode: bool = False,
        rich_text_mode: bool = True,
        smooth_stream_mode: bool = False,
        enhanced_debug_mode: bool = False,
    ):
        self.ui = None  # Will be set by InteractiveUI
        self.debug_mode = debug_mode
        self.rich_text_mode = rich_text_mode
        self.smooth_stream_mode = smooth_stream_mode
        self.enhanced_debug_mode = enhanced_debug_mode
        if self.enhanced_debug_mode:
            self.debug_mode = True
        self._printer = print  # Default to standard print
        self._is_interactive = False
        self.output_buffer: Buffer | None = None
        self.reasoning_output_buffer: Buffer | None = None
        self.current_reasoning: str = ""
        self.is_in_reasoning_block: bool = False
        self.actor_name = "🤖 Assistant"
        self.current_model_name: str | None = None
        self.rich_console = Console()
        self._word_queue = asyncio.Queue()
        self._smoother_task: asyncio.Task | None = None
        self._smoother: StreamSmoother | None = None
        self._inter_chunk_deltas: list[float] = []
        self._stream_finished: bool = False
        self._smoothing_aborted: bool = False
        self._last_debug_chunk: dict | None = None

        # Debugging stats
        self.total_bytes_received: int = 0
        self.total_content_bytes_received: int = 0

        # State for UI
        self.status = "Idle"
        # Holds the stats for the *in-progress* request.
        self.current_request_stats: RequestStats | None = None
        self._last_token_time: float | None = None

        # Response tracking
        self.line_count: int = 0
        self.chunk_count: int = 0
        # `current_response` tracks the TEXT RENDERED to the screen.
        self.current_response: str = ""
        # `_full_response_text` tracks the TOTAL response received from the API.
        # This is used for accurate token rate calculations.
        self._full_response_text: str = ""
        self.first_token_received = False

    def set_printer(self, printer: callable, is_interactive: bool):
        """Sets the function used for printing to the console."""
        self._printer = printer
        self._is_interactive = is_interactive

    def start_response(
        self,
        tokens_sent: int = 0,
        actor_name: str | None = None,
        model_name: str | None = None,
        is_continuation: bool = False,
    ):
        """Prepares for a new response stream."""
        # For a continuation (e.g., in an agent loop), we preserve the reasoning state.
        if not is_continuation:
            # It's critical to cancel any lingering tasks *before* creating a new one.
            if self._smoother_task and not self._smoother_task.done():
                self._smoother_task.cancel()
            self.current_reasoning = ""
            self.is_in_reasoning_block = False

        self.current_response = ""
        self._full_response_text = ""
        if self.output_buffer:
            self.output_buffer.reset()

        # Create a new queue, discarding the old one, to prevent processing stale data.
        self._word_queue = asyncio.Queue()

        # Create a new queue, discarding the old one, to prevent processing stale data.
        self._word_queue = asyncio.Queue()
        self._inter_chunk_deltas = []

        self.actor_name = actor_name or "🤖 Assistant"
        self.current_model_name = model_name
        self.current_request_stats = RequestStats(tokens_sent=tokens_sent)
        self._last_token_time = None
        self.line_count = 0
        self.chunk_count = 0
        self.total_bytes_received = 0
        self.total_content_bytes_received = 0
        self.first_token_received = False
        self._stream_finished = False
        self._smoothing_aborted = False
        self.status = "Waiting..."
        self._last_debug_chunk = None

        # Start the new renderer task *after* all state is reset.
        if self.smooth_stream_mode:
            self._smoother = StreamSmoother()
            self._smoother_task = asyncio.create_task(self._smoother_task_loop())

    @property
    def time_since_last_token(self) -> float:
        """Returns seconds since the last token was received, for spinner UI."""
        if self.status == "Streaming" and self._last_token_time:
            return time.time() - self._last_token_time
        return 0.0

    @property
    def smoothing_stats(self) -> SmoothingStats:
        """
        Calculates and returns statistics for the stream.
        This is used for both live smooth-mode display and final jitter stats.
        """
        stats = SmoothingStats(
            stream_finished=self._stream_finished,
        )

        # Populate smoother-specific stats only if in that mode
        if self.smooth_stream_mode:
            stats.queue_size = self._word_queue.qsize()
            stats.smoothing_aborted = self._smoothing_aborted
            if self._smoother:
                stats.buffer_drain_time_s = self._smoother.current_drain_time_s

        deltas = self._inter_chunk_deltas
        # Calculate network jitter stats
        if len(deltas) > 1:
            try:
                mean = statistics.mean(deltas)
                stdev = statistics.stdev(deltas) if len(deltas) > 2 else 0

                stats.arrivals = len(deltas)
                # Filter out near-zero deltas (artifacts) for a more meaningful min value
                non_zero_deltas = [d for d in deltas if d > 1e-3]  # 1ms
                min_val = min(non_zero_deltas) if non_zero_deltas else 0.0
                stats.min_delta = f"{min_val * 1000:.1f}"
                stats.mean_delta = f"{mean * 1000:.1f}"
                stats.stdev_delta = f"{stdev * 1000:.1f}"
                stats.max_delta = f"{max(deltas) * 1000:.1f}"

                gaps = (
                    [d for d in deltas if d > mean + 1.5 * stdev] if stdev > 0 else []
                )
                bursts = (
                    sum(1 for d in deltas if d < mean - 0.75 * stdev)
                    if mean and stdev
                    else 0
                )
                stats.gaps = len(gaps)
                stats.bursts = bursts
            except statistics.StatisticsError:
                pass  # Not enough data for stats
        return stats

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

    def _render_reasoning(self, text: str):
        """Renders reasoning text, updating the internal state and live buffer."""
        self.current_reasoning += text
        if self._is_interactive and self.reasoning_output_buffer:
            # We must update the full buffer text each time for prompt_toolkit.
            title = self.actor_name
            if self.current_model_name:
                title += f" ({self.current_model_name})"
            header = f"🤔 {title} (Thinking)"
            self.reasoning_output_buffer.text = f"{header}: {self.current_reasoning}"
            # Move the cursor to the end of the buffer to ensure scrolling.
            self.reasoning_output_buffer.cursor_position = len(
                self.reasoning_output_buffer.text
            )


    def show_raw_line(self, line: str):
        if self.debug_mode:
            if self.enhanced_debug_mode and line.startswith("data: "):
                # In enhanced mode, the diff from parsed chunks is shown instead,
                # so we suppress the raw line to avoid noise.
                return
            if not self.first_token_received:
                header = "🔍 DEBUG MODE: Showing raw protocol traffic\n" + "=" * 60
                self._print(header)
                logging.info(header)
                self.first_token_received = (
                    True  # Prevent header from printing multiple times
                )
            self.line_count += 1
            self.total_bytes_received += len(line.encode("utf-8"))
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

    def show_tool_call_request(self, tool_name: str, tool_args: dict):
        """Displays a formatted request for a tool call, for confirmation."""
        import json

        from prompt_toolkit.formatted_text import ANSI, HTML
        from rich.panel import Panel
        from rich.syntax import Syntax

        args_json = json.dumps(tool_args, indent=2)
        syntax = Syntax(args_json, "json", theme="monokai", word_wrap=True)
        panel = Panel(
            syntax,
            title="Arguments",
            title_align="left",
            border_style="dim magenta",
        )

        self._print(
            HTML(
                f"\n<style fg='ansimagenta' bg='ansiblack'>🔧 Agent wants to execute: <b>{escape(tool_name)}</b></style>"
            )
        )
        with self.rich_console.capture() as capture:
            self.rich_console.print(panel)
        self._print(ANSI(capture.get()))

    def show_agent_tool_call(self, tool_name: str, tool_args: dict):
        """Prints a standardized, pretty-printed representation of a tool call."""
        import json

        from prompt_toolkit.formatted_text import ANSI
        from rich.panel import Panel
        from rich.syntax import Syntax

        args_json = json.dumps(tool_args, indent=2)
        syntax = Syntax(args_json, "json", theme="monokai", word_wrap=True)

        panel = Panel(
            syntax,
            title=f"Executing: {tool_name}",
            title_align="left",
            border_style="dim magenta",
        )
        with self.rich_console.capture() as capture:
            self.rich_console.print(panel)
        self._print(ANSI(capture.get()))

    def show_agent_tool_result(self, tool_name: str, tool_result: Any):
        """Prints a standardized, pretty-printed representation of a tool result."""
        import json

        from prompt_toolkit.formatted_text import ANSI
        from rich.panel import Panel
        from rich.syntax import Syntax

        result_str = str(tool_result)

        try:
            # If the result is JSON, pretty-print it.
            result_data = json.loads(result_str)
            result_json = json.dumps(result_data, indent=2)
            syntax = Syntax(result_json, "json", theme="monokai", word_wrap=True)
            content = syntax
            border_style = "dim green"
            if isinstance(result_data, dict) and result_data.get("status") == "failure":
                border_style = "dim red"

        except (json.JSONDecodeError, TypeError):
            # Otherwise, print as plain text.
            syntax = Syntax(result_str, "text", theme="monokai", word_wrap=True)
            content = syntax
            border_style = "dim"

        panel = Panel(
            content,
            title=f"Result from: {tool_name}",
            title_align="left",
            border_style=border_style,
        )
        with self.rich_console.capture() as capture:
            self.rich_console.print(panel)
        self._print(ANSI(capture.get()))

    def commit_partial_response(self):
        """
        Commits the currently streamed response to the scrollback buffer
        without ending the request tracking. This is used when a stream
        is interrupted by a tool call.
        """
        # When a tool call happens, we commit the text content but *preserve*
        # the reasoning block, as the agent may continue thinking after the
        # tool result comes back. The full reasoning block is committed only
        # at the end of the turn.
        if not self._is_interactive and self.current_response:
            self._print("\n")
        elif self._is_interactive and self.current_response:
            # Essentially a stripped-down version of finish_response's rendering part.
            if self.rich_text_mode:
                final_text = self._escape_html_in_markdown(self.current_response)
                title = self.actor_name
                if self.current_model_name:
                    title += f" ({self.current_model_name})"
                panel_to_print = Panel(
                    Markdown(final_text, code_theme="monokai"),
                    title=title,
                    title_align="left",
                    border_style="dim",
                )
                with self.rich_console.capture() as capture:
                    self.rich_console.print(panel_to_print)
                self._printer(ANSI(capture.get()))
            else:
                title = self.actor_name
                if self.current_model_name:
                    title += f" ({self.current_model_name})"
                self._printer(HTML(f"{escape(title)}: {escape(self.current_response)}"))

        # Reset the text buffers for the next part of the turn (the tool call/result).
        # But DON'T reset the stats or other state.
        self.current_response = ""
        self._full_response_text = ""
        if self.output_buffer:
            self.output_buffer.reset()

    def _get_dict_diff(self, d1: dict, d2: dict) -> dict:
        """
        Recursively finds differences between two dictionaries.
        Handles nested dicts and lists of dicts by diffing corresponding items.
        """
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            return d2 if d1 != d2 else {}

        diff = {}
        all_keys = set(d1.keys()) | set(d2.keys())

        for k in sorted(list(all_keys)):
            v1 = d1.get(k)
            v2 = d2.get(k)

            if k not in d1:
                # Key was added and is not None
                if v2 is not None:
                    diff[k] = v2
            elif k not in d2:
                # Key was removed
                diff[k] = None  # Represent removal with None
            elif v1 != v2:
                # Value changed
                if isinstance(v1, dict) and isinstance(v2, dict):
                    sub_diff = self._get_dict_diff(v1, v2)
                    if sub_diff:
                        diff[k] = sub_diff
                elif (
                    isinstance(v1, list)
                    and isinstance(v2, list)
                    and len(v1) == len(v2)
                    and all(isinstance(i, dict) for i in v1 + v2)
                ):
                    # It's a list of dicts of the same length, diff them item by item.
                    list_diffs = [self._get_dict_diff(i1, i2) for i1, i2 in zip(v1, v2)]
                    # Only include if there's at least one non-empty diff
                    if any(list_diffs):
                        # Don't show empty dicts from unchanged items in the list
                        diff[k] = [ld for ld in list_diffs if ld]
                else:
                    # For simple values, or lists that changed structure, show the new value.
                    diff[k] = v2
        return diff

    def commit_reasoning(self):
        """Commits the current reasoning block to the display and resets it."""
        if not self.current_reasoning or not self._is_interactive:
            return

        title = self.actor_name
        if self.current_model_name:
            title += f" ({self.current_model_name})"

        panel = Panel(
            Markdown(self.current_reasoning, code_theme="monokai"),
            title=f"🤔 {title} (Thinking)",
            title_align="left",
            border_style="grey50",
        )
        with self.rich_console.capture() as capture:
            self.rich_console.print(panel)
        self._printer(ANSI(capture.get()))

        # Add to the turn's permanent reasoning log
        if self.ui and self.ui.conversation.turns:
            last_turn = self.ui.conversation.turns[-1]
            if last_turn.assistant_reasoning:
                last_turn.assistant_reasoning += f"\n\n---\n\n{self.current_reasoning}"
            else:
                last_turn.assistant_reasoning = self.current_reasoning

        # Reset for the next reasoning block in this turn.
        self.current_reasoning = ""
        self.is_in_reasoning_block = False
        if self.reasoning_output_buffer:
            self.reasoning_output_buffer.reset()

    async def _smoother_task_loop(self):
        """A background task that calls renderers with tokens from a queue."""
        try:
            while True:
                item = await self._word_queue.get()
                if item is None:  # Sentinel to stop
                    self._word_queue.task_done()
                    break

                token_type, token = item

                if not self._smoother:
                    await asyncio.sleep(0.01)  # Should not happen
                    continue

                live_tps = (
                    self.current_request_stats.live_tok_per_sec
                    if self.current_request_stats
                    else 0.0
                )
                queue_size = self._word_queue.qsize()
                delay = self._smoother.get_render_delay(live_tps, queue_size)

                if token_type == "content":
                    self._render_text(token)
                elif token_type == "reasoning":
                    self._render_reasoning(token)

                self._word_queue.task_done()
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            # On cancellation, immediately render any remaining text.
            while not self._word_queue.empty():
                try:
                    item = self._word_queue.get_nowait()
                    if item:
                        token_type, token = item
                        if token_type == "content":
                            self._render_text(token)
                        elif token_type == "reasoning":
                            self._render_reasoning(token)
                    self._word_queue.task_done()
                except asyncio.QueueEmpty:
                    break

    async def abort_smoothing(self):
        """
        Aborts the active smooth streaming task, dumps the queue, and reverts
        to line-rate streaming for the rest of the response.
        """
        if self._smoothing_aborted:
            return  # Avoid running twice
        self._smoothing_aborted = True
        if self._smoother_task and not self._smoother_task.done():
            self._smoother_task.cancel()
            try:
                await self._smoother_task
            except asyncio.CancelledError:
                pass  # This is the expected outcome.
            finally:
                self._smoother_task = None

        # Immediately render any text remaining in the queue.
        while not self._word_queue.empty():
            try:
                token = self._word_queue.get_nowait()
                if token is not None:
                    self._render_text(token)
            except asyncio.QueueEmpty:
                break

    def _escape_html_in_markdown(self, text: str) -> str:
        """
        Escapes HTML entities in a Markdown string, but preserves content within
        backticks (inline code) and triple-backticks (fenced code blocks).
        """
        code_blocks: list[str] = []

        def replace_with_placeholder(match: re.Match) -> str:
            # Use a unique, unlikely-to-be-typed placeholder
            placeholder = f"__PAI_CODE_BLOCK_PLACEHOLDER_{len(code_blocks)}__"
            code_blocks.append(match.group(0))
            return placeholder

        # 1. First, find and replace all fenced code blocks.
        #    The (?s) flag is equivalent to re.DOTALL, making . match newlines.
        text_no_fences = re.sub(r"(?s)```.*?```", replace_with_placeholder, text)

        # 2. Then, find and replace all inline code blocks in the remaining text.
        #    This regex is safer as it won't match across newlines for inline code.
        text_no_code = re.sub(r"`[^`\n\r]*`", replace_with_placeholder, text_no_fences)

        # 3. Escape all HTML special characters in the text that remains.
        #    The placeholders themselves won't be escaped.
        escaped_text = escape(text_no_code)

        # For Markdown, a single newline is treated as a space. To force a hard
        # line break, we must replace newlines with two spaces followed by a newline.
        # This is done here, before code blocks are restored, so it doesn't affect their content.
        escaped_text = escaped_text.replace("\n", "  \n")

        # 4. Restore the code blocks.
        for i, block in enumerate(code_blocks):
            placeholder = f"__PAI_CODE_BLOCK_PLACEHOLDER_{i}__"
            escaped_text = escaped_text.replace(placeholder, block, 1)

        return escaped_text

    async def show_parsed_chunk(
        self, chunk_data: dict, content: str, reasoning: str | None = None
    ):
        """Handles a parsed chunk of a stream, separating content and reasoning."""
        # The logic to detect termination vs. continuation must be robust against
        # interleaved content chunks. A state machine (`is_in_reasoning_block`)
        # is used to track whether we are actively processing a thought block.
        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
        # The presence of the `reasoning` key in the delta is the most reliable signal.
        has_reasoning_key_in_delta = "reasoning" in delta

        # A non-empty reasoning string continues or starts a thought process.
        if reasoning:
            if not self.is_in_reasoning_block:
                self.is_in_reasoning_block = True

            # Stream the reasoning token.
            if self.smooth_stream_mode and not self._smoothing_aborted:
                tokens = re.split(r"(\s+)", reasoning)
                for token in tokens:
                    if token:
                        await self._word_queue.put(("reasoning", token))
            else:
                self._render_reasoning(reasoning)

        # Termination condition for streaming: The key was present, but the value is None.
        # This is the explicit `reasoning: null` signal.
        elif has_reasoning_key_in_delta and reasoning is None:
            has_tool_calls = bool(delta.get("tool_calls"))
            if self.is_in_reasoning_block and not has_tool_calls:
                self.commit_reasoning()
        # If a chunk has no `reasoning` key, we do NOT terminate a block. This
        # correctly handles interleaved content chunks during a thought process.

        # Only count bytes and update total text if there is text.
        if content:
            self.total_content_bytes_received += len(content.encode("utf-8"))
        current_chunk_time = time.time()
        if self._last_token_time:
            self._inter_chunk_deltas.append(current_chunk_time - self._last_token_time)
        self._last_token_time = current_chunk_time

        if not self.first_token_received:
            self.status = "Streaming"
            if self.current_request_stats:
                self.current_request_stats.record_first_token()
            self.first_token_received = True
            if not self._is_interactive:
                title = self.actor_name
                if self.current_model_name:
                    title += f" ({self.current_model_name})"
                self._print(f"\n{title}: ", end="")

        self.chunk_count += 1
        # Immediately update the full response text for accurate token rate calculation.
        if content:
            # Immediately update the full response text for accurate token rate calculation.
            self._full_response_text += content
            if self.current_request_stats:
                self.current_request_stats.tokens_received = estimate_tokens(
                    self._full_response_text
                )

            if self.smooth_stream_mode and not self._smoothing_aborted:
                # Split the text while preserving whitespace as separate tokens.
                # This ensures that newlines and multiple spaces are handled correctly.
                tokens = re.split(r"(\s+)", content)
                for token in tokens:
                    if token:  # Don't queue empty strings
                        await self._word_queue.put(("content", token))
            else:
                # Non-smooth mode renders directly and updates the rendered text state.
                self._render_text(content)

        if self.debug_mode:
            duration = (
                self.current_request_stats.current_duration
                if self.current_request_stats
                else 0
            )
            if self.enhanced_debug_mode:
                import json

                diff = self._get_dict_diff(self._last_debug_chunk or {}, chunk_data)
                if diff:  # Only print if there are changes
                    # compact, single-line JSON format
                    diff_str = json.dumps(diff, separators=(",", ":"))
                    log_line = f"🔵 [{duration:6.2f}s] C{self.chunk_count:03d} DIFF: {diff_str}"
                    self._print(log_line)
                    logging.info(log_line)
                self._last_debug_chunk = chunk_data
            else:
                log_line = f"🟢 [{duration:6.2f}s] C{self.chunk_count:03d} TEXT: {repr(content)}"
                self._print(log_line)
                logging.info(log_line)

    async def finish_response(
        self, success: bool = True, usage: dict | None = None
    ) -> RequestStats | None:
        """
        Finalizes the response, prints stats, and resets the display state.
        If a 'usage' dict from the provider is passed, it will be used for exact
        token counts.
        """
        self._stream_finished = True
        # In smooth mode, signal the renderer to stop and wait for it to finish.
        if self.smooth_stream_mode and self._smoother_task:
            await self._word_queue.put(None)  # Send sentinel
            await self._smoother_task
            self._smoother_task = None

        # Sync the rendered text with the full text to ensure nothing is missed.
        self.current_response = self._full_response_text

        self.status = "Done"
        if self.current_request_stats:
            # Capture final stream stats before they are reset
            if self.first_token_received:
                self.current_request_stats.jitter_stats = self.smoothing_stats

            self.current_request_stats.finish(success=success)

            # Use provider's exact token counts if available, otherwise estimate.
            if usage:
                # Received tokens (check for openai, anthropic, ollama keys)
                if usage.get("completion_tokens") is not None:
                    self.current_request_stats.tokens_received = usage[
                        "completion_tokens"
                    ]
                elif usage.get("output_tokens") is not None:
                    self.current_request_stats.tokens_received = usage["output_tokens"]
                elif usage.get("eval_count") is not None:
                    self.current_request_stats.tokens_received = usage["eval_count"]
                else:
                    self.current_request_stats.tokens_received = estimate_tokens(
                        self.current_response
                    )

                # Sent tokens (check for openai, anthropic, ollama keys)
                if usage.get("prompt_tokens") is not None:
                    self.current_request_stats.tokens_sent = usage["prompt_tokens"]
                elif usage.get("input_tokens") is not None:
                    self.current_request_stats.tokens_sent = usage["input_tokens"]
                elif usage.get("prompt_eval_count") is not None:
                    self.current_request_stats.tokens_sent = usage["prompt_eval_count"]
            else:
                # Fallback to estimation from the final rendered text.
                self.current_request_stats.tokens_received = estimate_tokens(
                    self.current_response
                )

        # If any reasoning text is lingering, commit it.
        # This handles cases where the stream ends with a reasoning block.
        self.commit_reasoning()

        # In interactive mode, if we have a response, print it to the scrollback
        # history. This "finalizes" it, moving it from the temporary live
        # buffer to the main conversation transcript. This happens regardless
        # of success to ensure partial/cancelled outputs are preserved.
        if self._is_interactive and self.current_response:
            if self.rich_text_mode:
                # Escape HTML tags but preserve markdown code blocks to prevent
                # the renderer from consuming them.
                final_text = self._escape_html_in_markdown(self.current_response)
                # Render final output as Markdown inside a panel for clarity
                title = self.actor_name
                if self.current_model_name:
                    title += f" ({self.current_model_name})"
                panel_to_print = Panel(
                    Markdown(final_text, code_theme="monokai"),
                    title=title,
                    title_align="left",
                    border_style="dim",
                )
                # Capture the rich output and print it as ANSI text for prompt-toolkit
                with self.rich_console.capture() as capture:
                    self.rich_console.print(panel_to_print)
                self._printer(ANSI(capture.get()))
            else:
                title = self.actor_name
                if self.current_model_name:
                    title += f" ({self.current_model_name})"
                self._printer(HTML(f"{escape(title)}: {escape(self.current_response)}"))

        # On success, print final stats.
        if success and self.current_request_stats:
            stats = self.current_request_stats
            if self.debug_mode:
                total_words = len(self.current_response.split())
                protocol_bytes = (
                    self.total_bytes_received - self.total_content_bytes_received
                )
                overhead_per_word = (
                    protocol_bytes / total_words if total_words > 0 else 0
                )
                content_to_total_ratio = (
                    self.total_content_bytes_received / self.total_bytes_received
                    if self.total_bytes_received > 0
                    else 0
                )

                # Use the finalized stats from the request object, which may have been
                # updated with exact counts from the provider.
                content_tokens = self.current_request_stats.tokens_received
                json_payloads = self.chunk_count

                avg_content_bytes_per_token = (
                    self.total_content_bytes_received / content_tokens
                    if content_tokens > 0
                    else 0
                )
                avg_total_bytes_per_token = (
                    self.total_bytes_received / content_tokens
                    if content_tokens > 0
                    else 0
                )
                json_per_token_ratio = (
                    json_payloads / content_tokens if content_tokens > 0 else 0
                )

                summary_lines = [
                    "=" * 60,
                    f"🔍 DEBUG SUMMARY: {self.line_count} lines, {json_payloads} chunks, {stats.response_time:.2f}s",
                    "-" * 20,
                    "  Byte & Word Stats:",
                    f"  - Total Bytes Received:   {self.total_bytes_received:7d} B",
                    f"  - Content Bytes Received: {self.total_content_bytes_received:7d} B ({content_to_total_ratio:.1%})",
                    f"  - Protocol Overhead:      {protocol_bytes:7d} B",
                    f"  - Total Words Received:   {total_words:7d} words",
                    f"  - Overhead per Word:      {overhead_per_word:7.2f} bytes/word",
                    "-" * 20,
                    "  Token & Overhead Stats:",
                    f"  - Content Tokens (est.):  {content_tokens:7d} tokens",
                    f"  - JSON Payloads (chunks): {json_payloads:7d} chunks",
                    f"  - Avg. Content Bytes/Token: {avg_content_bytes_per_token:7.2f} B/tok",
                    f"  - Avg. Total Bytes/Token:   {avg_total_bytes_per_token:7.2f} B/tok",
                    f"  - JSON/Token Ratio:         {json_per_token_ratio:7.2f} chunks/tok",
                    "=" * 60,
                ]
                summary = "\n".join(summary_lines)
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
