"""Orchestrator for running a single prompt multiple times concurrently."""

import asyncio
from copy import deepcopy
from html import escape

from prompt_toolkit.formatted_text import HTML

from ..log_utils import save_conversation_formats
from ..models import ChatRequest, CompletionRequest, Turn, UIMode
from .base import BaseOrchestrator


class MultiplyOrchestrator(BaseOrchestrator):
    """Handles running a prompt N times concurrently."""

    async def run(self, user_input: str | None = None):
        if not user_input:
            self.ui.generation_in_progress.clear()
            self.ui.generation_task = None
            return

        multiplier = self.state.multiplier
        self.ui.active_concurrent_count = multiplier
        self.state.multiplier = 1  # Reset after use (one-shot)

        # Create a base request from the current mode (similar to a simplified DefaultOrchestrator)
        base_request: ChatRequest | CompletionRequest | None = None
        if self.state.mode == UIMode.COMPLETION:
            base_request = CompletionRequest(
                prompt=user_input,
                model=self.client.config.model_name,
                max_tokens=self.runtime_config.max_tokens,
                temperature=self.runtime_config.temperature,
                stream=self.runtime_config.stream,
            )
        else:  # Assumes a chat-like mode
            messages = self.conversation.get_messages_for_next_turn(user_input)
            base_request = ChatRequest(
                messages=messages,
                model=self.client.config.model_name,
                max_tokens=self.runtime_config.max_tokens,
                temperature=self.runtime_config.temperature,
                stream=self.runtime_config.stream,
                # Tools are not supported in multiply mode for simplicity
                tools=[],
            )

        if not base_request:
            self.pt_printer("❌ Could not create a request for multiply mode.")
            self.ui.generation_in_progress.clear()
            self.ui.generation_task = None
            self.ui.active_concurrent_count = 0
            return

        tasks = []
        for i in range(multiplier):
            req = deepcopy(base_request)
            display = self.ui.displays[i]
            task = asyncio.create_task(
                self.client.generate(
                    request=req,
                    verbose=self.runtime_config.verbose,
                    display_override=display,
                )
            )
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # This block ensures the UI is reset even if tasks are cancelled.
            self.ui.active_concurrent_count = 0
            self.ui.generation_in_progress.clear()
            self.ui.generation_task = None

        # --- Process and log results ---
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Use HTML to ensure consistent coloring and escape to prevent tag injection.
                # quote=False prevents mangling of apostrophes.
                self.pt_printer(
                    HTML(
                        f"<style fg='ansired'>❌ Request {i + 1} failed: {escape(str(result), quote=False)}</style>"
                    )
                )
                continue

            display = self.ui.displays[i]
            if display.current_request_stats:
                turn = Turn(
                    request_data=result.get("request", {}),
                    response_data=result.get("response", {}),
                    assistant_message=result.get("text", ""),
                    mode=self.state.mode,
                    stats=display.current_request_stats,
                    endpoint_name=self.client.config.name,
                )
                self.conversation.add_turn(turn, turn.stats)
            else:
                self.pt_printer(f"⚠️ Could not find stats for request {i + 1} to log.")

            # The final rendering is done by StreamingDisplay.finish_response, which is
            # called by the protocol adapters. We don't need to re-print here.

        # Save the conversation with all turns at the very end.
        try:
            save_conversation_formats(self.conversation, self.log_dir, self.pt_printer)
        except Exception as e:
            self.pt_printer(f"\n⚠️  Warning: Could not save session turns: {e}")
