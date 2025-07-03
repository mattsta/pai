"""Orchestrator for default chat, completion, and native agent modes."""

import asyncio
import json
from html import escape
from typing import Any

from jinja2 import Environment
from prompt_toolkit.formatted_text import HTML

from ..log_utils import save_conversation_formats
from ..models import ChatRequest, CompletionRequest, Turn, UIMode
from ..tools import get_tool_schemas
from ..utils import estimate_tokens
from .base import BaseOrchestrator


class DefaultOrchestrator(BaseOrchestrator):
    """Handles standard chat, completion, and native agent modes."""

    async def run(self, user_input: str | None = None) -> Any:
        if not user_input:
            self.ui.generation_in_progress.clear()
            self.ui.generation_task = None
            return

        request: ChatRequest | CompletionRequest | None = None
        try:
            if self.state.mode == UIMode.TEMPLATE_COMPLETION:
                if not self.state.chat_template_obj:
                    self.pt_printer(
                        HTML(
                            "<style fg='ansired'>❌ No chat template loaded. Use /template to load one for the current model.</style>"
                        )
                    )
                    self.ui.generation_in_progress.clear()
                    self.ui.generation_task = None
                    return

                messages = self.conversation.get_messages_for_next_turn(user_input)

                try:
                    rendered_prompt = self.state.chat_template_obj.render(
                        messages=messages, add_generation_prompt=True
                    )
                    if self.runtime_config.verbose:
                        self.pt_printer(
                            HTML(
                                f"\n<style fg='grey'>[Rendered Template Prompt]</style>\n{escape(rendered_prompt)}"
                            )
                        )
                except Exception as e:
                    self.pt_printer(
                        HTML(
                            f"<style fg='ansired'>❌ Error rendering chat template: {e}</style>"
                        )
                    )
                    self.ui.generation_in_progress.clear()
                    self.ui.generation_task = None
                    return

                request = CompletionRequest(
                    prompt=rendered_prompt,
                    model=self.client.config.model_name,
                    max_tokens=self.runtime_config.max_tokens,
                    temperature=self.runtime_config.temperature,
                    stream=self.runtime_config.stream,
                )
            elif self.state.mode == UIMode.COMPLETION:
                request = CompletionRequest(
                    prompt=user_input,
                    model=self.client.config.model_name,
                    max_tokens=self.runtime_config.max_tokens,
                    temperature=self.runtime_config.temperature,
                    stream=self.runtime_config.stream,
                )
            else:  # CHAT or NATIVE_AGENT
                messages = self.conversation.get_messages_for_next_turn(user_input)
                request = ChatRequest(
                    messages=messages,
                    model=self.client.config.model_name,
                    max_tokens=self.runtime_config.max_tokens,
                    temperature=self.runtime_config.temperature,
                    stream=self.runtime_config.stream,
                    tools=get_tool_schemas() if self.client.tools_enabled else [],
                )

            if request:
                confirmer = (
                    self.ui._confirm_tool_call
                    if self.runtime_config.confirm_tool_use
                    else None
                )
                result = await self.client.generate(
                    request, self.runtime_config.verbose, confirmer=confirmer
                )

                # The orchestrator is responsible for updating agent stats on the UI state.
                self.state.tools_used += result.get("tools_used", 0)
                if (loops := result.get("agent_loops")) is not None:
                    self.state.agent_loops = loops

                if result:
                    turn = Turn(
                        request_data=result.get("request", {}),
                        response_data=result.get("response", {}),
                        assistant_message=result.get("text", ""),
                    )

                    # For completion mode, we need to add a "user" message to the
                    # conversation explicitly, since it's not part of the request payload.
                    if self.state.mode == UIMode.COMPLETION:
                        # This feels a bit hacky. Maybe add_turn should handle it.
                        # It's better to modify add_turn.
                        pass # No, I will modify add_turn. The request_data has the prompt.

                    request_stats = self.client.stats.last_request_stats
                    self.conversation.add_turn(turn, request_stats)
                    try:
                        turn_file = self.log_dir / f"{turn.turn_id}-turn.json"
                        turn_file.write_text(
                            json.dumps(turn.to_dict(), indent=2), encoding="utf-8"
                        )
                        save_conversation_formats(
                            self.conversation,
                            self.log_dir,
                            printer=self.pt_printer,
                        )
                    except Exception as e:
                        self.pt_printer(
                            f"\n⚠️  Warning: Could not save session turn: {e}"
                        )
        except asyncio.CancelledError:
            request_stats = await self.client.display.finish_response(success=False)
            partial_text = self.client.display.current_response
            if request_stats and request and partial_text:
                request_stats.finish_reason = "cancelled"
                if isinstance(request, ChatRequest):
                    request_stats.tokens_sent = sum(
                        estimate_tokens(m.get("content", "")) for m in request.messages
                    )
                else:
                    request_stats.tokens_sent = estimate_tokens(request.prompt)
                self.client.stats.add_completed_request(request_stats)
                if self.state.mode != UIMode.COMPLETION:
                    self._log_cancelled_turn(request, partial_text)
            self.pt_printer(
                HTML("\n<style fg='ansiyellow'>🚫 Generation cancelled.</style>")
            )
        except Exception as e:
            await self.client.display.finish_response(success=False)
            self.pt_printer(HTML(f"<style fg='ansired'>❌ ERROR: {e}</style>"))
        finally:
            self.ui.generation_in_progress.clear()
            self.ui.generation_task = None

    def _log_cancelled_turn(
        self, request: ChatRequest | CompletionRequest, partial_text: str
    ):
        """Saves information about a cancelled turn for debugging."""
        request_data = request.to_dict(self.client.config.model_name)
        response_data = {
            "pai_note": "This response was cancelled by the user.",
            "choices": [{"message": {"role": "assistant", "content": partial_text}}],
        }
        turn = Turn(
            request_data=request_data,
            response_data=response_data,
            assistant_message=partial_text,
        )
        self.conversation.add_turn(turn)
        try:
            turn_file = self.log_dir / f"{turn.turn_id}-turn.json"
            turn_file.write_text(json.dumps(turn.to_dict(), indent=2), encoding="utf-8")
            save_conversation_formats(
                self.conversation, self.log_dir, printer=self.pt_printer
            )
        except Exception as e:
            self.pt_printer(f"\n⚠️  Warning: Could not save cancelled session turn: {e}")
