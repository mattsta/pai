"""Orchestrator for multi-model arena mode."""

import asyncio
import random
from html import escape
from typing import Any

from prompt_toolkit.formatted_text import HTML

from ..log_utils import save_conversation_formats
from ..models import ArenaConversationStyle, ArenaTurnOrder, ChatRequest, Turn, UIMode
from ..tools import get_tool_schemas
from ..utils import estimate_tokens
from .base import BaseOrchestrator


class ArenaOrchestrator(BaseOrchestrator):
    """Handles the arena loop and judging."""

    def _wildcard_temp_spike(self, request: ChatRequest) -> tuple[ChatRequest, str]:
        """Temporarily spikes the temperature for a single turn."""
        original_temp = request.temperature
        new_temp = round(random.uniform(1.2, 1.8), 2)
        request.temperature = new_temp
        message = f"🔥 Wildcard! Temperature spiked from {original_temp} to {new_temp}."
        return request, message

    def _wildcard_prompt_injection(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str]:
        """Injects a random instruction into the latest user prompt."""
        injections = [
            "Be more argumentative.",
            "Use an analogy in your response.",
            "Summarize the opponent's last point before making your own.",
            "Ask a clarifying question.",
            "Respond in the style of a Shakespearean play.",
            "Your response MUST include a pun.",
            "Explain it like you are talking to a five-year-old.",
        ]
        injection = random.choice(injections)
        # The last message is always the one we're responding to.
        messages[-1]["content"] += (
            f"\n\n(Wildcard Instruction: You MUST follow this instruction for this turn only: {injection})"
        )
        message = f"📝 Wildcard! The following instruction was added: '{injection}'"
        return messages, message

    async def run(self, user_input: str | None = None) -> Any:
        if not self.state.arena:
            self.pt_printer("❌ Arena state not found.")
            return

        if not user_input:
            # Should not happen if called correctly
            return

        # The `run` method is called once to start the arena.
        # It's the coroutine for the main generation task.
        # We set the initial message and start the orchestration loop.
        self.state.arena.initial_prompt = user_input
        self.state.arena.last_message = user_input
        self.ui.arena_paused_event.set()  # Start in a running state.

        await self._orchestrate()

    async def _orchestrate(self):
        original_endpoint_name = self.client.config.name
        request = None
        try:
            if not (state := self.state.arena):
                return
            while state.current_speech < state.max_speeches:
                await self.ui.arena_paused_event.wait()
                participant = state.arena_config.participants[state.turn_order_ids[0]]
                turn_num = (state.current_speech // len(state.turn_order_ids)) + 1
                if self.client.config.name.lower() != participant.endpoint.lower():
                    self.client.switch_endpoint(participant.endpoint)
                self.client.config.model_name = participant.model
                self.pt_printer(
                    HTML(
                        f"\n<style bg='ansiblue' fg='white'> 🥊 TURN {turn_num} | {participant.name} ({participant.model}) </style>"
                    )
                )

                # --- Build message history based on conversation style ---
                if (
                    state.arena_config.conversation_style
                    == ArenaConversationStyle.PAIRWISE
                ):
                    # Previous message is presented as from the 'user'
                    messages = participant.conversation.get_messages_for_next_turn(
                        state.last_message
                    )
                else:  # CHATROOM mode
                    # Build a single string transcript of the whole conversation
                    transcript_lines = []
                    if state.initial_prompt:
                        transcript_lines.append(f"User: {state.initial_prompt}")
                    for turn in self.conversation.turns:
                        if p_name := turn.participant_name:
                            transcript_lines.append(
                                f"{p_name}: {turn.assistant_message}"
                            )
                    transcript = "\n".join(transcript_lines)

                    # Augment system prompt to give the model its identity
                    system_prompt_with_identity = (
                        f"{participant.system_prompt}\n\n"
                        f"You are participating in a group discussion as '{participant.name}'. "
                        "Read the transcript and provide your next response."
                    )
                    user_prompt_with_transcript = (
                        "Below is the transcript of the conversation so far. "
                        f"Provide your response as '{participant.name}'.\n\n"
                        f"--- TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---"
                    )

                    messages = [
                        {"role": "system", "content": system_prompt_with_identity},
                        {"role": "user", "content": user_prompt_with_transcript},
                    ]
                request = ChatRequest(
                    messages=messages,
                    model=participant.model,
                    max_tokens=self.runtime_config.max_tokens,
                    temperature=self.runtime_config.temperature,
                    stream=self.runtime_config.stream,
                    tools=get_tool_schemas() if self.client.tools_enabled else [],
                )

                # --- Wildcard Logic ---
                if state.arena_config.wildcards_enabled and random.random() < 0.33:
                    wildcard_fn = random.choice(
                        [self._wildcard_temp_spike, self._wildcard_prompt_injection]
                    )
                    effect_message = ""
                    if wildcard_fn.__name__ == "_wildcard_temp_spike":
                        request, effect_message = self._wildcard_temp_spike(request)
                    else:  # prompt injection
                        messages, effect_message = self._wildcard_prompt_injection(
                            messages
                        )
                        # Re-assign messages to the request object
                        request.messages = messages

                    self.pt_printer(
                        HTML(
                            f"  - <style fg='ansimagenta'><b>{effect_message}</b></style>"
                        )
                    )
                # --- End Wildcard Logic ---

                result = await self.client.generate(
                    request,
                    self.runtime_config.verbose,
                    actor_name=f"🤖 {participant.name}",
                )
                assistant_message = result.get("text", "")
                assistant_reasoning = result.get("reasoning")
                response_data = result.get("response", {})

                final_assistant_message = assistant_message
                if self.runtime_config.keep_reasoning and assistant_reasoning:
                    final_assistant_message = f"<thinking>\n{assistant_reasoning}\n</thinking>\n{assistant_message}"
                    # Also update the response data that gets logged for history persistence
                    if rd_choices := response_data.get("choices"):
                        if rd_choices and rd_choices[0].get("message"):
                            rd_choices[0]["message"]["content"] = (
                                final_assistant_message
                            )

                turn = Turn(
                    request_data=result.get("request", {}),
                    response_data=response_data,
                    assistant_message=final_assistant_message,
                    assistant_reasoning=assistant_reasoning,
                    participant_name=participant.name,
                    model_name=participant.model,
                    mode=self.state.mode,
                    stats=self.client.stats.last_request_stats,
                    endpoint_name=participant.endpoint,
                )
                stats = self.client.stats.last_request_stats
                self.conversation.add_turn(turn, stats)
                participant.conversation.add_turn(turn, stats)
                save_conversation_formats(
                    self.conversation, self.log_dir, self.pt_printer, arena_state=state
                )
                state.last_message = assistant_message
                state.current_speech += 1
                # --- New Turn Order Logic ---
                if len(state.turn_order_ids) > 1:
                    if state.arena_config.turn_order == ArenaTurnOrder.RANDOM:
                        # Pop the current speaker (at index 0)
                        current_speaker = state.turn_order_ids.pop(0)
                        # Shuffle the rest of the participants
                        random.shuffle(state.turn_order_ids)
                        # Add the current speaker to the end of the shuffled list
                        state.turn_order_ids.append(current_speaker)
                    else:  # SEQUENTIAL (the default)
                        state.turn_order_ids = (
                            state.turn_order_ids[1:] + state.turn_order_ids[:1]
                        )
            self.pt_printer("\n🏁 Arena finished: Maximum turns reached.")
            await self._run_arena_judge()
        except asyncio.CancelledError:
            request_stats = await self.client.display.finish_response(success=False)
            partial_text = self.client.display.current_response
            if request_stats and request and partial_text:
                request_stats.finish_reason = "cancelled"
                request_stats.tokens_sent = sum(
                    estimate_tokens(m.get("content", "")) for m in request.messages
                )
                self.client.stats.add_completed_request(request_stats)
                self._log_cancelled_turn(request, partial_text)
            self.pt_printer(
                HTML("\n<style fg='ansiyellow'>🚫 Arena cancelled by user.</style>")
            )
        except Exception as e:
            if self.client.display.current_request_stats:
                await self.client.display.finish_response(success=False)
            self.pt_printer(
                HTML(
                    f"<style fg='ansired'>❌ ARENA ERROR: {escape(str(e), quote=False)}</style>"
                )
            )
            if request:
                # Log a failed turn to preserve history so the user's prompt isn't lost.
                request_data = request.to_dict(self.client.config.model_name)
                error_text = f"Error during generation: {str(e)}"
                response_data = {"error": {"message": error_text}}
                turn = Turn(
                    request_data=request_data,
                    response_data=response_data,
                    assistant_message=error_text,
                    mode=self.state.mode,
                    stats=self.client.stats.last_request_stats,
                    # We may not have a participant here if the error was early.
                    # Fallback to the client's current endpoint.
                    endpoint_name=participant.endpoint
                    if "participant" in locals()
                    else self.client.config.name,
                )
                self.conversation.add_turn(turn, self.client.stats.last_request_stats)
                try:
                    save_conversation_formats(
                        self.conversation,
                        self.log_dir,
                        printer=self.pt_printer,
                        arena_state=self.state.arena,
                    )
                except Exception as log_e:
                    self.pt_printer(
                        f"\n⚠️  Warning: Could not save failed session turn: {log_e}"
                    )
        finally:
            if self.state.arena:
                await self._run_arena_judge()
                self.pt_printer(
                    f"\n🏁 Arena '{self.state.arena.arena_config.name}' session ended."
                )
            if self.client.config.name.lower() != original_endpoint_name.lower():
                self.pt_printer(
                    f"✅ Restoring client to original endpoint: {original_endpoint_name}"
                )
                self.client.switch_endpoint(original_endpoint_name)
            self.ui.enter_mode(UIMode.CHAT, clear_history=False)
            self.ui.active_concurrent_count = 0
            self.ui.generation_in_progress.clear()
            self.ui.generation_task = None

    async def _run_arena_judge(self):
        if not self.state.arena or not self.state.arena.arena_config.judge:
            return
        judge = self.state.arena.arena_config.judge
        self.pt_printer(
            HTML(
                "\n<style bg='ansiyellow' fg='black'> 🧑‍⚖️ The Judge is now deliberating... </style>"
            )
        )
        messages = judge.conversation.get_history()
        full_dialogue = "\n\n".join(
            [
                f"{t.participant_name or 'user'}: {t.assistant_message or t.request_data.get('messages', [{}])[-1].get('content', '')}"
                for t in self.conversation.turns
            ]
        )
        messages.append(
            {
                "role": "user",
                "content": f"Here is the full conversation transcript:\n\n---\n{full_dialogue}\n---\n\nPlease provide your summary and verdict.",
            }
        )
        if self.client.config.name.lower() != judge.endpoint.lower():
            self.client.switch_endpoint(judge.endpoint)
        self.client.config.model_name = judge.model
        request = ChatRequest(
            messages=messages,
            model=judge.model,
            max_tokens=self.runtime_config.max_tokens,
            temperature=self.runtime_config.temperature,
            stream=self.runtime_config.stream,
        )
        result = await self.client.generate(
            request, self.runtime_config.verbose, actor_name=f"🧑‍⚖️ {judge.name}"
        )
        turn = Turn(
            request_data=result.get("request", {}),
            response_data=result.get("response", {}),
            assistant_message=result.get("text", ""),
            participant_name=judge.name,
            model_name=judge.model,
            mode=self.state.mode,
            stats=self.client.stats.last_request_stats,
            endpoint_name=judge.endpoint,
        )
        self.conversation.add_turn(turn, self.client.stats.last_request_stats)
        save_conversation_formats(
            self.conversation,
            self.log_dir,
            printer=self.pt_printer,
            arena_state=self.state.arena,
        )
