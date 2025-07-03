"""Orchestrator for multi-model arena mode."""

import asyncio
import random
from typing import Any

from prompt_toolkit.formatted_text import HTML

from ..log_utils import save_conversation_formats
from ..models import ArenaTurnOrder, ChatRequest, Turn, UIMode
from ..tools import get_tool_schemas
from ..utils import estimate_tokens
from .base import BaseOrchestrator


class ArenaOrchestrator(BaseOrchestrator):
    """Handles the arena loop and judging."""

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
                messages = participant.conversation.get_messages_for_next_turn(
                    state.last_message
                )
                request = ChatRequest(
                    messages=messages,
                    model=participant.model,
                    max_tokens=self.runtime_config.max_tokens,
                    temperature=self.runtime_config.temperature,
                    stream=self.runtime_config.stream,
                    tools=get_tool_schemas() if self.client.tools_enabled else [],
                )
                result = await self.client.generate(
                    request,
                    self.runtime_config.verbose,
                    actor_name=f"🤖 {participant.name}",
                )
                assistant_message = result.get("text", "")
                turn = Turn(
                    request_data=result.get("request", {}),
                    response_data=result.get("response", {}),
                    assistant_message=assistant_message,
                    participant_name=participant.name,
                    model_name=participant.model,
                )
                stats = self.client.stats.last_request_stats
                self.conversation.add_turn(turn, stats)
                participant.conversation.add_turn(turn, stats)
                save_conversation_formats(
                    self.conversation, self.log_dir, self.pt_printer
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
            self.pt_printer(HTML(f"<style fg='ansired'>❌ ARENA ERROR: {e}</style>"))
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
        )
        self.conversation.add_turn(turn, self.client.stats.last_request_stats)
        save_conversation_formats(
            self.conversation, self.log_dir, printer=self.pt_printer
        )
