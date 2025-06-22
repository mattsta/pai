"""Orchestrator for legacy agent mode (text-based tool use)."""
import json
import re
from html import escape
from typing import Any

from prompt_toolkit.formatted_text import HTML

from ..log_utils import save_conversation_formats
from ..models import ChatRequest, Turn
from ..tools import execute_tool
from .base import BaseOrchestrator


class LegacyAgentOrchestrator(BaseOrchestrator):
    """Handles the legacy agent loop."""

    async def run(self, user_input: str | None = None) -> Any:
        if not user_input:
            self.ui.generation_in_progress.clear()
            self.ui.generation_task = None
            return

        confirmer = (
            self.ui._confirm_tool_call if self.runtime_config.confirm_tool_use else None
        )
        messages = self.conversation.get_messages_for_next_turn(user_input)
        max_loops = 5
        used_tools_in_loop = False

        try:
            for i in range(max_loops):
                request = ChatRequest(
                    messages=messages,
                    model=self.client.config.model_name,
                    max_tokens=self.runtime_config.max_tokens,
                    temperature=self.runtime_config.temperature,
                    stream=self.runtime_config.stream,
                    tools=[],
                )
                result = await self.client.generate(
                    request, self.runtime_config.verbose
                )
                assistant_response = result.get("text", "")
                messages.append({"role": "assistant", "content": assistant_response})
                tool_match = re.search(
                    r"<tool_call>(.*?)</tool_call>", assistant_response, re.DOTALL
                )

                if tool_match:
                    used_tools_in_loop = True
                    self.client.display.current_response = ""
                    tool_xml = tool_match.group(1)
                    name_match = re.search(r"<name>(.*?)</name>", tool_xml)
                    args_match = re.search(r"<args>(.*?)</args>", tool_xml, re.DOTALL)

                    if not name_match or not args_match:
                        tool_result = "Error: Invalid tool_call format."
                    else:
                        tool_name = name_match.group(1).strip()
                        tool_args_str = args_match.group(1).strip()
                        from ..tools import ToolArgumentError, ToolError, ToolNotFound

                        try:
                            tool_args = json.loads(tool_args_str)
                            if confirmer and not await confirmer(tool_name, tool_args):
                                tool_result = "Tool execution cancelled by user."
                            else:
                                tool_result = execute_tool(tool_name, tool_args)
                        except json.JSONDecodeError:
                            tool_result = (
                                f"Error: Invalid JSON in <args> for {tool_name}."
                            )
                        except (ToolNotFound, ToolArgumentError, ToolError) as e:
                            tool_result = f"Error: {e}"

                    tool_result_message = f"TOOL_RESULT:\n```\n{tool_result}\n```"
                    messages.append({"role": "user", "content": tool_result_message})
                    self.pt_printer(
                        HTML(f"  - Result: {escape(str(tool_result))[:300]}...")
                    )
                else:
                    if used_tools_in_loop:
                        self.pt_printer(HTML("\n✅ Agent formulated a response."))
                    turn = Turn(
                        request_data=result.get("request", {}),
                        response_data=result.get("response", {}),
                        assistant_message=assistant_response,
                    )
                    self.conversation.add_turn(
                        turn, self.client.stats.last_request_stats
                    )
                    save_conversation_formats(
                        self.conversation, self.session_dir, printer=self.pt_printer
                    )
                    break
            else:
                self.pt_printer("⚠️ Agent reached maximum loops.")
        except Exception as e:
            self.client.display.finish_response(success=False)
            self.pt_printer(HTML(f"❌ LEGACY AGENT ERROR: {e}"))
        finally:
            self.ui.generation_in_progress.clear()
            self.ui.generation_task = None
