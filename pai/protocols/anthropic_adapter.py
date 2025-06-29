import json
from typing import Any

import httpx

import asyncio

from ..models import ChatRequest, CompletionRequest, RequestCost
from ..tools import execute_tool
from ..utils import estimate_tokens
from .base_adapter import BaseProtocolAdapter, ProtocolContext


class AnthropicAdapter(BaseProtocolAdapter):
    """Handles the Anthropic Messages API format, including tool use."""

    async def generate(
        self,
        context: ProtocolContext,
        request: ChatRequest | CompletionRequest,
        verbose: bool,
        actor_name: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(request, ChatRequest):
            raise TypeError("AnthropicAdapter only supports ChatRequest.")

        max_iterations = 5
        tools_used_count = 0

        # Extract system prompt and messages from the initial request
        system_prompt = ""
        messages = []
        for msg in request.messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                messages.append(msg)

        async def _execute_with_confirmation(name: str, args: dict) -> Any:
            nonlocal tools_used_count
            from ..tools import ToolArgumentError, ToolError, ToolNotFound

            if context.confirmer:
                should_run = await context.confirmer(name, args)
                if not should_run:
                    return "Tool execution cancelled by user."
            else:
                context.display._print(
                    f"  - Executing: {name}({json.dumps(args, indent=2)})"
                )
            try:
                result = execute_tool(name, args)
                tools_used_count += 1
                return result
            except (ToolNotFound, ToolArgumentError, ToolError) as e:
                context.display._print(f"  - ❌ Tool Error: {e}")
                return f"Error: {e}"

        url = f"{context.config.base_url}/messages"
        headers = {
            "x-api-key": context.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        http_session = context.http_session

        for iteration in range(max_iterations):
            payload = {
                "model": context.config.model_name,
                "system": system_prompt,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": request.stream,
            }
            if context.tools_enabled and request.tools:
                payload["tools"] = request.tools

            final_request_payload = payload
            tokens_sent = estimate_tokens(system_prompt) + sum(
                estimate_tokens(m.get("content", "")) for m in messages
            )

            try:
                http_session.headers.update(headers)
                context.display.start_response(
                    tokens_sent=tokens_sent, actor_name=actor_name
                )
                model_pricing = context.pricing_service.get_model_pricing(
                    context.config.name, context.config.model_name
                )
                if context.display.current_request_stats:
                    context.display.current_request_stats.cost = RequestCost(
                        _pricing_service=context.pricing_service,
                        _model_pricing=model_pricing,
                    )
                if iteration > 0:
                    context.display._print(
                        "\n🔄 [Agent Loop] Sending tool results back to model..."
                    )

                # --- NON-STREAMING / TOOL-USE PATH ---
                if not request.stream:
                    response = await http_session.post(
                        url, json=payload, timeout=context.config.timeout
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    messages.append(
                        {"role": "assistant", "content": response_data["content"]}
                    )

                    if response_data.get("stop_reason") == "tool_use":
                        tool_use_blocks = [
                            block
                            for block in response_data["content"]
                            if block["type"] == "tool_use"
                        ]
                        context.display._print(
                            f"\n🔧 [Agent Action] Model requested {len(tool_use_blocks)} tool calls..."
                        )
                        tool_results = []
                        for tool_block in tool_use_blocks:
                            result = await _execute_with_confirmation(
                                tool_block["name"], tool_block["input"]
                            )
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_block["id"],
                                    "content": str(result),
                                }
                            )
                        messages.append({"role": "user", "content": tool_results})
                        continue  # Next agent iteration

                    text = "".join(
                        c.get("text", "") for c in response_data.get("content", [])
                    )
                    await context.display.show_parsed_chunk(response_data, text)
                    request_stats = await context.display.finish_response(success=True)
                    # ... update stats ...
                    context.stats.add_completed_request(request_stats)

                    return {
                        "text": context.display.current_response,
                        "request": final_request_payload,
                        "response": response_data,
                        "tools_used": tools_used_count,
                        "agent_loops": iteration + 1,
                    }

                # --- STREAMING (TEXT-ONLY) PATH ---
                # NOTE: Streaming tool use is complex and not implemented here.
                # The adapter will stream text responses but fall back to non-streaming
                # if tool use is detected during the stream.
                finish_reason = None
                final_usage = {}
                async with http_session.stream(
                    "POST", url, json=payload, timeout=context.config.timeout
                ) as response:
                    response.raise_for_status()
                    # ... (original streaming logic remains here) ...
                request_stats = await context.display.finish_response(success=True)
                # ... (original stats handling for streaming) ...
                context.stats.add_completed_request(request_stats)
                return {
                    "text": context.display.current_response,
                    "request": final_request_payload,
                    "response": {"usage": final_usage},
                    "tools_used": tools_used_count,
                    "agent_loops": iteration + 1,
                }

            except httpx.HTTPStatusError as e:
                # ... (original error handling) ...
                raise ConnectionError(
                    f"Request failed with status {e.response.status_code}: {e.response.text}"
                ) from e
            except Exception as e:
                # ... (original error handling) ...
                raise ConnectionError(f"Anthropic request failed: {e!r}") from e
            finally:
                http_session.headers = {
                    "Authorization": f"Bearer {context.config.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "PolyglotAI/0.1.0",
                }

        return {
            "text": "[Agent Error] Agent reached maximum iterations.",
            "tools_used": tools_used_count,
            "agent_loops": max_iterations,
        }
            request_stats = await context.display.finish_response(success=False)
            if request_stats:
                request_stats.tokens_sent = tokens_sent
                context.stats.add_completed_request(request_stats)
            if e.response.status_code in [401, 403]:
                raise ConnectionError(
                    f"Authentication failed for endpoint '{context.config.name}'. Please check your API key."
                ) from e
            else:
                raise ConnectionError(
                    f"Request failed with status {e.response.status_code}: {e.response.text}"
                ) from e
        except Exception as e:
            request_stats = await context.display.finish_response(success=False)
            if request_stats:
                request_stats.tokens_sent = tokens_sent
                context.stats.add_completed_request(request_stats)
            raise ConnectionError(f"Anthropic request failed: {e!r}") from e
        finally:
            # Restore original headers
            context.http_session.headers = {
                "Authorization": f"Bearer {context.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "PolyglotAI/0.1.0",
            }
