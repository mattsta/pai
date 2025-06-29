import asyncio
import json
import time
from typing import Any

import httpx

from ..models import ChatRequest, RequestCost
from ..tools import ToolArgumentError, ToolError, ToolNotFound, execute_tool
from ..utils import estimate_tokens

# MODIFIED: Now uses the clean context object.
from .base_adapter import BaseProtocolAdapter, ProtocolContext


class OpenAIChatAdapter(BaseProtocolAdapter):
    """Handles any OpenAI-compatible /chat/completions endpoint."""

    async def generate(
        self,
        context: ProtocolContext,
        request: ChatRequest,
        verbose: bool,
        actor_name: str | None = None,
    ) -> dict[str, Any]:
        messages = list(request.messages)
        max_iterations = 5
        tools_used_count = 0

        async def _execute_and_format_tool_call(tool_call: dict) -> dict:
            """Executes a single tool call and formats the result."""
            name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                # Handle malformed JSON arguments from the model.
                error_msg = (
                    f"Error: Model provided invalid JSON arguments for tool '{name}'."
                )
                context.display._print(f"  - ❌ Tool Error: {error_msg}")
                return {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": name,
                    "content": error_msg,
                }

            result = await _execute_with_confirmation(name, args)
            return {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": name,
                "content": str(result),
            }

        async def _execute_with_confirmation(name: str, args: dict) -> Any:
            """Helper to wrap tool execution with an optional confirmation step."""
            nonlocal tools_used_count
            if context.confirmer:
                should_run = await context.confirmer(name, args)
                if not should_run:
                    return "Tool execution cancelled by user."
            else:
                context.display._print(
                    f"  - Executing: {name}({json.dumps(args, indent=2)})"
                )
            try:
                result = await execute_tool(name, args)
                tools_used_count += 1
                return result
            except (ToolNotFound, ToolArgumentError, ToolError) as e:
                context.display._print(f"  - ❌ Tool Error: {e}")
                return f"Error: {e}"

        for iteration in range(max_iterations):
            finish_reason = None
            url = f"{context.config.base_url}/chat/completions"
            # The request object now correctly includes tools in its dictionary representation.
            # We just need to ensure the messages for the current agentic loop are set.
            payload = request.to_dict(context.config.model_name)
            payload["messages"] = messages
            final_request_payload = payload

            tokens_sent = sum(estimate_tokens(m.get("content", "")) for m in messages)

            try:
                # Must start the response *before* printing the agent loop message
                context.display.start_response(
                    tokens_sent=tokens_sent, actor_name=actor_name
                )
                # Create and attach the cost tracker to the request stats
                model_pricing = context.pricing_service.get_model_pricing(
                    context.config.name, context.config.model_name
                )
                if stats := context.display.current_request_stats:
                    stats.cost = RequestCost(
                        _pricing_service=context.pricing_service,
                        _model_pricing=model_pricing,
                    )
                if iteration > 0:
                    context.display._print(
                        "\n🔄 [Agent Loop] Sending tool results back to model..."
                    )

                if not request.stream:
                    response = await context.http_session.post(
                        url, json=payload, timeout=context.config.timeout
                    )
                    response.raise_for_status()
                    response_data = response.json()

                    choice = response_data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    finish_reason = choice.get("finish_reason")

                    if tool_calls_data := message.get("tool_calls"):
                        context.display._print(
                            f"\n🔧 [Agent Action] Model requested {len(tool_calls_data)} tool calls..."
                        )
                        messages.append(message)
                        tasks = [
                            _execute_and_format_tool_call(tc) for tc in tool_calls_data
                        ]
                        tool_results = await asyncio.gather(*tasks)
                        messages.extend(tool_results)
                        continue  # Next agent iteration

                    # No tool calls, regular response
                    final_text = message.get("content", "")
                    await context.display.show_parsed_chunk(response_data, final_text)
                    request_stats = await context.display.finish_response(success=True)
                    if request_stats:
                        usage = response_data.get("usage", {})
                        input_tokens = usage.get("prompt_tokens", tokens_sent)
                        output_tokens = usage.get("completion_tokens", 0)
                        request_stats.tokens_sent = input_tokens
                        request_stats.tokens_received = output_tokens
                        if request_stats.cost:
                            request_stats.cost.update(
                                input_tokens=input_tokens, output_tokens=output_tokens
                            )
                        request_stats.finish_reason = finish_reason
                        context.stats.add_completed_request(request_stats)

                    final_request_payload["messages"] = messages
                    return {
                        "request": final_request_payload,
                        "response": response_data,
                        "text": context.display.current_response,
                        "tools_used": tools_used_count,
                        "agent_loops": iteration + 1,
                    }

                # Streaming logic starts here
                tool_calls = []

                async with context.http_session.stream(
                    "POST", url, json=payload, timeout=context.config.timeout
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            line_str = line
                            context.display.show_raw_line(line_str)
                            if line_str.startswith("data: "):
                                data = line_str[6:]
                                if data.strip() == "[DONE]":
                                    break
                                try:
                                    chunk_data = json.loads(data)
                                    choice = chunk_data.get("choices", [{}])[0]
                                    delta = choice.get("delta", {})

                                    if content := delta.get("content"):
                                        await context.display.show_parsed_chunk(
                                            chunk_data, content
                                        )
                                    if new_tool_calls := delta.get("tool_calls"):
                                        for tc in new_tool_calls:
                                            if len(tool_calls) <= tc["index"]:
                                                tool_calls.append(
                                                    {
                                                        "id": "",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "",
                                                            "arguments": "",
                                                        },
                                                    }
                                                )
                                            if tc.get("id"):
                                                tool_calls[tc["index"]]["id"] = tc["id"]
                                            if func := tc.get("function"):
                                                if name := func.get("name"):
                                                    tool_calls[tc["index"]]["function"][
                                                        "name"
                                                    ] += name
                                                if args := func.get("arguments"):
                                                    tool_calls[tc["index"]]["function"][
                                                        "arguments"
                                                    ] += args

                                    # Check for and store the finish reason for the choice.
                                    if reason := choice.get("finish_reason"):
                                        finish_reason = reason
                                except (json.JSONDecodeError, IndexError) as e:
                                    if context.display.debug_mode:
                                        context.display._print(
                                            f"⚠️  Stream parse error on line: {data!r} | Error: {e}"
                                        )
                                    continue

                if tool_calls:
                    context.display._print(
                        f"\n🔧 [Agent Action] Model requested {len(tool_calls)} tool calls..."
                    )
                    messages.append({"role": "assistant", "tool_calls": tool_calls})
                    tasks = [_execute_and_format_tool_call(tc) for tc in tool_calls]
                    tool_results = await asyncio.gather(*tasks)
                    messages.extend(tool_results)
                    continue

                request_stats = await context.display.finish_response(success=True)
                tokens_received = 0
                if request_stats:
                    request_stats.tokens_sent = tokens_sent
                    tokens_received = request_stats.tokens_received
                    if request_stats.cost:
                        request_stats.cost.update(
                            input_tokens=tokens_sent, output_tokens=tokens_received
                        )
                    request_stats.finish_reason = finish_reason
                    context.stats.add_completed_request(request_stats)

                # Construct a response object that mimics the non-streaming API
                response_data = {
                    "id": f"chatcmpl-pai-{time.time()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": context.config.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": context.display.current_response,
                            },
                            "finish_reason": finish_reason or "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": tokens_sent,
                        "completion_tokens": tokens_received,
                        "total_tokens": tokens_sent + tokens_received,
                    },
                }

                final_request_payload["messages"] = messages
                return {
                    "request": final_request_payload,
                    "response": response_data,
                    "text": context.display.current_response,
                    "tools_used": tools_used_count,
                    "agent_loops": iteration + 1,
                }
            except httpx.HTTPStatusError as e:
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
                # On failure, finalize stats as unsuccessful and re-raise
                request_stats = await context.display.finish_response(success=False)
                if request_stats:
                    request_stats.tokens_sent = tokens_sent
                    context.stats.add_completed_request(request_stats)
                raise ConnectionError(f"Request failed: {e!r}")

        return {
            "text": "[Agent Error] Agent reached maximum iterations.",
            "tools_used": tools_used_count,
            "agent_loops": max_iterations,
        }
