import asyncio
import json
import logging
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
            except json.JSONDecodeError as e:
                # Handle malformed JSON arguments from the model.
                failed_content = tool_call["function"]["arguments"]
                error_msg = (
                    f"Error: Model provided invalid JSON for tool '{name}' args. "
                    f"Error: {e}. Content: {failed_content!r}"
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

            context.display.show_agent_tool_call(name, args)
            try:
                result = await execute_tool(name, args)
                tools_used_count += 1
                context.display.show_agent_tool_result(name, result)
                return result
            except (ToolNotFound, ToolArgumentError, ToolError) as e:
                error_payload = {"status": "failure", "reason": str(e)}
                context.display.show_agent_tool_result(name, json.dumps(error_payload))
                return f"Error: {e}"

        for iteration in range(max_iterations):
            logging.info(f"AGENT: OpenAI adapter starting iteration {iteration + 1}/{max_iterations}.")
            finish_reason = None
            url = f"{context.config.base_url}/chat/completions"
            # The request object now correctly includes tools in its dictionary representation.
            # We just need to ensure the messages for the current agentic loop are set.
            payload = request.to_dict(context.config.model_name)
            payload["messages"] = messages
            final_request_payload = payload
            if context.display.debug_mode:
                debug_payload_str = json.dumps(
                    final_request_payload, indent=2, ensure_ascii=False
                )
                log_line = f"🔵 DEBUG: REQUEST PAYLOAD\n{debug_payload_str}"
                context.display._print(log_line)
                logging.info(log_line)

            tokens_sent = sum(estimate_tokens(m.get("content", "")) for m in messages)

            try:
                # Must start the response *before* printing the agent loop message
                await context.display.start_response(
                    tokens_sent=tokens_sent,
                    actor_name=actor_name,
                    model_name=request.model or context.config.model_name,
                    is_continuation=iteration > 0,
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
                    try:
                        response_data = response.json()
                    except json.JSONDecodeError as e:
                        raise ConnectionError(
                            f"Failed to decode JSON response: {e}. Content: {response.text!r}"
                        ) from e

                    if (
                        response_data.get("object") == "error"
                        or "error" in response_data
                    ):
                        error_details = response_data.get("error", {})
                        error_message = error_details.get(
                            "message", "Unknown API error"
                        )
                        pretty_details = json.dumps(error_details, indent=2)
                        raise ValueError(
                            f"API error: {error_message}\n{pretty_details}"
                        )

                    choice = response_data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    finish_reason = choice.get("finish_reason")

                    if tool_calls_data := message.get("tool_calls"):
                        logging.info(
                            f"AGENT: Model requested {len(tool_calls_data)} tool calls (non-streaming). Executing now."
                        )
                        # If the model provides text before the tool call, render it.
                        if leading_text := message.get("content"):
                            await context.display.show_parsed_chunk(
                                response_data, leading_text
                            )

                        messages.append(message)
                        tasks = [
                            _execute_and_format_tool_call(tc) for tc in tool_calls_data
                        ]
                        tool_results = await asyncio.gather(*tasks)
                        messages.extend(tool_results)
                        logging.info(
                            "AGENT: Finished executing tool calls (non-streaming). Preparing for next iteration."
                        )
                        continue  # Next agent iteration

                    # No tool calls, regular response
                    final_text = message.get("content", "")
                    final_reasoning = message.get("reasoning")
                    await context.display.show_parsed_chunk(
                        response_data, content=final_text or "", reasoning=final_reasoning
                    )
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
                        "reasoning": context.display.current_reasoning,
                        "tools_used": tools_used_count,
                        "agent_loops": iteration + 1,
                    }

                # Streaming logic starts here
                tool_calls = []
                final_usage = {}

                async with context.http_session.stream(
                    "POST", url, json=payload, timeout=context.config.timeout
                ) as response:
                    if response.is_error:
                        await response.aread()
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

                                    if (
                                        chunk_data.get("object") == "error"
                                        or "error" in chunk_data
                                    ):
                                        error_details = chunk_data.get("error", {})
                                        error_message = error_details.get(
                                            "message", "Unknown streaming error"
                                        )
                                        pretty_details = json.dumps(
                                            error_details, indent=2
                                        )
                                        raise ValueError(
                                            f"Streaming error from provider: {error_message}\n{pretty_details}"
                                        )

                                    # Some providers (e.g., Groq) send a final usage chunk.
                                    if usage := chunk_data.get("usage"):
                                        final_usage = usage

                                    choice = chunk_data.get("choices", [{}])[0]
                                    delta = choice.get("delta", {})
                                    content = delta.get("content")
                                    reasoning = delta.get("reasoning")

                                    # Always pass the chunk to the display. It handles diffing
                                    # and decides whether to render any text content.
                                    await context.display.show_parsed_chunk(
                                        chunk_data, content=content or "", reasoning=reasoning
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
                                    context.display._print(
                                        f"⚠️  Stream parse error on line: {data!r} | Error: {e}"
                                    )
                                    continue

                if tool_calls:
                    logging.info(
                        f"AGENT: Model requested {len(tool_calls)} tool calls (streaming). Executing now."
                    )
                    # The stream has ended and a tool call is requested.
                    # The `reasoning: null` signal in the stream should have already
                    # committed any reasoning block. We now just commit any partial text.
                    partial_text = context.display.current_response
                    await context.display.commit_partial_response()

                    # Now, message history *must* be updated with the partial text *and* the tool call.
                    # This ensures the model sees its own preceding text.
                    assistant_message = {
                        "role": "assistant",
                        "content": partial_text or None,
                        "tool_calls": tool_calls,
                    }
                    messages.append(assistant_message)

                    tasks = [_execute_and_format_tool_call(tc) for tc in tool_calls]
                    tool_results = await asyncio.gather(*tasks)
                    messages.extend(tool_results)
                    logging.info(
                        "AGENT: Finished executing tool calls (streaming). Preparing for next iteration."
                    )
                    continue

                request_stats = await context.display.finish_response(
                    success=True, usage=final_usage
                )
                if request_stats:
                    if request_stats.cost:
                        request_stats.cost.update(
                            input_tokens=request_stats.tokens_sent,
                            output_tokens=request_stats.tokens_received,
                        )
                    request_stats.finish_reason = finish_reason
                    context.stats.add_completed_request(request_stats)

                # Construct a response object that mimics the non-streaming API
                final_tokens_sent = (
                    request_stats.tokens_sent if request_stats else tokens_sent
                )
                final_tokens_received = (
                    request_stats.tokens_received if request_stats else 0
                )
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
                    "usage": final_usage
                    or {
                        "prompt_tokens": final_tokens_sent,
                        "completion_tokens": final_tokens_received,
                        "total_tokens": final_tokens_sent + final_tokens_received,
                    },
                }

                final_request_payload["messages"] = messages
                return {
                    "request": final_request_payload,
                    "response": response_data,
                    "text": context.display.current_response,
                    "reasoning": context.display.current_reasoning,
                    "tools_used": tools_used_count,
                    "agent_loops": iteration + 1,
                }
            except httpx.HTTPStatusError as e:
                request_stats = await context.display.finish_response(success=False)
                if request_stats:
                    request_stats.tokens_sent = tokens_sent
                    context.stats.add_completed_request(request_stats)
                error_message = ""
                try:
                    # Attempt to get a readable text response.
                    error_message = e.response.text
                except Exception as decoding_error:
                    # If reading as text fails, report the raw content and the decoding error.
                    error_message = (
                        f"[Error decoding response body: {decoding_error!r}] "
                        f"Raw content: {e.response.content!r}"
                    )
                raise ConnectionError(
                    f"Request failed with status {e.response.status_code}: {error_message}"
                ) from e
            except httpx.RequestError as e:
                request_stats = await context.display.finish_response(success=False)
                if request_stats:
                    request_stats.tokens_sent = tokens_sent
                    context.stats.add_completed_request(request_stats)
                # This catches timeouts, connection errors, etc.
                error_message = (
                    f"Network request to {e.request.url} failed. "
                    f"Error: {e.__class__.__name__}: {e}"
                )
                raise ConnectionError(error_message) from e
            except Exception as e:
                # On failure, finalize stats as unsuccessful and re-raise
                request_stats = await context.display.finish_response(success=False)
                if request_stats:
                    request_stats.tokens_sent = tokens_sent
                    context.stats.add_completed_request(request_stats)
                raise ConnectionError(f"Request failed: {e}")

        return {
            "text": "[Agent Error] Agent reached maximum iterations.",
            "tools_used": tools_used_count,
            "agent_loops": max_iterations,
        }
