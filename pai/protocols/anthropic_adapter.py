import asyncio
import json
import logging
from typing import Any

import httpx

from ..models import ChatRequest, CompletionRequest, RequestCost
from ..tools import ToolArgumentError, ToolError, ToolNotFound, execute_tool
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
            if context.display.debug_mode:
                debug_payload_str = json.dumps(
                    final_request_payload, indent=2, ensure_ascii=False
                )
                log_line = f"🔵 DEBUG: REQUEST PAYLOAD\n{debug_payload_str}"
                context.display._print(log_line)
                logging.info(log_line)
            tokens_sent = estimate_tokens(system_prompt) + sum(
                estimate_tokens(m.get("content", "")) for m in messages
            )

            try:
                http_session.headers.update(headers)
                context.display.start_response(
                    tokens_sent=tokens_sent,
                    actor_name=actor_name,
                    model_name=context.config.model_name,
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
                    try:
                        response_data = response.json()
                    except json.JSONDecodeError as e:
                        raise ConnectionError(
                            f"Failed to decode JSON response: {e}. Content: {response.text!r}"
                        ) from e

                    if response_data.get("type") == "error":
                        error_details = response_data.get("error", {})
                        error_message = error_details.get(
                            "message", "Unknown API error"
                        )
                        pretty_details = json.dumps(error_details, indent=2)
                        raise ValueError(
                            f"API error: {error_message}\n{pretty_details}"
                        )

                    messages.append(
                        {"role": "assistant", "content": response_data["content"]}
                    )

                    if response_data.get("stop_reason") == "tool_use":
                        # If the model provides text before the tool call, render it.
                        if text_blocks := [
                            block
                            for block in response_data.get("content", [])
                            if block["type"] == "text"
                        ]:
                            leading_text = "".join(b["text"] for b in text_blocks)
                            await context.display.show_parsed_chunk(
                                response_data, leading_text
                            )

                        tool_use_blocks = [
                            block
                            for block in response_data["content"]
                            if block["type"] == "tool_use"
                        ]
                        tasks = [
                            _execute_with_confirmation(
                                tool_block["name"], tool_block["input"]
                            )
                            for tool_block in tool_use_blocks
                        ]
                        results = await asyncio.gather(*tasks)

                        tool_results_content = [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_block["id"],
                                "content": str(result),
                            }
                            for tool_block, result in zip(tool_use_blocks, results)
                        ]
                        messages.append(
                            {"role": "user", "content": tool_results_content}
                        )
                        continue  # Next agent iteration

                    text = "".join(
                        c.get("text", "") for c in response_data.get("content", [])
                    )
                    await context.display.show_parsed_chunk(response_data, text)
                    request_stats = await context.display.finish_response(success=True)
                    if request_stats:
                        usage = response_data.get("usage", {})
                        input_tokens = usage.get("input_tokens", tokens_sent)
                        output_tokens = usage.get("output_tokens", 0)
                        request_stats.tokens_sent = input_tokens
                        request_stats.tokens_received = output_tokens
                        if request_stats.cost:
                            request_stats.cost.update(
                                input_tokens=input_tokens, output_tokens=output_tokens
                            )
                        request_stats.finish_reason = response_data.get("stop_reason")
                        context.stats.add_completed_request(request_stats)

                    final_request_payload["messages"] = messages
                    return {
                        "text": context.display.current_response,
                        "request": final_request_payload,
                        "response": response_data,
                        "tools_used": tools_used_count,
                        "agent_loops": iteration + 1,
                    }

                # --- STREAMING (TEXT-ONLY) PATH ---
                finish_reason = None
                final_usage = {}
                async with http_session.stream(
                    "POST", url, json=payload, timeout=context.config.timeout
                ) as response:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        line_str = line.strip()
                        context.display.show_raw_line(line_str)
                        if line_str.startswith("data:"):
                            chunk_str = line_str[6:]
                            try:
                                chunk = json.loads(chunk_str)
                                event_type = chunk.get("type")

                                if event_type == "error":
                                    error_details = chunk.get("error", {})
                                    error_message = error_details.get(
                                        "message", "Unknown streaming error"
                                    )
                                    pretty_details = json.dumps(error_details, indent=2)
                                    raise ValueError(
                                        f"Streaming error from provider: {error_message}\n{pretty_details}"
                                    )

                                if event_type == "message_start":
                                    if stats := context.display.current_request_stats:
                                        usage = chunk.get("message", {}).get(
                                            "usage", {}
                                        )
                                        input_tokens = usage.get(
                                            "input_tokens", tokens_sent
                                        )
                                        stats.tokens_sent = input_tokens
                                        if stats.cost:
                                            stats.cost.update(input_tokens=input_tokens)
                                elif event_type == "content_block_delta":
                                    chunk_text = chunk.get("delta", {}).get("text", "")
                                    await context.display.show_parsed_chunk(
                                        chunk, chunk_text
                                    )
                                elif event_type == "message_delta":
                                    # This event contains the final stop reason and usage stats.
                                    # It's a critical event for debugging and must be shown.
                                    await context.display.show_parsed_chunk(chunk, "")

                                    delta = chunk.get("delta", {})
                                    finish_reason = delta.get("stop_reason")
                                    # Anthropic streaming tool use is complex. If detected, we note it.
                                    if finish_reason == "tool_use":
                                        context.display._print(
                                            "\n⚠️  Tool use detected in stream. Please use non-streaming mode (`/stream off`) for agentic tasks with Anthropic."
                                        )
                                    final_usage = chunk.get("usage", {})
                            except (json.JSONDecodeError, IndexError) as e:
                                context.display._print(
                                    f"⚠️  Stream parse error: {e} on line: {chunk_str!r}"
                                )

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

                final_request_payload["messages"] = messages
                return {
                    "text": context.display.current_response,
                    "request": final_request_payload,
                    "response": {"usage": final_usage},
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
            except Exception as e:
                request_stats = await context.display.finish_response(success=False)
                if request_stats:
                    request_stats.tokens_sent = tokens_sent
                    context.stats.add_completed_request(request_stats)
                raise ConnectionError(f"Anthropic request failed: {e}") from e

        return {
            "text": "[Agent Error] Agent reached maximum iterations.",
            "tools_used": tools_used_count,
            "agent_loops": max_iterations,
        }
