import asyncio
import json
import logging
from typing import Any

import httpx

from ..models import ChatRequest, RequestCost
from ..tools import ToolArgumentError, ToolError, ToolNotFound, execute_tool
from ..utils import estimate_tokens
from .base_adapter import BaseProtocolAdapter, ProtocolContext


class OllamaAdapter(BaseProtocolAdapter):
    """Handles the Ollama /api/chat endpoint format, including tool use."""

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
            """Executes a single tool call and formats the result for Ollama."""
            name = tool_call["function"]["name"]
            args_raw = tool_call["function"]["arguments"]

            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                result = await _execute_with_confirmation(name, args)
            except json.JSONDecodeError as e:
                result = f"Error: Model provided invalid JSON arguments for tool '{name}': {e}. Content: {args_raw!r}"
                context.display._print(f"  - ❌ Tool Error: {result}")

            return {"role": "tool", "content": str(result)}

        async def _execute_with_confirmation(name: str, args: dict) -> Any:
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
            url = f"{context.config.base_url}/chat"
            payload = request.to_dict(context.config.model_name)
            payload["messages"] = messages
            # Only send tools if enabled and present
            if not (context.tools_enabled and request.tools):
                payload.pop("tools", None)
                payload.pop("tool_choice", None)

            if context.display.debug_mode:
                debug_payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
                log_line = f"🔵 DEBUG: REQUEST PAYLOAD\n{debug_payload_str}"
                context.display._print(log_line)
                logging.info(log_line)

            # Ollama expects "stream": false for non-streaming
            if "stream" not in payload:
                payload["stream"] = False

            final_request_payload = payload
            tokens_sent = sum(estimate_tokens(m.get("content", "")) for m in messages)

            try:
                context.display.start_response(
                    tokens_sent=tokens_sent,
                    actor_name=actor_name,
                    model_name=request.model or context.config.model_name,
                )
                model_pricing = context.pricing_service.get_model_pricing(
                    context.config.name, request.model or context.config.model_name
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

                # Non-streaming
                if not payload.get("stream"):
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

                    if error_message := response_data.get("error"):
                        pretty_details = json.dumps(response_data, indent=2)
                        raise ValueError(f"API error: {error_message}\n{pretty_details}")

                    message = response_data.get("message", {})

                    if tool_calls := message.get("tool_calls"):
                        context.display._print(
                            f"\n🔧 [Agent Action] Model requested {len(tool_calls)} tool calls..."
                        )
                        messages.append(message)  # Add assistant's tool call request
                        tasks = [_execute_and_format_tool_call(tc) for tc in tool_calls]
                        tool_results = await asyncio.gather(*tasks)
                        messages.extend(tool_results)
                        continue  # Next agent iteration

                    final_text = message.get("content", "")
                    await context.display.show_parsed_chunk(response_data, final_text)
                    request_stats = await context.display.finish_response(success=True)
                    if request_stats:
                        request_stats.tokens_sent = response_data.get(
                            "prompt_eval_count", tokens_sent
                        )
                        request_stats.tokens_received = response_data.get(
                            "eval_count", 0
                        )
                        if request_stats.cost:
                            request_stats.cost.update(
                                input_tokens=request_stats.tokens_sent,
                                output_tokens=request_stats.tokens_received,
                            )
                        context.stats.add_completed_request(request_stats)

                    final_request_payload["messages"] = messages
                    return {
                        "request": final_request_payload,
                        "response": response_data,
                        "text": context.display.current_response,
                        "tools_used": tools_used_count,
                        "agent_loops": iteration + 1,
                    }

                # Streaming
                final_response_object = {}
                async with context.http_session.stream(
                    "POST", url, json=payload, timeout=context.config.timeout
                ) as response:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        # ... streaming logic ...
                        # This part needs to accumulate tool calls from chunks,
                        # similar to the openai adapter. Since it's complex,
                        # for now, we'll focus on the non-streaming case.
                        # The streaming logic from the original implementation
                        # doesn't handle tool calls, so it needs a full rewrite
                        # which is out of scope for a single change.
                        # We will keep the original streaming logic for text responses.
                        if not line:
                            continue
                        context.display.show_raw_line(line)
                        try:
                            chunk_data = json.loads(line)

                            # Check for a streaming error object.
                            if error_message := chunk_data.get("error"):
                                pretty_details = json.dumps(chunk_data, indent=2)
                                raise ValueError(
                                    f"Streaming error from provider: {error_message}\n{pretty_details}"
                                )

                            if chunk_data.get("done"):
                                final_response_object = chunk_data
                                break

                            message_chunk = chunk_data.get("message", {})
                            if content := message_chunk.get("content"):
                                await context.display.show_parsed_chunk(
                                    chunk_data, content
                                )

                        except json.JSONDecodeError as e:
                            context.display._print(
                                f"⚠️  Stream parse error on line: {line!r} | Error: {e}"
                            )
                            continue

                request_stats = await context.display.finish_response(
                    success=True, usage=final_response_object
                )
                if request_stats:
                    request_stats.finish_reason = final_response_object.get(
                        "done_reason", "stop"
                    )
                    if request_stats.cost:
                        request_stats.cost.update(
                            input_tokens=request_stats.tokens_sent,
                            output_tokens=request_stats.tokens_received,
                        )
                    context.stats.add_completed_request(request_stats)

                payload["messages"] = messages
                return {
                    "request": payload,
                    "response": final_response_object,
                    "text": context.display.current_response,
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
                raise ConnectionError(f"Ollama request failed: {e}") from e

        return {
            "text": "[Agent Error] Agent reached maximum iterations.",
            "tools_used": tools_used_count,
            "agent_loops": max_iterations,
        }
