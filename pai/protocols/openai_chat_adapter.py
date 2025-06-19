# protocols/openai_chat_adapter.py
import time
import json
from typing import Dict, Any

# MODIFIED: Now uses the clean context object.
from .base_adapter import BaseProtocolAdapter, ProtocolContext, ChatRequest
from ..tools import get_tool_schemas, execute_tool


class OpenAIChatAdapter(BaseProtocolAdapter):
    """Handles any OpenAI-compatible /chat/completions endpoint."""

    async def generate(
        self, context: ProtocolContext, request: ChatRequest, verbose: bool
    ) -> Dict[str, Any]:
        messages = list(request.messages)
        max_iterations = 5

        for iteration in range(max_iterations):
            finish_reason = None
            # MODIFIED: Access all state via the context object.
            url = f"{context.config.base_url}/chat/completions"
            final_request_payload = request.to_dict(context.config.model_name)
            final_request_payload["messages"] = messages
            payload = final_request_payload

            if context.tools_enabled and get_tool_schemas():
                payload["tools"] = get_tool_schemas()
                payload["tool_choice"] = "auto"

            tokens_sent = sum(len(m.get("content", "").split()) for m in messages)

            if not request.stream:
                raise NotImplementedError(
                    "Non-streaming chat is not implemented in this adapter."
                )

            try:
                if iteration > 0:
                    context.display._print("\n🔄 [Agent Loop] Sending tool results back to model...")

                context.display.start_response()
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
                                        context.display.show_parsed_chunk(
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
                                except (json.JSONDecodeError, IndexError):
                                    continue

                if tool_calls:
                    context.display._print("\n🔧 [Agent Action] Model requested tool calls...")
                    messages.append({"role": "assistant", "tool_calls": tool_calls})
                    for tool_call in tool_calls:
                        name = tool_call["function"]["name"]
                        args = json.loads(tool_call["function"]["arguments"])
                        context.display._print(f"  - Executing: {name}({args})")
                        result = execute_tool(name, args)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "name": name,
                                "content": str(result),
                            }
                        )
                    continue

                request_stats = context.display.finish_response(success=True)
                tokens_received = 0
                if request_stats:
                    request_stats.tokens_sent = tokens_sent
                    request_stats.finish_reason = finish_reason
                    tokens_received = request_stats.tokens_received
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

                return {
                    "request": final_request_payload,
                    "response": response_data,
                    "text": context.display.current_response,
                }

            except Exception as e:
                # On failure, finalize stats as unsuccessful and re-raise
                request_stats = context.display.finish_response(success=False)
                if request_stats:
                    request_stats.tokens_sent = tokens_sent
                    context.stats.add_completed_request(request_stats)
                raise ConnectionError(f"Request failed: {e!r}")

        return {"text": "[Agent Error] Agent reached maximum iterations."}
