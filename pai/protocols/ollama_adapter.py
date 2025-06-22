import json
from typing import Any

from ..models import ChatRequest
from ..utils import estimate_tokens
from .base_adapter import BaseProtocolAdapter, ProtocolContext


class OllamaAdapter(BaseProtocolAdapter):
    """Handles the Ollama /api/chat endpoint format."""

    async def generate(
        self,
        context: ProtocolContext,
        request: ChatRequest,
        verbose: bool,
        actor_name: str | None = None,
    ) -> dict[str, Any]:
        url = f"{context.config.base_url}/chat"
        payload = request.to_dict(context.config.model_name)
        # Ollama doesn't use the tools key in this simple version
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

        # Ollama expects "stream": false for non-streaming
        if "stream" not in payload:
            payload["stream"] = False

        tokens_sent = sum(
            estimate_tokens(m.get("content", "")) for m in request.messages
        )

        try:
            context.display.start_response(
                tokens_sent=tokens_sent, actor_name=actor_name
            )

            # Non-streaming implementation
            if not payload["stream"]:
                response = await context.http_session.post(
                    url, json=payload, timeout=context.config.timeout
                )
                response.raise_for_status()
                response_data = response.json()
                message = response_data.get("message", {})
                final_text = message.get("content", "")
                await context.display.show_parsed_chunk(response_data, final_text)

                request_stats = await context.display.finish_response(success=True)
                if request_stats:
                    request_stats.tokens_sent = response_data.get(
                        "prompt_eval_count", tokens_sent
                    )
                    request_stats.tokens_received = response_data.get(
                        "eval_count", request_stats.tokens_received
                    )
                    context.stats.add_completed_request(request_stats)

                return {
                    "request": payload,
                    "response": response_data,
                    "text": context.display.current_response,
                }

            # Streaming implementation
            final_response_object = {}
            async with context.http_session.stream(
                "POST", url, json=payload, timeout=context.config.timeout
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    context.display.show_raw_line(line)
                    try:
                        chunk_data = json.loads(line)
                        if chunk_data.get("done"):
                            final_response_object = chunk_data
                            break

                        message_chunk = chunk_data.get("message", {})
                        if content := message_chunk.get("content"):
                            await context.display.show_parsed_chunk(
                                chunk_data, content
                            )

                    except json.JSONDecodeError:
                        if context.display.debug_mode:
                            context.display._print(
                                f"⚠️  Stream parse error on line: {line!r}"
                            )
                        continue

            request_stats = await context.display.finish_response(success=True)
            if request_stats:
                request_stats.tokens_sent = final_response_object.get(
                    "prompt_eval_count", tokens_sent
                )
                request_stats.tokens_received = final_response_object.get(
                    "eval_count", request_stats.tokens_received
                )
                request_stats.finish_reason = final_response_object.get(
                    "done_reason", "stop"
                )
                context.stats.add_completed_request(request_stats)

            return {
                "request": payload,
                "response": final_response_object,
                "text": context.display.current_response,
            }

        except Exception as e:
            request_stats = await context.display.finish_response(success=False)
            if request_stats:
                request_stats.tokens_sent = tokens_sent
                context.stats.add_completed_request(request_stats)
            raise ConnectionError(f"Ollama request failed: {e!r}") from e
