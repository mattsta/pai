import json
from typing import Any

import httpx

# Pricing per million tokens
ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}

from ..models import ChatRequest, CompletionRequest
from ..utils import estimate_tokens
from .base_adapter import BaseProtocolAdapter, ProtocolContext


class AnthropicAdapter(BaseProtocolAdapter):
    """Handles the Anthropic Messages API format."""

    def _calculate_cost(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> tuple[float, float]:
        """Calculates the cost of a request based on the model and token counts."""
        pricing = context.pricing_service.get_model_pricing(context.config.name, model_name)
        input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_token
        output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_token
        return input_cost, output_cost

    async def generate(
        self,
        context: ProtocolContext,
        request: ChatRequest | CompletionRequest,
        verbose: bool,
        actor_name: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(request, ChatRequest):
            raise TypeError("AnthropicAdapter only supports ChatRequest.")

        # 1. Translate the request to Anthropic's format
        system_prompt = ""
        messages = []
        for msg in request.messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # A more robust implementation would merge consecutive messages
                # of the same role, but for now we assume valid alternating history.
                messages.append(msg)

        url = f"{context.config.base_url}/messages"
        headers = {
            "x-api-key": context.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": context.config.model_name,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        # Anthropic doesn't support tools via this adapter yet.

        tokens_sent = estimate_tokens(system_prompt) + sum(
            estimate_tokens(m.get("content", "")) for m in messages
        )

        try:
            http_session = context.http_session
            http_session.headers.update(headers)

            context.display.start_response(
                tokens_sent=tokens_sent, actor_name=actor_name
            )

            # 2. Handle non-streaming case
            if not request.stream:
                response = await http_session.post(
                    url, json=payload, timeout=context.config.timeout
                )
                response.raise_for_status()
                response_data = response.json()
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
                    (
                        request_stats.input_cost,
                        request_stats.output_cost,
                    ) = self._calculate_cost(
                        context.config.model_name, input_tokens, output_tokens
                    )
                    request_stats.finish_reason = response_data.get("stop_reason")
                    context.stats.add_completed_request(request_stats)

                return {
                    "text": context.display.current_response,
                    "request": payload,
                    "response": response_data,
                }

            # 3. Handle streaming case
            finish_reason = None
            final_usage = {}
            async with http_session.stream(
                "POST", url, json=payload, timeout=context.config.timeout
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        line_str = line.strip()
                        context.display.show_raw_line(line_str)
                        if line_str.startswith("data:"):
                            chunk_str = line_str[6:]
                            try:
                                chunk = json.loads(chunk_str)
                                event_type = chunk.get("type")

                                if event_type == "message_start":
                                    input_tokens = (
                                        chunk.get("message", {})
                                        .get("usage", {})
                                        .get("input_tokens", tokens_sent)
                                    )
                                    if context.display.current_request_stats:
                                        context.display.current_request_stats.tokens_sent = input_tokens
                                elif event_type == "content_block_delta":
                                    chunk_text = chunk.get("delta", {}).get("text", "")
                                    await context.display.show_parsed_chunk(
                                        chunk, chunk_text
                                    )
                                elif event_type == "message_delta":
                                    finish_reason = chunk.get("delta", {}).get(
                                        "stop_reason"
                                    )
                                    final_usage = chunk.get("usage", {})

                            except (json.JSONDecodeError, IndexError) as e:
                                if context.display.debug_mode:
                                    context.display._print(
                                        f"⚠️  Stream parse error: {e} on line: {chunk_str!r}"
                                    )

            request_stats = await context.display.finish_response(success=True)
            if request_stats:
                output_tokens = final_usage.get(
                    "output_tokens", request_stats.tokens_received
                )
                request_stats.tokens_received = output_tokens
                (
                    request_stats.input_cost,
                    request_stats.output_cost,
                ) = self._calculate_cost(
                    context.config.model_name,
                    request_stats.tokens_sent,
                    output_tokens,
                )
                request_stats.finish_reason = finish_reason
                context.stats.add_completed_request(request_stats)

            return {
                "text": context.display.current_response,
                "request": payload,
                "response": {"usage": final_usage},
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
