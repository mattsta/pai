import json
from typing import Any

import httpx

from ..models import CompletionRequest, RequestCost
from ..utils import estimate_tokens

# MODIFIED: No longer imports from the main script. Imports the context object instead.
from .base_adapter import BaseProtocolAdapter, ProtocolContext

# Import APIError from the main module for type consistency if needed, but it's better to raise generic exceptions
# from ..polyglot import APIError # We will use a generic exception instead to keep it decoupled.


class LegacyCompletionAdapter(BaseProtocolAdapter):
    """Handles the legacy /completions endpoint format."""

    async def generate(
        self,
        context: ProtocolContext,
        request: CompletionRequest,
        verbose: bool,
        actor_name: str | None = None,
    ) -> dict[str, Any]:
        # MODIFIED: All client properties are now accessed through the context object.
        # e.g., client.config -> context.config
        url = f"{context.config.base_url}/completions"
        payload = request.to_dict(context.config.model_name)
        tokens_sent = estimate_tokens(request.prompt)

        try:
            context.display.start_response(
                tokens_sent=tokens_sent, actor_name=actor_name
            )
            # Create and attach the cost tracker to the request stats
            model_pricing = context.pricing_service.get_model_pricing(
                context.config.name, request.model or context.config.model_name
            )
            if stats := context.display.current_request_stats:
                stats.cost = RequestCost(
                    _pricing_service=context.pricing_service,
                    _model_pricing=model_pricing,
                )

            if not request.stream:
                response = await context.http_session.post(
                    url, json=payload, timeout=context.config.timeout
                )
                response.raise_for_status()
                response_data = response.json()
                text = response_data.get("choices", [{}])[0].get("text", "")
                await context.display.show_parsed_chunk(response_data, text)

                request_stats = await context.display.finish_response(success=True)
                if request_stats:
                    request_stats.tokens_sent = tokens_sent
                    context.stats.add_completed_request(request_stats)

                return {
                    "text": context.display.current_response,
                    "request": payload,
                    "response": response_data,
                }

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
                                chunk_text = chunk_data.get("choices", [{}])[0].get(
                                    "text", ""
                                )
                                await context.display.show_parsed_chunk(
                                    chunk_data, chunk_text
                                )  # This also accumulates the response
                            except (json.JSONDecodeError, IndexError) as e:
                                if context.display.debug_mode:
                                    context.display._print(
                                        f"⚠️  Stream parse error on line: {data!r} | Error: {e}"
                                    )
                                continue

            request_stats = await context.display.finish_response(success=True)
            if request_stats:
                request_stats.tokens_sent = tokens_sent
                if request_stats.cost:
                    request_stats.cost.update(
                        input_tokens=request_stats.tokens_sent,
                        output_tokens=request_stats.tokens_received,
                    )
                context.stats.add_completed_request(request_stats)

            return {
                "text": context.display.current_response,
                "request": payload,
                # Fabricate a response object for logging consistency
                "response": {"usage": {"prompt_tokens": tokens_sent}},
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
                error_body = ""
                try:
                    await e.response.aread()
                    error_body = e.response.text
                except httpx.ResponseNotRead:
                    error_body = "[Could not read streaming error response body]"
                except Exception:
                    error_body = "[Error reading response body]"
                raise ConnectionError(
                    f"Request failed with status {e.response.status_code}: {error_body}"
                ) from e
        except Exception as e:
            request_stats = await context.display.finish_response(success=False)
            if request_stats:
                request_stats.tokens_sent = tokens_sent
                context.stats.add_completed_request(request_stats)
            raise ConnectionError(f"Request failed: {e!r}")
