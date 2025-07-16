import json
import logging
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

        if context.display.debug_mode:
            debug_payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
            log_line = f"🔵 DEBUG: REQUEST PAYLOAD\n{debug_payload_str}"
            context.display._print(log_line)
            logging.info(log_line)
        tokens_sent = estimate_tokens(request.prompt)

        try:
            context.display.start_response(
                tokens_sent=tokens_sent,
                actor_name=actor_name,
                model_name=request.model or context.config.model_name,
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
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    raise ConnectionError(
                        f"Failed to decode JSON response: {e}. Content: {response.text!r}"
                    ) from e

                if response_data.get("object") == "error" or "error" in response_data:
                    error_details = response_data.get("error", {})
                    error_message = error_details.get("message", "Unknown API error")
                    pretty_details = json.dumps(error_details, indent=2)
                    raise ValueError(f"API error: {error_message}\n{pretty_details}")

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

                                # Check for a streaming error object before processing as a regular chunk.
                                if chunk_data.get("object") == "error" or "error" in chunk_data:
                                    error_details = chunk_data.get("error", {})
                                    error_message = error_details.get("message", "Unknown streaming error")
                                    pretty_details = json.dumps(error_details, indent=2)
                                    raise ValueError(
                                        f"Streaming error from provider: {error_message}\n{pretty_details}"
                                    )

                                chunk_text = chunk_data.get("choices", [{}])[0].get(
                                    "text", ""
                                )
                                await context.display.show_parsed_chunk(
                                    chunk_data, chunk_text
                                )  # This also accumulates the response
                            except (json.JSONDecodeError, IndexError) as e:
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
            raise ConnectionError(f"Request failed: {e}")
