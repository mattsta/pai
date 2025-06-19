# protocols/legacy_completion_adapter.py
import time
import json
from typing import Dict, Any

# MODIFIED: No longer imports from the main script. Imports the context object instead.
from .base_adapter import BaseProtocolAdapter, ProtocolContext, CompletionRequest
# Import APIError from the main module for type consistency if needed, but it's better to raise generic exceptions
# from ..polyglot import APIError # We will use a generic exception instead to keep it decoupled.


class LegacyCompletionAdapter(BaseProtocolAdapter):
    """Handles the legacy /completions endpoint format."""

    async def generate(
        self, context: ProtocolContext, request: CompletionRequest, verbose: bool
    ) -> Dict[str, Any]:
        # MODIFIED: All client properties are now accessed through the context object.
        # e.g., client.config -> context.config
        url = f"{context.config.base_url}/completions"
        payload = request.to_dict(context.config.model_name)
        tokens_sent = len(request.prompt.split())
        start_time = time.time()

        if not request.stream:
            raise NotImplementedError(
                "Non-streaming /completions is not implemented in this adapter."
            )

        try:
            context.display.start_response()
            full_text = ""
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
                                context.display.show_parsed_chunk(
                                    chunk_data, chunk_text
                                )  # This also accumulates the response
                            except (json.JSONDecodeError, IndexError):
                                continue

            elapsed, tokens_received, ttft = context.display.finish_response()
            context.stats.add_request(
                tokens_sent, tokens_received, elapsed, ttft=ttft
            )
            return {"text": context.display.current_response}

        except Exception as e:
            elapsed = time.time() - start_time
            context.stats.add_request(tokens_sent, 0, elapsed, success=False)
            # Use a generic exception type, but importantly, use repr(e) for a full error message.
            raise ConnectionError(f"Request failed: {e!r}")
