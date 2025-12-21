"""Chat component for message input and streaming responses.

Single responsibility: Handle chat interactions
- Message submission
- Response streaming via SSE
- Conversation history display
"""

import asyncio
import html
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse, StreamingResponse
from starlette.routing import Route

from ...models import ChatRequest, Turn
from .base import Component

if TYPE_CHECKING:
    from ..session import WebSession


class ChatComponent(Component):
    """Handles chat message input and response streaming."""

    @property
    def component_id(self) -> str:
        return "chat"

    def get_routes(self) -> list[Route]:
        return [
            Route("/chat/send", self.send_message, methods=["POST"]),
            Route("/chat/stream/{session_id}", self.stream_response, methods=["GET"]),
            Route("/chat/history", self.get_history, methods=["GET"]),
            Route("/chat/clear", self.clear_history, methods=["POST"]),
        ]

    async def send_message(self, request: Request) -> HTMLResponse:
        """Handle message submission, return updated chat and trigger streaming."""
        form = await request.form()
        message = str(form.get("message", "")).strip()
        session_id = str(form.get("session_id", ""))

        if not message:
            return HTMLResponse(
                '<div class="error">Please enter a message</div>',
                status_code=400,
            )

        session = await self.session_manager.get_or_create_session(session_id)

        # Store the pending message for streaming
        session._pending_message = message  # type: ignore[attr-defined]

        # Return the user message bubble and a placeholder for the response
        # The placeholder will trigger SSE streaming
        user_bubble = f'''
        <div class="message user-message">
            <div class="message-content">{html.escape(message)}</div>
        </div>
        <div class="message assistant-message" id="response-{session.session_id}">
            <div class="message-content" id="stream-target">
                <span class="typing-indicator">Thinking...</span>
            </div>
        </div>
        <script>
            // Trigger SSE stream for response
            startStream("{session.session_id}");
        </script>
        '''
        return HTMLResponse(user_bubble)

    async def stream_response(self, request: Request) -> StreamingResponse:
        """Stream the AI response via Server-Sent Events."""
        session_id = request.path_params["session_id"]
        session = await self.session_manager.get_session(session_id)

        if not session:
            return StreamingResponse(
                self._error_stream("Session not found"),
                media_type="text/event-stream",
            )

        return StreamingResponse(
            self._generate_response(session),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _generate_response(
        self, session: "WebSession"
    ) -> AsyncGenerator[str, None]:
        """Generate SSE events for the streaming response."""
        message = getattr(session, "_pending_message", None)
        if not message:
            yield self._sse_event("error", "No pending message")
            return

        # Clear pending message
        session._pending_message = None  # type: ignore[attr-defined]

        # Build the request
        messages = session.conversation.get_messages_for_next_turn(message)
        request = ChatRequest(
            messages=messages,
            model=session.client.config.model_name,
            max_tokens=session.runtime_config.max_tokens,
            temperature=session.runtime_config.temperature,
            stream=True,
        )

        full_response = ""

        try:
            async with session._generation_lock:
                # Get the protocol adapter
                adapter = session.client._get_adapter()
                request_data = adapter.build_request(
                    request, session.client.config.model_name
                )

                # Make streaming request
                async with session.client.http_session.stream(
                    "POST",
                    session.client.config.get_endpoint_url(),
                    json=request_data,
                    headers=session.client.config.get_headers(),
                    timeout=60.0,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield self._sse_event(
                            "error", f"API Error: {error_text.decode()}"
                        )
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        # Parse SSE data
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)
                                content = self._extract_content(chunk)
                                if content:
                                    full_response += content
                                    # Send content chunk
                                    yield self._sse_event("content", content)
                            except json.JSONDecodeError:
                                continue

            # Add turn to conversation
            turn = Turn(
                request_data=request_data,
                response_data={"choices": [{"message": {"content": full_response}}]},
                assistant_message=full_response,
            )
            session.conversation.add_turn(turn, session.client.stats.last_request_stats)

            # Signal completion
            yield self._sse_event("done", "")

        except asyncio.CancelledError:
            yield self._sse_event("cancelled", "")
        except Exception as e:
            yield self._sse_event("error", str(e))

    def _extract_content(self, chunk: dict) -> str:
        """Extract content from a streaming chunk (handles multiple formats)."""
        # OpenAI format
        if choices := chunk.get("choices"):
            if delta := choices[0].get("delta"):
                return delta.get("content", "")

        # Anthropic format
        if chunk.get("type") == "content_block_delta":
            if delta := chunk.get("delta"):
                return delta.get("text", "")

        return ""

    def _sse_event(self, event: str, data: str) -> str:
        """Format a Server-Sent Event."""
        # Escape newlines in data for SSE
        escaped_data = data.replace("\n", "\\n")
        return f"event: {event}\ndata: {escaped_data}\n\n"

    async def _error_stream(self, message: str) -> AsyncGenerator[str, None]:
        """Generate an error SSE stream."""
        yield self._sse_event("error", message)

    async def get_history(self, request: Request) -> HTMLResponse:
        """Return the conversation history as HTML."""
        session_id = request.query_params.get("session_id", "")
        session = await self.session_manager.get_session(session_id)

        if not session:
            return HTMLResponse('<div class="empty-state">No conversation yet</div>')

        html_parts = []
        for turn in session.conversation.turns:
            # User message (from request)
            if turn.request_data and turn.request_data.get("messages"):
                user_msg = turn.request_data["messages"][-1].get("content", "")
                if user_msg:
                    html_parts.append(f"""
                    <div class="message user-message">
                        <div class="message-content">{html.escape(user_msg)}</div>
                    </div>
                    """)

            # Assistant message
            if turn.assistant_message:
                html_parts.append(f"""
                <div class="message assistant-message">
                    <div class="message-content">{html.escape(turn.assistant_message)}</div>
                </div>
                """)

        if not html_parts:
            return HTMLResponse('<div class="empty-state">No messages yet</div>')

        return HTMLResponse("".join(html_parts))

    async def clear_history(self, request: Request) -> HTMLResponse:
        """Clear the conversation history."""
        form = await request.form()
        session_id = str(form.get("session_id", ""))
        session = await self.session_manager.get_session(session_id)

        if session:
            session.conversation = type(session.conversation)()

        return HTMLResponse('<div class="empty-state">Conversation cleared</div>')
