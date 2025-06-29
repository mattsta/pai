import enum
import json

import httpx

from pai.tools import tool


class HttpMethod(enum.Enum):
    """The HTTP method to use for the request."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"


@tool
async def make_http_request(
    url: str,
    method: HttpMethod,
    headers: str = "{}",
    params: str = "{}",
    body: str = "",
) -> str:
    """Makes an HTTP request and returns a structured JSON response.

    This tool is for interacting with web APIs. It returns a JSON object
    containing the status code, response headers, and response body.

    Args:
        url (str): The URL to send the request to.
        method (HttpMethod): The HTTP method to use (GET, POST, etc.).
        headers (str): A JSON string representing request headers.
        params (str): A JSON string representing URL query parameters.
        body (str): The request body, for methods like POST or PUT.
    """
    try:
        parsed_headers = json.loads(headers)
        parsed_params = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in headers or params. {e}"

    response_data = {}
    try:
        # Use an async client to avoid blocking the event loop.
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.request(
                method.value,
                url,
                headers=parsed_headers,
                params=parsed_params,
                content=body,
            )

            response_data = {
                "status_code": response.status_code,
                "reason_phrase": response.reason_phrase,
                "headers": dict(response.headers),
                "body": response.text,
            }
    except httpx.HTTPStatusError as e:
        response_data = {
            "error": "HTTPStatusError",
            "status_code": e.response.status_code,
            "body": e.response.text,
        }
    except httpx.RequestError as e:
        response_data = {"error": "RequestError", "message": str(e)}
    except Exception as e:
        response_data = {"error": "UnexpectedError", "message": str(e)}

    # Return as a pretty-printed JSON string
    return json.dumps(response_data, indent=2)
