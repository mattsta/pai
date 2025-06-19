import httpx
from pai.tools import tool


@tool
def fetch_url(url: str) -> str:
    """Fetches the text content of a given URL.

    This tool can be used to get information from web pages.
    It returns the first 5000 characters to avoid overly long responses.

    Args:
        url (str): The URL to fetch, including the protocol (e.g., 'https://...').
    """
    try:
        # Use a timeout to prevent hanging on slow requests
        with httpx.Client() as client:
            response = client.get(url, follow_redirects=True, timeout=10.0)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Return a slice of the content to keep it manageable
            content = response.text
            return content[:5000]

    except httpx.HTTPStatusError as e:
        return f"Error: Received status {e.response.status_code} from {url}"
    except httpx.RequestError as e:
        return f"Error: Failed to fetch URL {url}. Reason: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
