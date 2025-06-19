import inspect
import json
from typing import Callable, List, Dict, Any

TOOL_REGISTRY: Dict[str, Callable] = {}


def tool(func: Callable) -> Callable:
    """Decorator to register a function as a tool the AI can use."""
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""
    param_docs = {}
    if "Args:" in docstring:
        args_section = docstring.split("Args:")[1].split("Returns:")[0]
        for line in args_section.strip().split("\n"):
            if ":" in line:
                name, desc = line.split(":", 1)
                param_docs[name.strip()] = desc.strip()

    type_mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        if param.annotation in type_mapping:
            properties[name] = {
                "type": type_mapping[param.annotation],
                "description": param_docs.get(name, ""),
            }
            if param.default is inspect.Parameter.empty:
                required.append(name)

    tool_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": docstring.split("\n")[0],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    TOOL_REGISTRY[func.__name__] = {"function": func, "schema": tool_schema}
    return func


def get_tool_schemas() -> List[Dict[str, Any]]:
    return [t["schema"] for t in TOOL_REGISTRY.values()] if TOOL_REGISTRY else []


def execute_tool(name: str, args: Dict) -> Any:
    if name not in TOOL_REGISTRY:
        return f"Error: Tool '{name}' not found."
    func = TOOL_REGISTRY[name]["function"]
    try:
        return func(**args)
    except Exception as e:
        return f"Error executing tool {name}: {e}"


@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Gets the current weather for a specified location.

    Args:
        location (str): The city and state, e.g., "San Francisco, CA".
        unit (str): The unit of temperature, either "celsius" or "fahrenheit".
    """
    if "tokyo" in location.lower():
        return json.dumps(
            {"location": "Tokyo", "temperature": "15", "condition": "Cloudy"}
        )
    if "paris" in location.lower():
        return json.dumps(
            {"location": "Paris", "temperature": "22", "condition": "Sunny"}
        )
    return json.dumps({"location": location, "temperature": "30", "condition": "Hot"})
