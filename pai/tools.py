import inspect
import json
import enum
import pathlib
import importlib.util
from typing import Callable, List, Dict, Any

TOOL_REGISTRY: Dict[str, Callable] = {}


class TemperatureUnit(enum.Enum):
    """Enumeration for temperature units."""

    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


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
        param_type = param.annotation
        property_details = None
        if inspect.isclass(param_type) and issubclass(param_type, enum.Enum):
            property_details = {
                "type": "string",
                "description": param_docs.get(name, ""),
                "enum": [e.value for e in param_type],
            }
        elif param_type in type_mapping:
            property_details = {
                "type": type_mapping[param_type],
                "description": param_docs.get(name, ""),
            }

        if property_details:
            properties[name] = property_details
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

    tool_info = TOOL_REGISTRY[name]
    func = tool_info["function"]
    sig = inspect.signature(func)

    # Convert arguments to their correct types, including Enums
    converted_args = {}
    try:
        for param_name, param_obj in sig.parameters.items():
            if param_name in args:
                arg_value = args[param_name]
                param_type = param_obj.annotation
                if inspect.isclass(param_type) and issubclass(param_type, enum.Enum):
                    converted_args[param_name] = param_type(arg_value)
                else:
                    converted_args[param_name] = arg_value
            # Let Python handle missing args with default values
        return func(**converted_args)
    except ValueError as e:
        # Specifically for enum conversion errors
        return f"Error: Invalid argument value provided for tool '{name}'. {e}"
    except Exception as e:
        return f"Error executing tool {name} with args {args}: {e}"


@tool
def get_current_weather(
    location: str, unit: TemperatureUnit = TemperatureUnit.CELSIUS
) -> str:
    """Gets the current weather for a specified location.

    Args:
        location (str): The city and state, e.g., "San Francisco, CA".
        unit (TemperatureUnit): The unit of temperature.
    """
    if "tokyo" in location.lower():
        temp = "15" if unit == TemperatureUnit.CELSIUS else "59"
        return json.dumps(
            {"location": "Tokyo", "temperature": temp, "condition": "Cloudy"}
        )
    if "paris" in location.lower():
        temp = "22" if unit == TemperatureUnit.CELSIUS else "72"
        return json.dumps(
            {"location": "Paris", "temperature": temp, "condition": "Sunny"}
        )
    temp = "30" if unit == TemperatureUnit.CELSIUS else "86"
    return json.dumps({"location": location, "temperature": temp, "condition": "Hot"})


@tool
def read_file(path: str) -> str:
    """Reads the contents of a text file from the local filesystem.

    Args:
        path (str): The relative or absolute path to the file.
    """
    try:
        # Note: Be careful with tools that access the filesystem.
        # In a real application, you should add validation to restrict access
        # to certain directories for security.
        return pathlib.Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: File not found at '{path}'"
    except Exception as e:
        return f"Error reading file '{path}': {e}"


def load_tools_from_directory(directory: str, printer: Callable = print):
    """Dynamically loads tools from Python files in a given directory."""
    path = pathlib.Path(directory)
    if not path.is_dir():
        # Don't print an error if the default dir doesn't exist.
        return

    printer(f"🔎 Loading custom tools from: {path.resolve()}")
    found = False
    for file_path in path.glob("*.py"):
        if file_path.stem.startswith("_"):
            continue
        try:
            # Import the module so the @tool decorator can register the function
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            printer(f"  ✅ Loaded custom tool module: {file_path.name}")
            found = True
        except Exception as e:
            printer(f"  ❌ Failed to load {file_path.name}: {e}")
    if not found:
        printer("  (No custom tools found)")
