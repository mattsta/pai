import inspect
import json
import enum
import pathlib
import sys
import importlib
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


def get_tool_manifest() -> str:
    """Generates a text manifest of all registered tools for legacy models."""
    if not TOOL_REGISTRY:
        return "No tools available."

    manifest = "You have access to the following tools:\n\n"
    for name, info in TOOL_REGISTRY.items():
        schema = info["schema"]["function"]
        manifest += f"- Name: {name}\n"
        manifest += f"  Description: {schema['description']}\n"
        if properties := schema["parameters"]["properties"]:
            manifest += "  Arguments:\n"
            for arg_name, details in properties.items():
                arg_type = details.get("type", "any")
                enum_values = details.get("enum")
                if enum_values:
                    arg_type = f"string (enum: {', '.join(enum_values)})"
                manifest += f"    - {arg_name} ({arg_type}): {details.get('description', '')}\n"
        else:
            manifest += "  Arguments: None\n"
        manifest += "\n"
    return manifest


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


def load_tools_from_directory(directory: str, printer: Callable = print):
    """Dynamically loads tools from Python files in a given directory."""
    path = pathlib.Path(directory)
    if not path.is_dir():
        # Don't print an error if the default dir doesn't exist.
        return

    # To enable relative imports, we treat the tool directory as a package.
    # Its parent directory must be on the Python path.
    parent_dir = str(path.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # The directory must contain an __init__.py to be a package.
    if not (path / "__init__.py").exists():
        printer(
            f"  ⚠️  Warning: Skipping tool directory '{path.name}'. It is not a Python package (missing __init__.py)."
        )
        return

    printer(f"🔎 Loading custom tools from: {path.resolve()}")
    found = False
    for file_path in path.glob("*.py"):
        # Skip __init__ files and any other private-like files.
        if file_path.stem.startswith("__init__") or file_path.stem.startswith("_"):
            continue
        try:
            # Construct the full module name for a proper import.
            module_name = f"{path.name}.{file_path.stem}"
            importlib.import_module(module_name)
            printer(f"  ✅ Loaded custom tool module: {file_path.name}")
            found = True
        except Exception as e:
            printer(f"  ❌ Failed to load {file_path.name}: {e}")
    if not found:
        printer("  (No custom tools found)")
