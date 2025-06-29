import json

from pai.tools import tool


@tool
def generate_python_class(
    class_name: str, fields: str, include_init: bool = True
) -> str:
    """Generates Python code for a dataclass with specified fields.

    Args:
        class_name (str): The name of the dataclass to generate (e.g., 'User').
        fields (str): A comma-separated list of fields with type hints, e.g., "name: str, age: int, is_active: bool".
        include_init (bool): Whether to include an __init__ method. Dataclasses handle this automatically. Set to False for a simple class.
    """
    result = {}
    try:
        if not class_name.isidentifier():
            raise ValueError(f"'{class_name}' is not a valid Python identifier.")

        field_lines = []
        for field in fields.split(","):
            field = field.strip()
            if not field:
                continue
            if ":" not in field:
                raise ValueError(
                    f"Field '{field}' is missing a type hint (e.g., 'name: str')."
                )
            name, type_hint = field.split(":", 1)
            name = name.strip()
            type_hint = type_hint.strip()
            if not name.isidentifier():
                raise ValueError(
                    f"Field name '{name}' is not a valid Python identifier."
                )
            field_lines.append(f"    {name}: {type_hint}")

        if include_init:
            code = "from dataclasses import dataclass\n\n@dataclass\n"
        else:
            code = ""

        code += f"class {class_name}:\n"
        if not field_lines:
            code += "    pass\n"
        else:
            code += "\n".join(field_lines) + "\n"
        result = {"status": "success", "result": code}
    except ValueError as e:
        result = {"status": "failure", "reason": f"Invalid input: {e}"}
    except Exception as e:
        result = {"status": "failure", "reason": f"An unexpected error occurred: {e}"}

    return json.dumps(result, indent=2)
