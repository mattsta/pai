import csv
import io
import json

import yaml

from pai.tools import tool


@tool
def json_to_yaml(json_string: str) -> str:
    """Converts a JSON string to a YAML string.

    Args:
        json_string (str): A well-formed JSON string.
    """
    try:
        data = json.loads(json_string)
        result = {"status": "success", "result": yaml.dump(data, sort_keys=False)}
    except json.JSONDecodeError as e:
        result = {"status": "failure", "reason": f"Invalid JSON provided. {e}"}
    except Exception as e:
        result = {"status": "failure", "reason": f"An unexpected error occurred: {e}"}
    return json.dumps(result, indent=2)


@tool
def yaml_to_json(yaml_string: str) -> str:
    """Converts a YAML string to a pretty-printed JSON string.

    Args:
        yaml_string (str): A well-formed YAML string.
    """
    try:
        # Use safe_load to prevent arbitrary code execution from malicious YAML
        data = yaml.safe_load(yaml_string)
        result = {"status": "success", "result": data}
    except yaml.YAMLError as e:
        result = {"status": "failure", "reason": f"Invalid YAML provided. {e}"}
    except Exception as e:
        result = {"status": "failure", "reason": f"An unexpected error occurred: {e}"}
    return json.dumps(result, indent=2)


@tool
def csv_to_json(csv_data: str, delimiter: str = ",") -> str:
    """Converts CSV data to a JSON array of objects.

    The first row of the CSV is assumed to be the header row.

    Args:
        csv_data (str): The raw CSV data as a single string.
        delimiter (str): The delimiter character for the CSV (default is a comma).
    """
    try:
        # Use io.StringIO to treat the string data as a file
        csv_file = io.StringIO(csv_data)
        # Use DictReader to automatically use the first row as keys
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        result = {"status": "success", "result": list(reader)}
    except csv.Error as e:
        result = {"status": "failure", "reason": f"Invalid CSV data provided. {e}"}
    except Exception as e:
        result = {"status": "failure", "reason": f"An unexpected error occurred: {e}"}
    return json.dumps(result, indent=2)
