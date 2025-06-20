import json
import pathlib

from pai.tools import tool

# --- Security Note ---
# These tools interact directly with the local filesystem. This is powerful but
# carries risks. The `is_safe_path` function is a basic security measure to
# prevent the AI from accessing files outside the project's working directory.
# You can disable this by changing WORKSPACE_JAIL to False.

WORKSPACE_JAIL = True
WORKSPACE = pathlib.Path.cwd().resolve()


def is_safe_path(path_str: str) -> bool:
    """Checks if a path is within the allowed workspace directory."""
    if not WORKSPACE_JAIL:
        return True

    # Construct the absolute path and resolve any '..' components.
    target_path = (WORKSPACE / path_str).resolve()

    # Check if the workspace path is a parent of the target path.
    return WORKSPACE in target_path.parents or target_path == WORKSPACE


@tool
def read_file(path: str) -> str:
    """Reads the entire content of a text file.

    Args:
        path (str): The path to the file, relative to the current working directory.
    """
    if not is_safe_path(path):
        return "Error: Path is outside the allowed workspace."
    try:
        return (WORKSPACE / path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: File not found at '{path}'"
    except Exception as e:
        return f"Error reading file '{path}': {e}"


@tool
def list_directory(path: str = ".") -> str:
    """Lists the files and directories at the given path.

    Returns a JSON string containing a list of entries.

    Args:
        path (str): The path to list, relative to the current working directory.
    """
    if not is_safe_path(path):
        return "Error: Path is outside the allowed workspace."
    try:
        full_path = (WORKSPACE / path).resolve()
        entries = [e.name for e in full_path.iterdir()]
        if not entries:
            return "[]"  # Return empty JSON array
        return json.dumps(entries)
    except FileNotFoundError:
        return f"Error: Directory not found at '{path}'"
    except Exception as e:
        return f"Error listing directory '{path}': {e}"


@tool
def write_file(path: str, content: str = "") -> str:
    """Writes content to a file, creating it if it doesn't exist or overwriting it if it does.

    Args:
        path (str): The path for the new file, relative to the current working directory.
        content (str): The content to write to the file.
    """
    if not is_safe_path(path):
        return "Error: Path is outside the allowed workspace."
    try:
        p = WORKSPACE / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"File '{path}' written successfully."
    except Exception as e:
        return f"Error writing file '{path}': {e}"


@tool
def append_to_file(path: str, content: str) -> str:
    """Appends content to the end of an existing file.

    Args:
        path (str): The path to the file, relative to the current working directory.
        content (str): The content to append.
    """
    if not is_safe_path(path):
        return "Error: Path is outside the allowed workspace."
    try:
        with open(WORKSPACE / path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Content appended to '{path}' successfully."
    except FileNotFoundError:
        return f"Error: File not found at '{path}'"
    except Exception as e:
        return f"Error appending to file '{path}': {e}"


@tool
def delete_file(path: str) -> str:
    """Deletes a single file from the filesystem.

    Args:
        path (str): The path to the file to delete.
    """
    if not is_safe_path(path):
        return "Error: Path is outside the allowed workspace."
    try:
        file_path = (WORKSPACE / path).resolve()
        if not file_path.is_file():
            return f"Error: Not a file or not found at '{path}'"

        file_path.unlink()
        return f"File '{path}' deleted successfully."
    except Exception as e:
        return f"Error deleting file '{path}': {e}"


@tool
def delete_directory(path: str) -> str:
    """Deletes a directory and all its contents recursively. DANGEROUS.

    Args:
        path (str): The path to the directory to delete.
    """
    import shutil

    if not is_safe_path(path):
        return "Error: Path is outside the allowed workspace."

    dir_path = (WORKSPACE / path).resolve()
    if dir_path == WORKSPACE:
        return "Error: Cannot delete the root workspace directory."

    try:
        if not dir_path.is_dir():
            return f"Error: Not a directory or not found at '{path}'"

        shutil.rmtree(dir_path)
        return f"Directory '{path}' deleted successfully."
    except Exception as e:
        return f"Error deleting directory '{path}': {e}"
