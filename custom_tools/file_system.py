import asyncio
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
async def read_file(path: str) -> str:
    """Reads the entire content of a text file.

    Args:
        path (str): The path to the file, relative to the current working directory.
    """
    if not is_safe_path(path):
        result = {
            "status": "failure",
            "reason": "Path is outside the allowed workspace.",
        }
        return json.dumps(result, indent=2)
    try:
        content = await asyncio.to_thread(
            (WORKSPACE / path).read_text, encoding="utf-8"
        )
        result = {"status": "success", "result": content}
    except FileNotFoundError:
        result = {"status": "failure", "reason": f"File not found at '{path}'"}
    except Exception as e:
        result = {"status": "failure", "reason": f"Error reading file '{path}': {e}"}
    return json.dumps(result, indent=2)


@tool
async def list_directory(path: str = ".") -> str:
    """Lists the files and directories at the given path.

    Returns a JSON object containing a list of entries.

    Args:
        path (str): The path to list, relative to the current working directory.
    """
    if not is_safe_path(path):
        result = {
            "status": "failure",
            "reason": "Path is outside the allowed workspace.",
        }
        return json.dumps(result, indent=2)
    try:
        full_path = (WORKSPACE / path).resolve()
        entries = await asyncio.to_thread(lambda: [e.name for e in full_path.iterdir()])
        result = {"status": "success", "result": entries}
    except FileNotFoundError:
        result = {"status": "failure", "reason": f"Directory not found at '{path}'"}
    except Exception as e:
        result = {
            "status": "failure",
            "reason": f"Error listing directory '{path}': {e}",
        }
    return json.dumps(result, indent=2)


@tool
async def write_file(path: str, content: str = "") -> str:
    """Writes content to a file, creating it if it doesn't exist or overwriting it if it does.

    Args:
        path (str): The path for the new file, relative to the current working directory.
        content (str): The content to write to the file.
    """
    if not is_safe_path(path):
        result = {
            "status": "failure",
            "reason": "Path is outside the allowed workspace.",
        }
        return json.dumps(result, indent=2)
    try:
        p = WORKSPACE / path
        await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(p.write_text, content, encoding="utf-8")
        result = {"status": "success", "result": f"File '{path}' written successfully."}
    except Exception as e:
        result = {"status": "failure", "reason": f"Error writing file '{path}': {e}"}
    return json.dumps(result, indent=2)


@tool
async def append_to_file(path: str, content: str) -> str:
    """Appends content to the end of an existing file.

    Args:
        path (str): The path to the file, relative to the current working directory.
        content (str): The content to append.
    """
    if not is_safe_path(path):
        result = {
            "status": "failure",
            "reason": "Path is outside the allowed workspace.",
        }
        return json.dumps(result, indent=2)

    def _do_append():
        with open(WORKSPACE / path, "a", encoding="utf-8") as f:
            f.write(content)

    try:
        await asyncio.to_thread(_do_append)
        result = {
            "status": "success",
            "result": f"Content appended to '{path}' successfully.",
        }
    except FileNotFoundError:
        result = {"status": "failure", "reason": f"File not found at '{path}'"}
    except Exception as e:
        result = {
            "status": "failure",
            "reason": f"Error appending to file '{path}': {e}",
        }
    return json.dumps(result, indent=2)


@tool
async def delete_file(path: str) -> str:
    """Deletes a single file from the filesystem.

    Args:
        path (str): The path to the file to delete.
    """
    if not is_safe_path(path):
        result = {
            "status": "failure",
            "reason": "Path is outside the allowed workspace.",
        }
        return json.dumps(result, indent=2)
    try:
        file_path = (WORKSPACE / path).resolve()
        is_file = await asyncio.to_thread(file_path.is_file)
        if not is_file:
            result = {
                "status": "failure",
                "reason": f"Not a file or not found at '{path}'",
            }
        else:
            await asyncio.to_thread(file_path.unlink)
            result = {
                "status": "success",
                "result": f"File '{path}' deleted successfully.",
            }
    except Exception as e:
        result = {"status": "failure", "reason": f"Error deleting file '{path}': {e}"}
    return json.dumps(result, indent=2)


@tool
async def delete_directory(path: str) -> str:
    """Deletes a directory and all its contents recursively. DANGEROUS.

    Args:
        path (str): The path to the directory to delete.
    """
    import shutil

    if not is_safe_path(path):
        result = {
            "status": "failure",
            "reason": "Path is outside the allowed workspace.",
        }
        return json.dumps(result, indent=2)

    dir_path = (WORKSPACE / path).resolve()
    if dir_path == WORKSPACE:
        result = {
            "status": "failure",
            "reason": "Cannot delete the root workspace directory.",
        }
        return json.dumps(result, indent=2)

    try:
        is_dir = await asyncio.to_thread(dir_path.is_dir)
        if not is_dir:
            result = {
                "status": "failure",
                "reason": f"Not a directory or not found at '{path}'",
            }
        else:
            await asyncio.to_thread(shutil.rmtree, dir_path)
            result = {
                "status": "success",
                "result": f"Directory '{path}' deleted successfully.",
            }
    except Exception as e:
        result = {
            "status": "failure",
            "reason": f"Error deleting directory '{path}': {e}",
        }
    return json.dumps(result, indent=2)
