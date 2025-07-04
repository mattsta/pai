import asyncio
import difflib
import json
import pathlib
import re

from pai.tools import tool

from .file_system import WORKSPACE, is_safe_path

# The regex to parse a single SEARCH/REPLACE block from a script.
# It captures the file path, search content, and replace content.
# The re.DOTALL flag allows '.' to match newlines, which is crucial for multiline blocks.
EDIT_BLOCK_REGEX = re.compile(
    r"````(?:[a-zA-Z]+)?\n"  # Optional language hint (non-capturing)
    r"(.+?)\n"  # Capture group 1: File path
    r"<<<<<<< SEARCH\n"
    r"(.*?)"  # Capture group 2: Search content
    r"=======\n"
    r"(.*?)"  # Capture group 3: Replace content
    r">>>>>>> REPLACE\n"
    r"````",  # Closing fence
    re.DOTALL,
)


@tool
async def apply_search_replace(edit_script: str) -> str:
    """Parses a script containing one or more SEARCH/REPLACE blocks and applies the edits.

    Each block must strictly follow the format (without the || which stops other editors from breaking here):
    ||````[language]
    ||path/to/file.ext
    ||<<<<<<< SEARCH
    ||content_to_find
    ||=======
    ||content_to_replace_with
    ||>>>>>>> REPLACE
    ||````

    The tool processes each block sequentially and returns a single JSON object.
    The `result` field contains a list of results for each block.
    For new files, the SEARCH block should be empty. For edits, it must exactly match.

    Args:
        edit_script (str): A string containing one or more valid SEARCH/REPLACE blocks.
    """
    operation_results = []
    blocks = EDIT_BLOCK_REGEX.findall(edit_script)

    if not blocks:
        result = {
            "status": "failure",
            "reason": "No valid SEARCH/REPLACE blocks were found in the provided script. Please ensure the format is correct.",
        }
        return json.dumps(result, indent=2)

    for file_path_str, search_block, replace_block in blocks:
        file_path = pathlib.Path(file_path_str.strip())
        operation_result = {"file_path": str(file_path), "status": "pending"}

        if not is_safe_path(str(file_path)):
            operation_result["status"] = "failure"
            operation_result["reason"] = "Path is outside the allowed workspace."
            operation_results.append(operation_result)
            continue

        try:
            full_path = WORKSPACE / file_path

            # Handle new file creation
            if not search_block:
                if await asyncio.to_thread(full_path.exists):
                    operation_result["status"] = "failure"
                    operation_result["reason"] = (
                        "File already exists. Cannot create a new file with an empty "
                        "SEARCH block if the file is not new."
                    )
                else:
                    await asyncio.to_thread(
                        full_path.parent.mkdir, parents=True, exist_ok=True
                    )
                    await asyncio.to_thread(
                        full_path.write_text, replace_block, encoding="utf-8"
                    )
                    operation_result["status"] = "success"
                    operation_result["reason"] = f"Created new file: {file_path}"
            # Handle file editing
            else:
                if not await asyncio.to_thread(full_path.is_file):
                    operation_result["status"] = "failure"
                    operation_result["reason"] = f"File not found at '{file_path}'."
                else:
                    original_content = await asyncio.to_thread(
                        full_path.read_text, encoding="utf-8"
                    )
                    # Normalize line endings for robust matching
                    normalized_content = original_content.replace("\r\n", "\n")
                    normalized_search = search_block.replace("\r\n", "\n")

                    if normalized_search not in normalized_content:
                        # Provide a diff for debugging if the match failed
                        diff = difflib.unified_diff(
                            search_block.splitlines(keepends=True),
                            original_content.splitlines(keepends=True),
                            fromfile="SEARCH BLOCK",
                            tofile=str(file_path),
                            n=2,  # two lines of context
                        )
                        closest_match = "\n".join(list(diff))
                        operation_result["status"] = "failure"
                        operation_result["reason"] = (
                            "SEARCH block not found in file. The file content may have changed, or line endings might differ."
                        )
                        operation_result["debug_hint"] = (
                            f"A diff of the SEARCH block against the file content suggests there are differences. Diff:\n{closest_match}"
                        )

                    else:
                        # Replace only the first occurrence in the normalized content
                        normalized_new_content = normalized_content.replace(
                            normalized_search, replace_block.replace("\r\n", "\n"), 1
                        )
                        # If the original file used CRLF, convert back
                        if "\r\n" in original_content:
                            final_content = normalized_new_content.replace("\n", "\r\n")
                        else:
                            final_content = normalized_new_content

                        await asyncio.to_thread(
                            full_path.write_text, final_content, encoding="utf-8"
                        )
                        operation_result["status"] = "success"
                        operation_result["reason"] = f"Applied edit to {file_path}"

        except Exception as e:
            operation_result["status"] = "failure"
            operation_result["reason"] = f"An unexpected error occurred: {e!r}"

        operation_results.append(operation_result)

    final_result = {"status": "success", "result": operation_results}
    return json.dumps(final_result, indent=2)
