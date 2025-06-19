# PAI Conversion: Merging `gptwink` into `pai`

This document outlines the strategy and steps to merge the best features of the legacy `gptwink` tool into the new, more robust `pai` framework. The goal is to create a single, superior command-line interface that combines `pai`'s extensible architecture with `gptwink`'s powerful user experience and data persistence features.

## 1. High-Level Analysis

| Feature                | `gptwink` (Legacy)                              | `pai` (Modern)                               | Plan                                                                                          |
| :--------------------- | :---------------------------------------------- | :------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| **Architecture**       | Single monolithic file, async with `aiohttp`.   | Modular, protocol-based, async with `httpx`.   | **Keep `pai`'s architecture.** Its modularity is a core strength. The sync `requests` library has been replaced with `httpx` to support a non-blocking UI. |
| **Configuration**      | `.env` file for API keys, hardcoded logic.      | `polyglot.toml` for multiple endpoints.      | **Keep `pai`'s configuration.** It is superior for managing multiple providers.               |
| **Conversation History** | Stateful `Answer` object.                         | Simple list of dicts in `interactive_mode`.    | **Adopt `gptwink`'s OOP approach.** Create a stateful `Conversation` class in `pai` to manage history turns. |
| **Session Persistence**  | Saves every Q/A pair to a unique folder.        | None.                                        | **Implement `gptwink`'s persistence.** Every interactive session will be saved for review.    |
| **UI & Interactivity**   | Rich `prompt-toolkit` bottom toolbar with live stats. | Basic `prompt-toolkit` input.                | **Implement `gptwink`'s UI.** Add a live status toolbar to `pai`.                             |
| **Prompt Management**    | Hardcoded shortcuts (`/normcode`, `/tutor`).        | Basic `/system` command.                     | **Implement a file-based prompt system.** This is an improvement on `gptwink`'s concept.      |

## 2. Detailed Implementation Plan

The following changes will be made to the `pai` codebase.

### Step 1: Improve Conversation State Management

The current `messages` list in `pai`'s `interactive_mode` is brittle. We will replace it with a more robust, object-oriented approach inspired by `gptwink`'s `Answer` class.

-   [x] **Create `Conversation` Class:** A new class to manage the full conversation history. It will hold a list of `Turn` objects.
-   [x] **Create `Turn` Class:** A dataclass representing a single request-response cycle. It will contain:
    -   `turn_id`: A unique ID (using `ulid`).
    -   `timestamp`: The time of the turn.
    -   `request_data`: The payload sent to the API.
    -   `response_data`: The full response from the API.
    -   `assistant_message`: The final extracted text content from the assistant.
-   [x] **Integration:** The `interactive_mode` loop will now append new `Turn` objects to the `Conversation` object instead of managing a raw list of dictionaries.

### Step 2: Implement Session Persistence

`pai` sessions will be automatically saved for review, a key feature from `gptwink`.

-   [x] **Create `sessions/` directory:** All interactive sessions will be saved here.
-   [x] **Unique Session Folders:** On starting interactive mode, a new directory will be created, e.g., `sessions/2025-06-24-10-30-00-interactive/`.
-   [x] **Save Turns as JSON:** After each `Turn` is completed, it will be serialized to a JSON file and saved within the session folder, e.g., `01J3QZ...-turn.json`. This provides a complete, reviewable log of the entire conversation.
-   [x] Also conversations will be serialized to the gptwink html format as well.
-   [x] **Add Dependency:** The `ulid-py` library will be added to `pyproject.toml` to generate unique, sortable IDs for turns.

### Step 3: Enhance the UI with a Status Toolbar

We will implement `gptwink`'s most useful UI feature: a live status bar.

-   [x] **Modify `interactive_mode`:** The `PromptSession` will be configured with a `bottom_toolbar`.
-   [x] **Overhaul Toolbar and Streaming Display:** The UI has been completely re-architected around a persistent `prompt_toolkit.Application` to solve all stability and rendering issues.
    -   **Core Architectural Change:** The simple `PromptSession` in a `while` loop was replaced with a full `Application` instance. This application runs a continuous event loop, ensuring the UI (including the toolbar) is always present and responsive.
    -   **Solved UI Freezing:** Network requests are now dispatched to background `asyncio` tasks. They do not block the UI's event loop. Application state is managed via an `asyncio.Event` (`generation_in_progress`) that conditionally swaps UI components (e.g., hiding the input prompt and showing a "Waiting..." message).
    -   **Stable Streaming Output:** All previous rendering bugs (overwriting text, multiple newlines) were solved by introducing a dedicated `Buffer` for live output. The background generation task no longer `prints` to the screen; it simply updates the `.text` property of this buffer. The `Application` is responsible for rendering the buffer's contents, eliminating all rendering artifacts.
    -   **Multi-Line, Readable Layout:** The `get_toolbar_text()` function has been updated to produce a multi-line display, giving stats more room to breathe and making them easier to read at a glance.
    -   **Live Session Metrics:** The toolbar now provides a true real-time view of the session. The `Total Tokens` and `Avg Tok/s` metrics update live as new tokens are streamed, giving immediate feedback on performance and cost. The `Live Tok/s` metric for the *current* response stream remains for granular, per-request insight.
    -   [x] **Persistent & Searchable Command History:** The interactive prompt now supports a rich command history. This includes up/down arrow navigation, prefix-based search (e.g., type `/s` and press up), and reverse incremental search (`Ctrl+R`), powered by `prompt-toolkit`. History is stored in `~/.pai/history.txt`.
-   [x] This provides the user with immediate, persistent context about their session state.

### Step 4: Add a File-Based Prompt System

`gptwink`'s hardcoded prompt shortcuts were useful but inflexible. We will create a dynamic, file-based system.

-   **Create `prompts/` directory:** This directory will be added to the project.
-   **User-Created Prompts:** Users can add their own system prompts by creating `.txt` or `.md` files in this directory (e.g., `prompts/code_refactor.md`).
-   **New Commands:**
    -   `/prompts`: Lists all available prompt files in the `prompts/` directory.
    -   `/prompt <name>`: Loads the content of the specified file. This can either start a new conversation with that content as the system prompt or insert it into the current one.
- Retain both chat and completion mode endpoints for maximum experimental compatibility.

### Step 5: Cleanup and Finalization

Once the features above are integrated and tested:

-   The `gptwink/` directory and all its contents will be deleted.
-   The `main.py` file in the project root will be deleted, as `pai.pai:main` is the correct entry point.
-   The `README.md` will be updated to reflect the new features (session saving, prompts, and UI enhancements).

By following this plan, we will successfully evolve `pai` into a best-in-class tool, carrying forward the valuable lessons and features from its predecessor.
