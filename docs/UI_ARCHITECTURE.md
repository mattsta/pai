# Polyglot AI: UI Architecture and Lessons Learned

This document details the final, stable architecture of the Polyglot AI interactive UI and critically examines the failed attempts that led to its design. Understanding this evolution is key to maintaining and extending the application without re-introducing subtle, frustrating bugs.

## The Final Architecture: `prompt-toolkit.Application`

The core of our stable UI is a persistent `prompt_toolkit.Application` instance. This is a fundamental shift away from a simple `input()` or `PromptSession` loop.

The key components are:

1.  **`Application` Object**: This is the root of the UI. It runs its own asynchronous event loop, continuously rendering the screen based on the defined layout and application state. It owns the entire terminal screen while it's running.

2.  **Layout Management (`HSplit`, `ConditionalContainer`, `Window`)**: The UI is a tree of layout components.
    -   `HSplit`: Stacks components vertically. Our layout is an `HSplit` containing the live output area, the input/waiting area, and the bottom toolbar.
    -   `Window`: A basic container for displaying content.
    -   `ConditionalContainer`: A powerful component that displays its contents only when a given `Filter` (or `Condition`) is active. This is how we swap between showing the user input prompt and the "[Waiting for response...]" message.

3.  **State Management (`asyncio.Event` and `Buffer`)**: This is the most critical concept. The UI is reactive; it changes based on the state of a few key objects.
    -   `generation_in_progress` (`asyncio.Event`): This event acts as the master switch for the UI's state. When it's set, the `ConditionalContainer` hides the input prompt and shows the "waiting" message. When it's cleared, the UI reverts.
    -   `input_buffer` (`prompt_toolkit.Buffer`): Holds the text for the user's input. It is configured with an `accept_handler` callback. The key to correct history management is that this handler calls `buffer.reset(append_to_history=True)`. This single method correctly appends the input to history, clears the buffer, and resets the history navigation cache so the new entry is immediately available via the up/down arrows.
    -   `streaming_output_buffer` (`prompt_toolkit.Buffer`): A dedicated buffer for the live, streaming AI response. **This is the solution to all previous rendering bugs.** The background generation task *only* updates the `.text` property of this buffer. To fix a bug where the window would not scroll with long responses, the code now also manually moves the cursor to the end of this buffer (`output_buffer.cursor_position = len(...)`) after each streaming text update, which signals to `prompt-toolkit` to keep the new content in view.
    -   **`SearchToolbar`**: A built-in widget that is added to the layout. It is hidden by default but becomes visible when a search is initiated (e.g., via `Ctrl+R`), providing the UI for reverse incremental history search.

4.  **Key Bindings**: For most actions (like `Ctrl+D` to exit or history search), the application relies on `prompt-toolkit`'s default key bindings. However, it implements a custom binding for `Ctrl+C` to provide state-aware behavior:
    -   If a model generation is in progress, `Ctrl+C` cancels the background `asyncio` task.
    -   If the UI is idle, `Ctrl+C` clears the current input buffer. If the buffer is already empty, it creates a new prompt line. It does **not** exit the application.

5.  **Asynchronous Tasks**: When the user sends a message, an `asyncio.create_task()` dispatches the network-bound generation work to a background task. A reference to this task is stored. This allows the `Application`'s event loop to continue running unimpeded and allows the `Ctrl+C` key binding to cancel the task if needed.

## A History of Failure: How We Got Here

Our first several attempts failed because we tried to fight the `prompt-toolkit` framework instead of embracing its state-driven model.

1.  **Initial Mistake: `while True` + `await prompt_async()`**
    -   **Problem**: This common pattern is simple but flawed for a complex UI. The `await prompt_async()` call is blocking. While waiting for the user to press Enter, the UI updates. But once the prompt is submitted and the application awaits the network response, the `prompt_toolkit` event loop is no longer running, and the entire UI (including the toolbar) vanishes.

2.  **Failed Fix #1: Abusing `patch_stdout`**
    -   **Problem**: `patch_stdout` is a context manager designed to allow *synchronous*, non-`prompt-toolkit`-aware code to `print()` without corrupting the UI. We misused it by wrapping an `async` waiting loop (`while not event.is_set(): await asyncio.sleep(...)`). This fought with the `Application`'s renderer and was not a stable solution.

3.  **Failed Fix #2: Printing with Carriage Returns (`\r`)**
    -   **Problem**: We attempted to create a streaming effect by repeatedly printing the entire response line, ending with a `\r` to move the cursor to the start. This is a fragile terminal hack that is completely incompatible with a full-screen application framework like `prompt-toolkit`. The framework has its own complex renderer that redraws the entire UI. Our manual `\r` calls interfered with the renderer's cursor positioning, causing bizarre artifacts like the "multiple agent lines" bug.

## The "Aha!" Moment: Stop Fighting the Framework

The key lesson was to **stop trying to manually draw to the screen**. The correct pattern is:

1.  Build a declarative UI layout using `prompt-toolkit` components.
2.  Define the application's state using `Buffer`s and `Condition`s.
3.  Run background tasks (`asyncio`) to perform work.
4.  The *only* thing the background tasks should do is update the shared state (e.g., `my_buffer.text = new_text`).
5.  Let `prompt-toolkit`'s `Application` see the state change and handle all rendering and redrawing.

This decoupled architecture is robust, bug-free, and the only correct way to build a complex, responsive TUI with this library.
