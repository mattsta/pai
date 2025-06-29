# PAI Session Logging and Persistence

This document explains how Polyglot AI automatically logs interactive sessions, how the process works, and how you can customize it.

## Automatic Logging

One of the core features of `pai` is its automatic, comprehensive session logging. You do not need to take any action to enable this; it is active by default for every interactive session.

### How it Works

1.  **Session Start:** When you launch `pai` in interactive mode, it immediately creates a new, unique directory in `logs/`. The directory name is timestamped to make it easy to find later (e.g., `logs/2025-06-25_10-30-00-interactive/`).

2.  **During the Session:** After every single turn—that is, after you send a prompt and the AI returns a complete response—the framework automatically writes and updates log files within that session's directory.

3.  **What is Logged?** For each turn, two types of files are generated and kept up-to-date:
    *   **Turn-specific JSON (`<turn_id>-turn.json`):** A new JSON file is created for every single turn. This file is a complete, raw dump of the `Turn` object and contains:
        *   A unique, sortable `turn_id`.
        *   The exact timestamp of the turn.
        *   The full `request_data` payload that was sent to the API.
        *   The full `response_data` received from the API.
        *   The final, extracted `assistant_message`.
    *   **Conversation HTML Logs:** Two HTML files (`conversation.html` and `gptwink_conversation.html`) are re-generated from scratch after *every turn*. This means that if you open `conversation.html` in a web browser, you can simply refresh the page after each turn to see the latest messages appended to the log.

### Maintaining Logs

The `logs/` directory can grow over time. Maintenance is currently a manual process. You can safely delete any old log folders from the `logs/` directory to free up space.

## Customizing Log Output

The HTML logs are generated using [Jinja2](https://jinja.palletsprojects.com/) templates, making them easy to customize.

### Template Location

The templates are located in the `pai/templates/` directory:

*   `conversation.html`: A modern, readable format with message bubbles and styling.
*   `gptwink_format.html`: A simple, legacy-compatible format designed for easy parsing by other scripts.

### How to Modify a Template

You can directly edit these HTML files. They use standard Jinja2 syntax. For example, `conversation.html` iterates through the message history with a `{% for message in history %}` loop. You can change the CSS, add or remove information, or completely redesign the layout.

### How to Add a New Log Format

If you need a new output format (e.g., a plain text log or a different HTML structure), follow these steps:

1.  **Create a New Template:** Add a new template file to the `pai/templates/` directory (e.g., `my_custom_log.txt`). Design it using Jinja2 syntax to render the `history` object as you see fit.

2.  **Register the Template:** Open `pai/log_utils.py` and find the `save_conversation_formats` function. Inside this function, there is a `formats` dictionary. Add a new entry for your template:

    ```python
    # In save_conversation_formats()
    formats = {
        "conversation.html": "conversation.html",
        "gptwink_format.html": "gptwink_conversation.html",
        "my_custom_log.txt": "conversation.txt",  # Add your new format here
    }
    ```

    The key is the name of your template file in `pai/templates/`, and the value is the filename that will be generated in the session directory.

The framework will now automatically generate `conversation.txt` alongside the other logs in every session.
