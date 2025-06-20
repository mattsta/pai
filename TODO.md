# Polyglot AI: Development TODO List

This is a list of concrete, near-term tasks to improve the framework.

### High Priority

- [x] Implement full, non-streaming support for all providers for single-shot command-line use.
- [x] Add robust error handling within the stream-parsing loops of each provider to prevent crashes on malformed data.
- [x] Create an `ANTHROPIC.md` guide in a `docs/providers` folder explaining how to build a new provider.
- [ ] Write basic unit tests for the `tools.py` decorator and execution logic.

### Medium Priority

- [ ] Refactor configuration management from `argparse` to `Pydantic` for cleaner validation and env var handling.
- [x] Create an `agent` command that uses the `code_editor.md` prompt by default to enable coding-assistant workflows.
- [ ] Add an `Anthropic` provider for Claude models.
- [ ] Integrate the `rich` library to render model output as Markdown.
- [ ] Implement session saving and loading (`/save` and `/load` commands).
- [x] Improve the `/stats` command to show tokens-per-second for the *last* request in addition to the session average.

### Low Priority / Research

- [x] ~~Investigate `httpx` for a future migration to async I/O.~~ (Completed)
- [ ] Design a system for managing multiple `system` prompts within a session.
- [ ] Add a `--log-file` option to write all debug output to a file for later analysis.

### Feature: Multi-Model Arena
- [x] **Phase 1: Configuration and Data Models**
    - [x] **Config (`polyglot.toml`):**
        - [x] Add a new `[arenas.NAME]` table format for defining arenas.
        - [x] Each arena configuration now defines two or more participants.
        - [x] Each participant has a `name`, `model`, `system_prompt_key`, and `endpoint`.
    - [x] **Data Models (`pai/models.py`):**
        - [x] Created `ArenaParticipant` dataclass to hold the configured state for one model in the arena.
        - [x] Created `Arena` dataclass to hold the configuration for a two-participant arena session.
        - [x] Modified the `Turn` dataclass to include `model_name` and `participant_name` to track which model generated a response.
- [x] **Phase 2: Core Logic and Orchestration**
    - [x] **Commands (`pai/commands.py`):**
        - [x] Added a new `/arena <arena_name> [max_turns]` command.
        - [x] Command logic loads the arena configuration from `polyglot.toml`, validating that it has two participants on the current endpoint.
        - [x] The command enables a new `arena_mode` in the UI and clears history for the new session.
    - [x] **Orchestration (`pai/pai.py`):**
        - [x] In `InteractiveUI`, added `arena_mode` flag and state for the active `Arena` config.
        - [x] Created a new `_run_arena_loop` async method to manage the conversation.
        - [x] The loop correctly manages separate `Conversation` objects for each participant, ensuring valid turn-by-turn histories are maintained for each model.
        - [x] The loop feeds one model's output as the user input to the other.
        - [x] The loop runs for the specified number of turns and can be interrupted by `Ctrl+C`.
- [x] **Phase 3: User Interface and Experience**
    - [x] **UI State (`pai/pai.py`):**
        - [x] The bottom toolbar (`_get_toolbar_text`) now displays the active arena name, participants, and their models.
        - [x] The input prompt (`_create_application`) now changes in arena mode to show which participant will receive the initial user prompt (e.g., "⚔️ Prompt for Proposer:").
    - [x] **UI Display (`pai/pai.py`):**
        - [x] The `StreamingDisplay` now accepts an `actor_name` to display which participant is "speaking".
        - [x] Instead of "🤖 Assistant:", the output is now clearly labeled (e.g., "🤖 Proposer:"). The call stack was updated to pass this metadata through the display pipeline.
- [x] **Phase 4: Logging and Persistence**
    - [x] **Turn Logging (`pai/pai.py`):**
        - [x] `Turn` objects created in the arena loop are correctly populated with `model_name` and `participant_name`.
        - [x] The main session log now contains a correctly interleaved sequence of turns from all participants.
    - [x] **HTML Templates (`pai/templates/conversation.html`):**
        - [x] Modified `get_rich_history_for_template` to add a `participant_index` to each message.
        - [x] The HTML template now uses this index to assign unique CSS classes and colors, visually distinguishing messages from each participant in the arena log.
- [ ] **Phase 5: Future Enhancements (Post-MVP)**
    - [x] Allow participants to be on different endpoints within the same arena.
    - [ ] Implement a mechanism for the user to interject and steer the conversation mid-loop.
    - [ ] Add a third "judge" model that observes the conversation and provides a summary or verdict at the end.
