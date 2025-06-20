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
- [ ] **Phase 3: User Interface and Experience**
    - [ ] **UI State (`pai/pai.py`):**
        - [ ] The bottom toolbar (`_get_toolbar_text`) needs to display the active arena and participants when in arena mode.
        - [ ] The input prompt (`_create_application`) should change to reflect arena mode, an_d indicate which model will receive the user's initial prompt.
    - [ ] **UI Display (`pai/pai.py`):**
        - [ ] The `StreamingDisplay` and UI print functions must be updated to show which participant is "speaking".
        - [ ] Instead of just "🤖 Assistant:", the output should be clearly labeled, e.g., "🤖 Proposer (gpt-4o-mini):". This will require passing participant metadata through the display pipeline.
- [ ] **Phase 4: Logging and Persistence**
    - [ ] **Turn Logging (`pai/pai.py`):**
        - [ ] When creating `Turn` objects inside the arena loop, the new `model_name` and `participant_name` fields must be correctly populated.
        - [ ] This ensures the unified session log will contain an interleaved sequence of turns from both models, each clearly attributed.
    - [ ] **HTML Templates (`pai/templates/conversation.html`):**
        - [ ] Modify the template to use the `participant_name` and `model_name` from the turn/message data.
        - [ ] Add CSS classes and logic to visually distinguish messages from each participant (e.g., using different background colors or labels).
- [ ] **Phase 5: Future Enhancements (Post-MVP)**
    - [ ] Allow participants to be on different endpoints within the same arena.
    - [ ] Implement a mechanism for the user to interject and steer the conversation mid-loop.
    - [ ] Add a third "judge" model that observes the conversation and provides a summary or verdict at the end.
