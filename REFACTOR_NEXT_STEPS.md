# Polyglot AI: Next Steps for Refactoring

This document outlines a series of proposed refactoring phases to further improve the codebase's modularity, testability, and maintainability. These changes build upon the solid foundation of the existing architecture and aim to prepare the project for future growth, such as a formal plugin system.

### Phase 1: State Management Refactor (Centralize UI State)

The `InteractiveUI` class currently manages its mode (Chat, Agent, Arena, etc.) using several independent boolean flags (`is_chat_mode`, `native_agent_mode`, `arena_state`). This can be made more robust and easier to extend.

-   [x] **Introduce a State Machine:**
    -   [x] In `pai/models.py`, create a `UIMode` enum (e.g., `IDLE`, `CHAT`, `NATIVE_AGENT`, `LEGACY_AGENT`, `ARENA`).
    -   [x] In `pai/pai.py`, refactor `InteractiveUI` to use a single `self.mode: UIMode` attribute instead of multiple flags.
    -   [x] Create a `UIState` context class or similar structure to hold mode-specific data (e.g., the `ArenaState` object when in `ARENA` mode). This will replace `self.arena_state`.

-   [x] **Centralize State Transitions:**
    -   [x] Create methods on `InteractiveUI` for handling state transitions (e.g., `enter_arena_mode(config)`, `enter_agent_mode()`).
    -   [x] `Command` classes will call these methods instead of setting boolean flags directly, decoupling them from the UI's internal implementation.

### Phase 2: Orchestrator Extraction (Decouple Business Logic from UI)

The `InteractiveUI` class is currently responsible for both UI management and the business logic of running different modes (e.g., `_run_arena_orchestrator`). Extracting this logic will make the UI component simpler and the business logic easier to test independently.

-   [x] **Create Orchestrator Classes:**
    -   [x] Create a new `pai/orchestration/` directory.
    -   [x] Create a `BaseOrchestrator` abstract class with an `async def run(...)` method.
    -   [x] Move the logic from `_run_arena_orchestrator` and `_run_arena_judge` into a new `ArenaOrchestrator` class in `pai/orchestration/arena.py`.
    -   [x] Move the logic from `_run_legacy_agent_loop` into a `LegacyAgentOrchestrator` in `pai/orchestration/legacy_agent.py`.
    -   [x] The main `_process_and_generate` logic can be moved to a `DefaultOrchestrator`.

-   [x] **Update `InteractiveUI`:**
    -   [x] The `_on_buffer_accepted` method will become a dispatcher. Based on the current `UIMode`, it will instantiate the appropriate orchestrator and run it.
    -   [x] The `InteractiveUI` will pass necessary context (like the `PolyglotClient` and `Conversation` objects) to the orchestrator upon creation.

### Phase 3: Command and Client Decoupling (Completed)

The `Command` classes and the `PolyglotClient` are now more cleanly decoupled from direct state manipulation.

-   [x] **Refined Command Dependencies:** All commands that modify `RuntimeConfig` (e.g., `/temp`, `/tokens`, `/stream`, `/debug`) now call dedicated setter/toggler methods on the `InteractiveUI` class. This centralizes state management logic within the UI controller.
-   [x] **Clarified Client Responsibilities:** Commands that modify the active endpoint's configuration (`/model`, `/timeout`) now call dedicated methods on the `PolyglotClient` (`client.set_model`, `client.set_timeout`). This properly encapsulates client state within the client object itself.

### Phase 4: Tool System Refinement

The `@tool` decorator in `pai/tools.py` is very powerful but also very complex. Breaking it down will improve clarity and testability.

-   [ ] **Extract Schema Generation:**
    -   [ ] Create a private helper function `_generate_schema_for_function(func: Callable) -> dict` within `pai/tools.py`.
    -   [ ] Move all the `inspect` logic for parsing signatures and docstrings into this new function.
    -   [ ] The `@tool` decorator will then become much simpler: it will call `_generate_schema_for_function` and register the function and its schema in the `TOOL_REGISTRY`.

-   [ ] **Improve Tool Error Handling:**
    -   [ ] The `execute_tool` function currently returns error strings. This works but is not ideal for programmatic use.
    -   [ ] Refactor `execute_tool` to raise specific exceptions (`ToolNotFound`, `ToolArgumentError`) and have the calling code (the protocol adapters) catch these and format them into the appropriate "tool result" message for the AI.
