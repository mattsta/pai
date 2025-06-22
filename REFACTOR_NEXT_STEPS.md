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

-   [ ] **Create Orchestrator Classes:**
    -   [ ] Create a new `pai/orchestration/` directory.
    -   [ ] Create a `BaseOrchestrator` abstract class with an `async def run(...)` method.
    -   [ ] Move the logic from `_run_arena_orchestrator` and `_run_arena_judge` into a new `ArenaOrchestrator` class in `pai/orchestration/arena.py`.
    -   [ ] Move the logic from `_run_legacy_agent_loop` into a `LegacyAgentOrchestrator` in `pai/orchestration/legacy_agent.py`.
    -   [ ] The main `_process_and_generate` logic can be moved to a `DefaultOrchestrator`.

-   [ ] **Update `InteractiveUI`:**
    -   [ ] The `_on_buffer_accepted` method will become a dispatcher. Based on the current `UIMode`, it will instantiate the appropriate orchestrator and run it.
    -   [ ] The `InteractiveUI` will pass necessary context (like the `PolyglotClient` and `Conversation` objects) to the orchestrator upon creation.

### Phase 3: Command and Client Decoupling

The `Command` classes and the `PolyglotClient` are still somewhat tightly coupled to the UI and each other.

-   [ ] **Refine Command Dependencies:**
    -   [ ] Review all `Command` classes in `pai/commands.py`.
    -   [ ] Replace direct state modifications (e.g., `self.ui.runtime_config.temperature = ...`) with calls to setter methods on the `InteractiveUI` (e.g., `self.ui.set_temperature(...)`). This provides a cleaner interface and a single place to handle side effects of state changes.

-   [ ] **Clarify Client Responsibilities:**
    -   [ ] The `PolyglotClient` currently handles endpoint switching and holds the active endpoint configuration (`EndpointConfig`). This is good.
    -   [ ] However, commands like `/model` and `/timeout` modify this state through `self.ui.client.config`. This should be formalized through methods on the `PolyglotClient` itself (e.g., `client.set_model(...)`).

### Phase 4: Tool System Refinement

The `@tool` decorator in `pai/tools.py` is very powerful but also very complex. Breaking it down will improve clarity and testability.

-   [ ] **Extract Schema Generation:**
    -   [ ] Create a private helper function `_generate_schema_for_function(func: Callable) -> dict` within `pai/tools.py`.
    -   [ ] Move all the `inspect` logic for parsing signatures and docstrings into this new function.
    -   [ ] The `@tool` decorator will then become much simpler: it will call `_generate_schema_for_function` and register the function and its schema in the `TOOL_REGISTRY`.

-   [ ] **Improve Tool Error Handling:**
    -   [ ] The `execute_tool` function currently returns error strings. This works but is not ideal for programmatic use.
    -   [ ] Refactor `execute_tool` to raise specific exceptions (`ToolNotFound`, `ToolArgumentError`) and have the calling code (the protocol adapters) catch these and format them into the appropriate "tool result" message for the AI.
