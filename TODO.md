# Polyglot AI: Development TODO List

This is a list of concrete, near-term tasks to improve the framework.

### High Priority

- [ ] Implement full, non-streaming support for all providers for single-shot command-line use.
- [ ] Add robust error handling within the stream-parsing loops of each provider to prevent crashes on malformed data.
- [ ] Create an `ANTHROPIC.md` guide in a `docs/providers` folder explaining how to build a new provider.
- [ ] Write basic unit tests for the `tools.py` decorator and execution logic.

### Medium Priority

- [ ] Refactor configuration management from `argparse` to `Pydantic` for cleaner validation and env var handling.
- [ ] Add an `Anthropic` provider for Claude models.
- [ ] Integrate the `rich` library to render model output as Markdown.
- [ ] Implement session saving and loading (`/save` and `/load` commands).
- [ ] Improve the `/stats` command to show tokens-per-second for the *last* request in addition to the session average.

### Low Priority / Research

- [ ] Investigate `httpx` for a future migration to async I/O.
- [ ] Design a system for managing multiple `system` prompts within a session.
- [ ] Add a `--log-file` option to write all debug output to a file for later analysis.
