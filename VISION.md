# Polyglot AI: Vision and Guiding Principles

### The Problem: A Fragmented AI Landscape

The world of generative AI is exploding, but interacting with it remains a fragmented and often frustrating experience for developers and researchers. Each new model provider introduces a new SDK, a different API schema, unique authentication methods, and subtle behavioral quirks.

Switching between OpenAI for its powerful tool-calling, an open-source model via Featherless for its cost-effectiveness, and a specialized model for its unique capabilities requires constant code changes, context switching, and a deep understanding of each ecosystem's intricacies. Debugging cross-provider issues is a nightmare.

### Our Vision: A Unified Interface to the World of AI

We envision a future where interacting with any AI model is as simple as changing a single line in a configuration file or typing a command in a terminal.

**Polyglot AI aims to be the universal translator and swiss-army knife for generative AI interaction.**

It is a single, powerful, and transparent command-line interface that provides a consistent and debuggable environment for any model on any provider. It is built for the practitioner who needs to rapidly test, compare, and utilize different models without getting bogged down in boilerplate code and API specifics.

### Guiding Principles

1.  **Provider Agnostic, Not Abstracted Away:** We will support any provider, but we will not hide their unique features. The goal is to provide a common *interface*, not a lowest-common-denominator abstraction. If a provider supports unique features like tool-calling, we will expose them.

2.  **Interactivity and Introspection First:** The primary use case is a human-in-the-loop. The user must always have the ability to inspect the state, view the raw data flow (`debug` mode), and understand exactly what the system is doing. There will be no "magic." This is reinforced by features like the live status toolbar, which gives the user constant, immediate feedback on the session's performance and state.

3.  **Extensibility is a Core Feature:** The framework's value grows with the number of providers it supports. Adding a new provider should be a trivial, well-documented process. The architecture will always prioritize this modularity.

4.  **Local-First and User-Controlled:** This is not a cloud service. It is a tool that runs on your machine, using your API keys. All conversation data, including comprehensive session logs, is stored locally in the `sessions/` directory. You are in complete control of your data.

5.  **Pragmatic, Not Prescriptive:** Polyglot AI is a tool, not an opinionated framework like LangChain. It gives you the power to construct agentic loops and complex interactions, but it does not force you into a specific way of thinking. It provides the building blocks; you provide the creativity.
