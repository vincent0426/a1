<div align="center">
  <img src="docs/assets/blast_icon_only.png" width="200" height="200" alt="BLAST Logo">
</div>

<p align="center" style="font-size: 24px">The a1 compiler for safe, reliable, deterministic AI agents.</p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Docs-FFE067)](https://docs.a1project.org)
[![Discord](https://img.shields.io/badge/Discord-FFE067)](https://discord.gg/NqrkJwYYh4)
[![Twitter Follow](https://img.shields.io/twitter/follow/realcalebwin?style=social)](https://x.com/realcalebwin)

</div>

a1 is an agent compiler. It takes an `Agent` (set of tools and a description) and compiles either AOT (ahead-of-time) into a `Tool` or JIT (just-in-time) for immediate execution tuned to the agent input.

```
pip install a1
```

## 🏎️ Why use an agent compiler?

* **Safety.** a1 generates code for every input task, reducing the attack surface for AI agents significantly.
* **Speed.** a1 makes codegen practical for agents with aggressive parallelism and static checking.
* **Determinism.** a1 optimizes for maximal determinism via a customizable cost function.

Agent compilers emerged from frustration with agent frameworks where every agent runs a static while loop program. An agent compiler can do the same (just set `Verify=IsLoop()`) but has the freedom to optimize.

## 🚀 How to get started?

```python
from a1 import Agent, tool, LLM, Done, Runtime
from pydantic import BaseModel

# Define a simple tool
@tool(name="add", description="Add two numbers")
async def add(a: int, b: int) -> int:
    return a + b

# Define input/output schemas
class MathInput(BaseModel):
    problem: str

class MathOutput(BaseModel):
    answer: str

# Create an agent
agent = Agent(
    name="math_agent",
    description="Solves simple math problems",
    input_schema=MathInput,
    output_schema=MathOutput,
    tools=[add, LLM("gpt-4.1")],
)

# Use the agent with AOT compilation
async def main():
    # Compile ahead-of-time
    compiled = await agent.aot()
    result = await compiled(problem="What is 2 + 2?")
    print(result)

    # Or execute just-in-time
    result = await runtime.jit(agent, problem="What is 5 + 3?")
    print(result)

import asyncio
asyncio.run(main())
```

See the `tests/` directory for extensive examples of everything a1 can do. Docs coming soon to [docs.a1project.org](https://docs.a1project.org)

## ✨ Features

* Import your Langchain agents
* Observability via OpenTelemetry
* Tools - instantiate from MCP or OpenAPI
* RAG - load from any SQL database, any fsspec path (e.g. `s3://my-place/here` or `somewhere/local`).
* Skills - define manually or crawl online docs.
* Context engineering - compile multi-agent systems that manage multiple contexts.
* Zero lock-in - use any LLM, any secure code execution cloud.
* Only gets better as researchers develop increasingly powerful methods to `Generate`, `Cost` estimate, and `Verify` agent code.

## 🤝 Contributing

Awesome! See our [Contributing Guide](https://docs.blastproject.org/development/contributing) for details.

## 📄 MIT License

As it should be!