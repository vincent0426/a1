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

```bash
uv pip install a1-compiler
```

## 🏎️ Why use an agent compiler?

* **Safety** a1 generates code for every agent input and isolates LLM contexts as much as possible, reducing the amount of potentially untrusted data an LLM is exposed to. 
* **Speed** a1 makes codegen practical for agents with aggressive parallelism and static checking.
* **Determinism** a1 optimizes for determinism via a swappable cost function.

Agent compilers emerged from frustration with agent frameworks where every agent runs a static while loop program. Slow, unsafe, and highly nondeterministic. An agent compiler can perform the same while loop (just set `Verify=IsLoop()`) but has the freedom to explore superoptimal execution plans, while subject to engineered constraints.

## 🚀 How to get started?

```python
from a1 import Agent, tool, LLM
from pydantic import BaseModel

# Define a simple tool
@tool(name="add", description="Add two numbers")
async def add(a: int, b: int) -> int:
    return a + b

# Define input/output schemas
class MathInput(BaseModel):
    problem: str

class MathOutput(BaseModel):
    answer: int

# Create an agent with tools and LLM
agent = Agent(
    name="math_agent",
    description="Solves simple math problems",
    input_schema=MathInput,
    output_schema=MathOutput,
    tools=[add, LLM(model="gpt-4o")],  # LLMs are tools!
)

async def main():
    # Compile ahead-of-time
    compiled = await agent.aot()
    result = await compiled.execute(problem="What is 2 + 2?")
    print(f"AOT result: {result}")

    # Or execute just-in-time
    result = await agent.jit(problem="What is 5 + 3?")
    print(f"JIT result: {result}")

import asyncio
asyncio.run(main())
```

See the `tests/` directory for extensive examples of everything a1 can do. Docs coming soon to [docs.a1project.org](https://docs.a1project.org)

## ✨ Features

* **Import** your Langchain agents
* **Observability** via OpenTelemetry
* **Tools** instantiated from MCP, OpenAPI, or FastAPI servers
* **RAG** instantiated given any SQL database or fsspec path (e.g. `s3://my-place/here`, `gs://...`, or local filesystem)
  * Unified `RAG` router that automatically switches between file and database operations
  * `FileSystemRAG` for file operations (ls, grep, cat) using fsspec
  * `SQLRAG` for database queries using SQLAlchemy with pandas
* **Skills** defined manually or by crawling online docs
* **Context engineering** via a simple API that lets compiled code manage multi-agent behavior
* **Zero lock-in** use any LLM, any secure code execution cloud
* Only gets better as researchers develop increasingly powerful methods to `Generate`, `Cost` estimate, and `Verify` agent code

## 🤝 Contributing

Awesome! See our [Contributing Guide](/CONTRIBUTING.md) for details.

## 📄 MIT License

As it should be!