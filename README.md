# a1

**A modern agent compiler for building and executing LLM-powered agents**

a1 is a Python library that lets you build AI agents by composing tools, then compile them to efficient executable code. It provides both ahead-of-time (AOT) compilation for production and just-in-time (JIT) execution for development.

## Features

- 🔧 **Tool Composition**: Build agents from reusable tools with Pydantic schemas
- ⚡ **AOT Compilation**: Compile agents to cached Python code for fast execution
- 🔄 **JIT Execution**: Develop and test with on-the-fly code generation
- 🎯 **Smart Code Generation**: LLM-powered code synthesis with verification
- 📊 **Cost Optimization**: Rank and select optimal code candidates
- 🔌 **MCP Integration**: Load tools from Model Context Protocol servers
- 📁 **RAG Toolsets**: Built-in file system and SQL query tools
- 📈 **Observability**: OpenTelemetry instrumentation out of the box
- 🔗 **LangChain Compatible**: Convert existing LangChain agents

## Installation

```bash
# Install with uv (recommended)
uv pip install a1

# Or with pip
pip install a1

# Development installation
git clone https://github.com/yourusername/a1
cd a1
uv sync --dev
```

## Quick Start

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
    tools=[add, LLM("gpt-4o-mini"), Done()],
    terminal_tools=["done"]
)

# Use the agent with AOT compilation
async def main():
    runtime = Runtime()
    
    # Compile ahead-of-time (cached)
    compiled = await runtime.aot(agent)
    result = await compiled(problem="What is 2 + 2?")
    print(result)
    
    # Or execute just-in-time
    result = await runtime.jit(agent, MathInput(problem="What is 5 + 3?"))
    print(result)

import asyncio
asyncio.run(main())
```

## Core Concepts

### Agent

An agent is a composition of tools with defined behavior. It has:
- Input/output schemas (Pydantic models)
- A list of available tools
- Terminal conditions (which tools end execution)

### Tool

A tool is a callable function with schema validation:

```python
@tool(name="search", description="Search the web")
async def search(query: str) -> dict:
    # Implementation
    return {"results": [...]}
```

### ToolSet

A collection of related tools:

```python
from a1 import FileSystemRAG

# Create RAG toolset for file operations
fs_tools = FileSystemRAG("./documents")
# Provides: ls, grep, cat tools
```

### Runtime

The execution environment that manages:
- Code generation strategies
- Verification and cost estimation
- Context/history management
- Caching

```python
from a1 import Runtime, SimpleGenerate, SimpleVerify, SimpleCost

runtime = Runtime(
    generate=SimpleGenerate(llm_tool=LLM("gpt-4")),
    verify=[SimpleVerify()],
    cost=SimpleCost(tool_costs={"expensive_api": 100.0}),
)
```

## Advanced Features

### MCP Integration

Load tools from Model Context Protocol servers:

```python
from a1 import ToolSet

# Load from MCP server
mcp_config = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/data"]
}

toolset = await ToolSet.load_from_mcp(mcp_config, name="filesystem")
```

### Context Management

Track conversation history across executions:

```python
runtime = Runtime()

# Access contexts
runtime.H["main"]  # Main conversation
runtime.H["sidebar"]  # Alternative context

# Use compaction strategies
from a1 import SlidingWindowCompact

runtime.compact = SlidingWindowCompact(window_size=10)
```

### Custom Strategies

Implement custom code generation, verification, or cost strategies:

```python
from a1 import Generate, Verify, Cost

class MyGenerate(Generate):
    async def generate(self, agent, input_data, context):
        # Custom generation logic
        return "generated code"

class MyVerify(Verify):
    def verify(self, code, agent):
        # Custom verification
        return True, None

runtime = Runtime(
    generate=MyGenerate(),
    verify=[MyVerify()],
)
```

### LangChain Integration

Convert LangChain agents:

```python
from langchain.agents import initialize_agent
from a1 import Agent

# Create LangChain agent
lc_agent = initialize_agent(...)

# Convert to a1
a1_agent = Agent.from_langchain(lc_agent)
```

## CLI

```bash
# Clear compilation cache
a1 cache clear

# List cached compilations
a1 cache list

# Show version
a1 --version
```

## Architecture

a1 separates agent definition from execution:

1. **Definition**: Compose tools into an agent specification
2. **Compilation**: Generate and verify Python code (AOT or JIT)
3. **Execution**: Run compiled code with runtime support
4. **Observation**: Track with OpenTelemetry

This separation enables:
- Faster execution (compiled code vs interpreter overhead)
- Better optimization (rank multiple code candidates)
- Easier testing (verify code before execution)
- Production deployment (cache compiled agents)

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by and builds upon concepts from:
- [smolagents](https://github.com/huggingface/smolagents) - Code-based agent framework
- [LangChain](https://github.com/langchain-ai/langchain) - Agent composition patterns
- [Model Context Protocol](https://modelcontextprotocol.io/) - Tool standardization
- [any-llm](https://github.com/mozilla-ai/any-llm) - Unified LLM interface
