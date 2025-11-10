# a1 Quick Reference

## Installation

```bash
uv pip install a1
# or
pip install a1
```

## Basic Usage

### Define a Tool

```python
from a1 import tool

@tool(name="add", description="Add two numbers")
async def add(a: int, b: int) -> int:
    return a + b
```

### Create an Agent

```python
from a1 import Agent, LLM, Done
from pydantic import BaseModel

class Input(BaseModel):
    query: str

class Output(BaseModel):
    answer: str

agent = Agent(
    name="my_agent",
    description="Does something useful",
    input_schema=Input,
    output_schema=Output,
    tools=[add, LLM("gpt-4o-mini"), Done()],
    terminal_tools=["done"]
)
```

### Execute Agent

```python
from a1 import Runtime

runtime = Runtime()

# AOT: Compile and cache
compiled = await runtime.aot(agent)
result = await compiled(query="What is 2+2?")

# JIT: Execute on-the-fly
result = await runtime.jit(agent, Input(query="What is 2+2?"))
```

### Use Global Runtime

```python
from a1 import aot, jit, execute

# Uses get_runtime() internally
compiled = await aot(agent)
result = await jit(agent, input_data)
result = await execute(tool, input_data)
```

## Built-in Tools

### LLM

```python
from a1 import LLM

# Create LLM tool
llm = LLM("gpt-4o-mini")
llm = LLM("claude-3-5-sonnet-20241022")
llm = LLM("mistral:mistral-small-latest")

# With schemas
llm = LLM("gpt-4", input_schema=MyInput, output_schema=MyOutput)
```

### Done

```python
from a1 import Done

# Simple done
done = Done()

# With schema
done = Done(output_schema=MyOutput)
```

## RAG Toolsets

### FileSystem

```python
from a1 import FileSystemRAG

# Local filesystem
fs_tools = FileSystemRAG("./documents")

# S3
s3_tools = FileSystemRAG("s3://bucket/prefix")

# Tools: ls, grep, cat
```

### SQL

```python
from a1 import SQLRAG

# From connection string
sql_tools = SQLRAG("postgresql://user:pass@host/db")

# From DataFrame
import pandas as pd
df = pd.DataFrame(...)
sql_tools = SQLRAG(df, schema="my_table")

# Tool: sql (SELECT only)
```

## MCP Integration

```python
from a1 import ToolSet

mcp_config = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
}

toolset = await ToolSet.load_from_mcp(mcp_config, name="fs")
```

## Runtime Configuration

### Custom Strategies

```python
from a1 import Runtime, SimpleGenerate, SimpleVerify, SimpleCost, SlidingWindowCompact

runtime = Runtime(
    generate=SimpleGenerate(llm_tool=LLM("gpt-4")),
    verify=[SimpleVerify()],
    cost=SimpleCost(tool_costs={"expensive": 100.0}),
    compact=SlidingWindowCompact(window_size=10)
)
```

### Context Manager

```python
with Runtime() as rt:
    result = await rt.jit(agent, input_data)
    # rt is now the global runtime
```

## Context Management

### Access Contexts

```python
runtime = Runtime()

# Main context
main = runtime.H["main"]

# Custom context
runtime.H["sidebar"] = Context()

# No-history context
from a1 import no_history
temp = no_history()
```

### Message Types

```python
from a1 import Message

msg = Message(role="user", content="Hello")
msg = Message(role="assistant", content="Hi", tool_calls=[...])
msg = Message(role="tool", content="Result", name="tool_name", tool_call_id="123")
msg = Message(role="system", content="You are helpful")
```

## Custom Strategies

### Generate

```python
from a1 import Generate

class MyGenerate(Generate):
    async def generate(self, agent, input_data, context):
        # Your logic here
        return "generated code"
```

### Verify

```python
from a1 import Verify

class MyVerify(Verify):
    def verify(self, code, agent):
        # Your checks here
        return True, None  # (is_valid, error_message)
```

### Cost

```python
from a1 import Cost

class MyCost(Cost):
    def compute_cost(self, code, agent):
        # Your calculation
        return 42.0
```

### Compact

```python
from a1 import Compact

class MyCompact(Compact):
    def compact(self, contexts):
        # Your compaction logic
        return contexts
```

## CLI Commands

```bash
# Clear cache
a1 cache clear

# List cache
a1 cache list

# Custom cache dir
a1 cache clear --dir .my_cache

# Version
a1 --version
```

## LangChain Conversion

```python
from langchain.agents import initialize_agent
from a1 import Agent

lc_agent = initialize_agent(...)
a1_agent = Agent.from_langchain(lc_agent)
```

## Observability

### OpenTelemetry

Automatic instrumentation:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# Set up OTEL
trace.set_tracer_provider(TracerProvider())

# a1 automatically instruments:
# - runtime.aot()
# - runtime.jit()
# - runtime.execute()
```

### Attributes Tracked

- `agent.name`
- `tool.name`
- `cache.hit`
- `cache.enabled`
- `generation.type`

## Common Patterns

### Simple Agent

```python
@tool(name="search")
async def search(query: str) -> list:
    return [...]

agent = Agent(
    name="searcher",
    description="Search agent",
    input_schema=create_model("Input", query=(str, ...)),
    output_schema=create_model("Output", results=(list, ...)),
    tools=[search, LLM("gpt-4o-mini"), Done()]
)
```

### Multi-tool Agent

```python
tools = [
    tool1,
    tool2,
    FileSystemRAG("./docs"),
    SQLRAG(db_conn),
    LLM("gpt-4"),
    Done()
]

agent = Agent(name="worker", ..., tools=tools)
```

### Loop Agent

```python
from a1 import IsLoop

runtime = Runtime(
    verify=[SimpleVerify(), IsLoop()]
)

# IsLoop detected → uses template instead of LLM generation
compiled = await runtime.aot(agent)
```

## Error Handling

```python
from a1 import Runtime

runtime = Runtime()

try:
    result = await runtime.jit(agent, input_data)
except RuntimeError as e:
    # Generation or verification failed
    print(f"Error: {e}")
```

## Best Practices

1. **Use type hints** - Enables automatic schema generation
2. **Add descriptions** - Helps LLM understand tools
3. **Cache in production** - Set `cache=True` for `aot()`
4. **Use compaction** - Prevent context overflow
5. **Test tools separately** - Before adding to agents
6. **Start simple** - Add complexity incrementally
7. **Monitor with OTEL** - Track performance and costs

## Environment Variables

```bash
# LLM API keys
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export MISTRAL_API_KEY="..."

# Cache directory
export A1_CACHE_DIR=".a1"
```

## Tips & Tricks

### Debugging

```python
import logging
logging.basicConfig(level=logging.INFO)

# Shows:
# - Code execution
# - LLM calls
# - Tool calls
# - Cache operations
```

### Testing

```python
from a1 import SimpleExecutor

executor = SimpleExecutor()
result = await executor.execute("x = 2 + 2\nresult = x")
assert result.output == 4
```

### Multiple Candidates

```python
# Generate multiple code candidates (future feature)
# For now, use multiple verify strategies to filter
runtime = Runtime(
    verify=[
        SimpleVerify(),
        MyCustomVerifier1(),
        MyCustomVerifier2()
    ]
)
```
