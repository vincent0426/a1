"""
Simple example demonstrating a1 usage.

Creates a math agent that can add numbers.
"""

import asyncio
from pydantic import BaseModel
from a1 import Agent, tool, LLM, Done, Runtime


# Define a simple tool using the @tool decorator
@tool(name="add", description="Add two numbers together")
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool(name="multiply", description="Multiply two numbers together")
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Define input/output schemas
class MathInput(BaseModel):
    """Input for math agent."""
    problem: str


class MathOutput(BaseModel):
    """Output from math agent."""
    answer: str


async def main():
    """Run the example."""
    # Create an agent
    agent = Agent(
        name="math_agent",
        description="Solves simple math problems using addition and multiplication",
        input_schema=MathInput,
        output_schema=MathOutput,
        tools=[
            add,
            multiply,
            LLM("gpt-4o-mini"),
            Done(MathOutput)
        ],
        terminal_tools=["done"]
    )
    
    # Create runtime
    runtime = Runtime()
    
    print("=== Math Agent Example ===\n")
    
    # Example 1: AOT compilation
    print("1. AOT Compilation (cached):")
    compiled = await runtime.aot(agent, cache=True)
    result = await compiled(problem="What is 2 + 2?")
    print(f"   Input: What is 2 + 2?")
    print(f"   Output: {result}\n")
    
    # Example 2: JIT execution
    print("2. JIT Execution (on-the-fly):")
    result = await runtime.jit(agent, MathInput(problem="What is 5 * 3?"))
    print(f"   Input: What is 5 * 3?")
    print(f"   Output: {result}\n")
    
    # Example 3: Using context manager
    print("3. Using Runtime context manager:")
    with Runtime() as rt:
        result = await rt.jit(agent, MathInput(problem="What is (2 + 3) * 4?"))
        print(f"   Input: What is (2 + 3) * 4?")
        print(f"   Output: {result}\n")
    
    print("=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
