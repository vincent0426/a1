"""Quick test to verify RetryStrategy implementation."""

import asyncio
from pydantic import BaseModel, Field
from a1 import LLM, RetryStrategy


class MathResult(BaseModel):
    """Test output schema."""
    answer: float
    explanation: str


async def test_retry():
    """Test LLM with RetryStrategy."""
    
    # Create LLM with retry strategy
    retry_strategy = RetryStrategy(max_iterations=3, num_candidates=2)
    llm = LLM("groq:openai/gpt-oss-20b", retry_strategy=retry_strategy)
    
    # Call it
    result = await llm(
        content="What is 15 + 27?",
        output_schema=MathResult
    )
    
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    
    if isinstance(result, MathResult):
        print(f"✓ Got structured output!")
        print(f"  Answer: {result.answer}")
        print(f"  Explanation: {result.explanation}")
    else:
        print(f"✗ Got string instead: {result}")


if __name__ == "__main__":
    asyncio.run(test_retry())
