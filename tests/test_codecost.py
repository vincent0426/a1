"""
Tests for code cost estimation module.
"""

import pytest
from a1.codecost import (
    compute_code_cost,
    compute_block_cost,
    BaseCost,
    TOOL_LATENCIES,
)
from a1.cfg_builder import CFGBuilder, BasicBlock
from a1 import Agent, Tool, tool
from pydantic import BaseModel


class TestComputeCodeCost:
    """Test code cost computation."""
    
    def test_simple_tool_call(self):
        """Test cost for simple tool call."""
        code = """
result = await llm(prompt="test")
"""
        cost = compute_code_cost(code, {"llm": 10.0})
        assert cost == 10.0
    
    def test_multiple_tool_calls(self):
        """Test cost for multiple tool calls."""
        code = """
x = await search(query="test")
y = await llm(prompt="analyze")
"""
        costs = {"search": 5.0, "llm": 10.0}
        cost = compute_code_cost(code, costs)
        assert cost == 15.0
    
    def test_loop_multiplier(self):
        """Test cost multiplier for loops."""
        code = """
for item in items:
    result = await llm(prompt=item)
"""
        cost = compute_code_cost(code, {"llm": 10.0})
        # Base cost 10.0 * loop multiplier 10 = 100.0
        assert cost == 100.0
    
    def test_nested_loops(self):
        """Test cost for nested loops."""
        code = """
for i in range(10):
    for j in range(5):
        result = await tool(i, j)
"""
        cost = compute_code_cost(code, {"tool": 1.0})
        # Base cost 1.0 * 10^2 = 100.0
        assert cost == 100.0
    
    def test_comprehension_cost(self):
        """Test cost for list comprehension."""
        code = """
results = [await tool(x) for x in items]
"""
        cost = compute_code_cost(code, {"tool": 1.0})
        # Comprehension depth 1 = multiplier 10
        assert cost == 10.0
    
    def test_nested_comprehension(self):
        """Test cost for nested comprehension."""
        code = """
results = [await tool(x, y) for x in items for y in x.subitems]
"""
        cost = compute_code_cost(code, {"tool": 1.0})
        # Two generators = depth 2 = multiplier 100
        assert cost == 100.0
    
    def test_mixed_loop_comprehension(self):
        """Test cost for loop containing comprehension."""
        code = """
for item in items:
    results = [await tool(x) for x in item.children]
"""
        cost = compute_code_cost(code, {"tool": 1.0})
        # Loop depth 1 + comprehension depth 1 = total depth 2
        assert cost == 100.0
    
    def test_unknown_tool_zero_cost(self):
        """Test that unknown tools have zero cost."""
        code = """
result = await unknown_tool(param="value")
"""
        cost = compute_code_cost(code, {})
        assert cost == 0.0
    
    def test_syntax_error_fallback(self):
        """Test fallback to code length on syntax error."""
        code = "this is not valid python!!!"
        cost = compute_code_cost(code)
        assert cost == float(len(code))
    
    def test_branching_takes_max_path(self):
        """Test that branching takes maximum cost path."""
        code = """
if condition:
    await expensive_tool()  
else:
    await cheap_tool()
"""
        costs = {"expensive_tool": 100.0, "cheap_tool": 1.0}
        cost = compute_code_cost(code, costs)
        # Should take expensive path
        assert cost == 100.0


class TestBaseCost:
    """Test BaseCost strategy class."""
    
    def test_simple_cost_creation(self):
        """Test creating BaseCost with custom costs."""
        strategy = BaseCost(
            tool_costs={"custom_tool": 50.0},
            loop_multiplier=5.0
        )
        
        assert strategy.tool_costs["custom_tool"] == 50.0
        assert strategy.loop_multiplier == 5.0
    
    def test_simple_cost_compute(self):
        """Test computing cost with BaseCost."""
        # Create mock agent
        class Input(BaseModel):
            x: int
        
        @tool(name="test_tool")
        async def test_tool(x: int) -> int:
            return x * 2
        
        agent = Agent(
            name="test",
            description="Test agent",
            input_schema=Input,
            output_schema=Input,
            tools=[test_tool]
        )
        
        strategy = BaseCost(tool_costs={"test_tool": 10.0})
        
        code = """
result = await test_tool(x=5)
"""
        cost = strategy.compute_cost(code, agent)
        assert cost == 10.0
    
    def test_simple_cost_with_agent_tools(self):
        """Test that BaseCost uses agent tools for cost map."""
        class Input(BaseModel):
            x: int
        
        @tool(name="tool_a")
        async def tool_a(x: int) -> int:
            return x
        
        @tool(name="tool_b")
        async def tool_b(x: int) -> int:
            return x
        
        agent = Agent(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Input,
            tools=[tool_a, tool_b]
        )
        
        strategy = BaseCost()
        
        code = """
await tool_a(x=1)
await tool_b(x=2)
"""
        cost = strategy.compute_cost(code, agent)
        # Unknown tools default to 1.0 each
        assert cost == 2.0


class TestComputeBlockCost:
    """Test computing cost for individual blocks."""
    
    def test_block_with_no_calls(self):
        """Test block with no function calls."""
        block = BasicBlock(bid=1)
        cost = compute_block_cost(block, loop_depth=0)
        assert cost == 0.0
    
    def test_block_with_calls(self):
        """Test block with function calls."""
        block = BasicBlock(bid=1)
        # Add calls: (func_name, call_node, comp_depth)
        block.calls.append(("llm", None, 0))
        block.calls.append(("search", None, 0))
        
        costs = {"llm": 10.0, "search": 5.0}
        cost = compute_block_cost(block, loop_depth=0, tool_name_to_cost=costs)
        assert cost == 15.0
    
    def test_block_in_loop(self):
        """Test block cost when in a loop."""
        block = BasicBlock(bid=1)
        block.calls.append(("tool", None, 0))
        
        cost = compute_block_cost(block, loop_depth=1, tool_name_to_cost={"tool": 1.0})
        # Base 1.0 * 10^1 = 10.0
        assert cost == 10.0
    
    def test_block_with_comprehension_depth(self):
        """Test block cost with comprehension depth."""
        block = BasicBlock(bid=1)
        block.calls.append(("tool", None, 2))  # comp_depth=2
        
        cost = compute_block_cost(block, loop_depth=0, tool_name_to_cost={"tool": 1.0})
        # Base 1.0 * 10^2 = 100.0
        assert cost == 100.0
    
    def test_block_combined_depths(self):
        """Test block with both loop and comprehension depth."""
        block = BasicBlock(bid=1)
        block.calls.append(("tool", None, 1))  # comp_depth=1
        
        cost = compute_block_cost(block, loop_depth=1, tool_name_to_cost={"tool": 1.0})
        # Base 1.0 * 10^(1+1) = 100.0
        assert cost == 100.0
