"""
Code generation module for agent code execution mode.

Provides Generate strategies that produce single code candidates.
Runtime orchestrates parallel generation, validation, and ranking.
"""

import json
import logging
import re
from typing import Optional, List, Tuple, Any

logger = logging.getLogger(__name__)


# Example code showing realistic patterns
EXAMPLE_CODE = """
E1: immediate return
```python
result = "The response to the user's task can be immediately returned"
```

E2: call single tool with handling of empty tool result
```python
x = await tool_a(42)
result = f"It is {x}" if x else "No results found"
```

E3: multiple tool calls
```python
await tool_c()
x = await tool_d(id=123)
y = await tool_e(p1=x)
result = f"Completed with result: {y}"
```

E4: control flow with loops and conditionals
```python
await tool_c()
x = await tool_e(123)
results = [await tool_f(item=item) for item in x if item > 3]
result = f"Summary of {len(results)} results"
```
"""

EXAMPLE_FUNCTION = """
E1: simple function
```python
async def process_data(input_data: str) -> str:
    x = await tool_a(input_data)
    return f"Processed: {x}"
```

E2: function with multiple tools
```python
async def complex_task(query: str, limit: int = 10) -> dict:
    results = await tool_b(q=query)
    filtered = [await tool_c(item) for item in results[:limit]]
    return {"count": len(filtered), "data": filtered}
```
"""

RULES = """
- Do NOT include comments in your code (except docstrings for functions)
- Do NOT redefine or reimplement tools - they are already available in scope
- Do NOT redefine Input/Output schemas - they are already defined above
- Call available tools using async/await
- Create instance of Output schema, assign to variable named 'output'
- Handle None/empty results gracefully
"""


# ============================================================================
# Generate Base Class and Implementations
# ============================================================================

class Generate:
    """
    Base class for code generation strategies.
    
    Generates a single code candidate as a string.
    Does NOT handle validation, cost estimation, or ranking - that's Runtime's job.
    """
    
    async def generate(
        self,
        agent: Any,  # Agent
        task: str,
        return_function: bool = False,
        past_attempts: Optional[List[Tuple[str, str]]] = None,
    ) -> Optional[str]:
        """
        Generate a single code candidate.
        
        Args:
            agent: Agent with tools available
            task: Task description (or function description for AOT)
            return_function: If True, generate a function definition. If False, generate code block.
            past_attempts: List of (candidate_code, validation_error) tuples for retry logic
        
        Returns:
            Generated code as string, or None if generation fails
        """
        raise NotImplementedError


class BaseGenerate(Generate):
    """
    Base code generation implementation using an LLM.
    
    Generates ONE candidate at a time. Runtime handles parallel generation.
    
    Args:
        llm_tool: Tool to use for code generation (e.g., LLM("gpt-4"))
        timezone: Timezone for timestamp context (default: "UTC")
    """
    
    def __init__(
        self,
        llm_tool: Any,  # Tool
        timezone: str = "UTC"
    ):
        self.llm_tool = llm_tool
        self.timezone = timezone
    
    async def generate(
        self,
        agent: Any,
        task: str,
        return_function: bool = False,
        past_attempts: Optional[List[Tuple[str, str]]] = None,
    ) -> Optional[str]:
        """Generate a single code candidate using LLM."""
        # Build definition code
        definition_code = self._build_definition_code(agent, return_function=return_function)
        
        # Build conversation history
        conversation = []
        
        # System message
        examples = EXAMPLE_FUNCTION if return_function else EXAMPLE_CODE
        system_msg = f"""You are an expert in writing Python code that calls tools.
<examples>
Here are examples of good code to generate. Use them as reference but never copy them directly.
{examples}
</examples>
<instructions>
Generate completion of the given code block to implement the TASK.
If an error is reported, fix the previously generated code accordingly.
</instructions>
"""
        conversation.append({"role": "system", "content": system_msg})
        
        # Build prompt
        prompt_parts = []
        
        # Add timestamp
        prompt_parts.extend(self._build_timestamp())
        
        # If there are past attempts, show the error
        if past_attempts:
            last_code, last_error = past_attempts[-1]
            prompt_parts.append(f"Previous attempt failed with error:\n{last_error}")
            prompt_parts.append("")
            prompt_parts.append("Please fix the code:")
            prompt_parts.append("```python")
            prompt_parts.append(last_code)
            prompt_parts.append("```")
        else:
            # Initial generation - show definitions and ask for completion
            prompt_parts.append("```python")
            prompt_parts.append(definition_code)
            prompt_parts.append("```")
            prompt_parts.append("")
            if return_function:
                prompt_parts.append(f"TASK: {task}")
                prompt_parts.append("")
                prompt_parts.append("Generate ONLY the function body to complete the async def shown above.")
                prompt_parts.append("Do NOT repeat the function signature or redefine schemas/tools.")
                prompt_parts.append("Just write the implementation inside the function.")
            else:
                prompt_parts.append(f"TASK: {task}")
                prompt_parts.append("")
                prompt_parts.append("Generate code to accomplish the task using the tools defined above.")
            # Note: We deliberately leave it open - the LLM will generate code
        
        prompt = '\n'.join(prompt_parts)
        conversation.append({"role": "user", "content": prompt})
        
        # Call LLM
        try:
            logger.info(f"Generating code for task: {task[:100]}...")
            # The LLM tool expects content
            # We pass the conversation as a JSON string for code generation
            messages_str = json.dumps(conversation)
            result = await self.llm_tool(content=messages_str)
            
            # Extract response content - handle both string and LLMOutput
            if isinstance(result, str):
                completion = result
            elif hasattr(result, 'content'):
                completion = result.content
            elif isinstance(result, dict) and 'content' in result:
                completion = result['content']
            else:
                completion = str(result)
            
            # Extract code from response
            code = self._extract_code_from_response(completion)
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None
    
    def _build_definition_code(self, agent: Any, return_function: bool = False) -> str:
        """Build definition code showing tool signatures and agent schemas."""
        lines = []
        
        # Import Pydantic for type definitions
        lines.append("from pydantic import BaseModel, Field")
        lines.append("from typing import Optional, List, Dict, Any, Union")
        lines.append("")
        lines.append("# RULES FOR CODE GENERATION:")
        lines.append("#  - Do NOT include comments in your code (except docstrings for functions)")
        lines.append("#  - Tools shown below are ALREADY AVAILABLE - just call them, don't import or redefine")
        lines.append("#  - Input and Output schemas shown below are ALREADY DEFINED - don't redefine them")
        lines.append("#  - Use async/await for all tool calls")
        lines.append("#  - Create instance of Output schema, assign to variable named 'output'")
        lines.append("#  - Handle None/empty results gracefully")
        if return_function:
            lines.append("#  - Complete the function body - function signature is already provided below")
        lines.append("")
        
        # Show agent's INPUT schema for function parameters
        if return_function and hasattr(agent.input_schema, '__name__') and hasattr(agent.input_schema, 'model_fields'):
            lines.append("# ============================================================================")
            lines.append("# AGENT INPUT SCHEMA - Function parameters MUST match these fields")
            lines.append("# ============================================================================")
            lines.append(f"class {agent.input_schema.__name__}(BaseModel):")
            for field_name, field_info in agent.input_schema.model_fields.items():
                # Get type annotation as string
                if hasattr(field_info.annotation, '__name__'):
                    field_type = field_info.annotation.__name__
                else:
                    field_type = str(field_info.annotation)
                    
                # Get description if available
                if hasattr(field_info, 'description') and field_info.description:
                    desc = field_info.description
                else:
                    desc = ""
                
                if field_info.is_required():
                    lines.append(f'    {field_name}: {field_type} = Field(..., description="{desc}")')
                else:
                    lines.append(f'    {field_name}: Optional[{field_type}] = Field(None, description="{desc}")')
            lines.append("")
        
        # Show agent's output schema that code must produce
        lines.append("# ============================================================================")
        lines.append("# AGENT OUTPUT SCHEMA - Your code must produce this type")
        lines.append("# ============================================================================")
        if hasattr(agent.output_schema, '__name__') and hasattr(agent.output_schema, 'model_fields'):
            lines.append(f"class {agent.output_schema.__name__}(BaseModel):")
            for field_name, field_info in agent.output_schema.model_fields.items():
                # Get type annotation as string
                if hasattr(field_info.annotation, '__name__'):
                    field_type = field_info.annotation.__name__
                else:
                    field_type = str(field_info.annotation)
                    
                # Get description if available
                if hasattr(field_info, 'description') and field_info.description:
                    desc = field_info.description
                else:
                    desc = ""
                
                if field_info.is_required():
                    lines.append(f'    {field_name}: {field_type} = Field(..., description="{desc}")')
                else:
                    lines.append(f'    {field_name}: Optional[{field_type}] = Field(None, description="{desc}")')
            lines.append("")
        
        lines.append("# ============================================================================")
        lines.append("# AVAILABLE TOOLS - Call these functions, don't implement them!")
        lines.append("# ============================================================================")
        lines.append("")
        
        # Add tool definitions
        for tool in agent.get_all_tools():
            # Skip LLM and Done tools - they're built-in
            if "llm" in tool.name.lower() or "done" in tool.name.lower():
                continue
                
            # Generate Pydantic model for input if schema exists
            if hasattr(tool, 'input_schema') and tool.input_schema:
                try:
                    if hasattr(tool.input_schema, 'model_json_schema'):
                        schema = tool.input_schema.model_json_schema()
                    elif isinstance(tool.input_schema, dict):
                        schema = tool.input_schema
                    else:
                        schema = None
                except Exception as e:
                    # Skip tools that can't be serialized to JSON schema
                    logger.warning(f"Skipping tool {tool.name} schema generation: {e}")
                    continue
                
                if schema and schema.get("properties"):
                    model_name = f"{tool.name.title().replace('_', '')}Input"
                    lines.append(f"class {model_name}(BaseModel):")
                    
                    properties = schema["properties"]
                    required = schema.get("required", [])
                    
                    for param_name, param_schema in properties.items():
                        param_type = self._json_type_to_python(param_schema.get("type", "str"))
                        is_required = param_name in required
                        param_desc = param_schema.get("description", "")
                        
                        if is_required:
                            lines.append(f'    {param_name}: {param_type} = Field(..., description="{param_desc}")')
                        else:
                            lines.append(f'    {param_name}: Optional[{param_type}] = Field(None, description="{param_desc}")')
                    
                    lines.append("")
            
            # Generate Pydantic model for output if schema exists  
            output_model_name = None
            if hasattr(tool, 'output_schema') and tool.output_schema:
                try:
                    if hasattr(tool.output_schema, 'model_json_schema'):
                        schema = tool.output_schema.model_json_schema()
                        output_model_name = tool.output_schema.__name__
                    elif isinstance(tool.output_schema, dict):
                        schema = tool.output_schema
                    else:
                        schema = None
                except Exception as e:
                    logger.warning(f"Skipping tool {tool.name} output schema: {e}")
                    schema = None
                
                if schema and schema.get("properties"):
                    if not output_model_name:
                        output_model_name = f"{tool.name.title().replace('_', '')}Output"
                    lines.append(f"class {output_model_name}(BaseModel):")
                    
                    properties = schema["properties"]
                    required = schema.get("required", [])
                    
                    for param_name, param_schema in properties.items():
                        param_type = self._json_type_to_python(param_schema.get("type", "str"))
                        is_required = param_name in required
                        param_desc = param_schema.get("description", "")
                        
                        if is_required:
                            lines.append(f'    {param_name}: {param_type} = Field(..., description="{param_desc}")')
                        else:
                            lines.append(f'    {param_name}: Optional[{param_type}] = Field(None, description="{param_desc}")')
                    
                    lines.append("")
            
            # Generate function signature with proper typing
            return_type = output_model_name if output_model_name else "Any"
            
            lines.append(f"async def {tool.name}(**kwargs) -> {return_type}:")
            lines.append(f'    """')
            lines.append(f'    {tool.description}')
            lines.append(f'    """')
            lines.append(f'    raise NotImplementedError("Provided by runtime")')
            lines.append('')
        
        # Add template function signature at the end for AOT mode
        if return_function:
            lines.append("# ============================================================================")
            lines.append("# COMPLETE THE FUNCTION BELOW - Input/Output schemas and tools already defined above")
            lines.append("# ============================================================================")
            
            # Build function signature from input schema
            if hasattr(agent.input_schema, 'model_fields'):
                params = []
                for field_name, field_info in agent.input_schema.model_fields.items():
                    # Get type annotation as string
                    if hasattr(field_info.annotation, '__name__'):
                        field_type = field_info.annotation.__name__
                    else:
                        field_type = str(field_info.annotation)
                    params.append(f"{field_name}: {field_type}")
                
                param_str = ", ".join(params)
            else:
                param_str = "**kwargs"
            
            # Get output schema name
            output_name = agent.output_schema.__name__ if hasattr(agent.output_schema, '__name__') else "Output"
            
            # Generate template signature
            lines.append(f"async def {agent.name}({param_str}) -> {output_name}:")
            if agent.description:
                lines.append(f'    """')
                lines.append(f'    {agent.description}')
                lines.append(f'    """')
            lines.append("    # YOUR CODE HERE - call tools and create Output instance")
        
        return '\n'.join(lines)
    
    def _json_type_to_python(self, json_type: str) -> str:
        """Convert JSON schema type to Python type hint."""
        type_map = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]",
        }
        return type_map.get(json_type, "Any")
    
    def _build_timestamp(self) -> List[str]:
        """Build timestamp string for context."""
        from datetime import datetime
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(self.timezone)
            now = datetime.now(tz)
            day_of_week = now.strftime("%A")
            date_part = now.strftime("%B %d, %Y")
            time_part = now.strftime("%I:%M:%S %p")
            tz_abbr = now.strftime("%Z")
            timestamp_str = f"{day_of_week}, {date_part} {time_part} {tz_abbr}"
            return [f"Now is {timestamp_str}.", ""]
        except Exception as e:
            logger.warning(f"Failed to get current time for timezone {self.timezone}: {e}")
            return []
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from markdown code blocks."""
        # Primary: fenced code blocks
        patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```py\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                extracted = matches[0].strip()
                if extracted:
                    return extracted
        
        # Fallback heuristics
        lines = [l.rstrip() for l in response.splitlines()]
        code_like = []
        in_code = False
        for l in lines:
            stripped = l.strip()
            if not stripped:
                continue
            # Start collecting when we see common Python starters
            if not in_code and (
                stripped.startswith(('result', 'await ', 'async def ', 'def ', 'for ', 'if ', 'while ', 'import ', 'from ', 'class ', '#'))
                or 'await ' in stripped
            ):
                in_code = True
            if in_code:
                code_like.append(l)
        fallback = '\n'.join(code_like).strip()
        if fallback:
            # Remove trailing triple backticks if model left them open
            fallback = re.sub(r'```+$', '', fallback).strip()
            return fallback if fallback else None
        return None


__all__ = [
    "Generate",
    "BaseGenerate",
    "EXAMPLE_CODE",
    "EXAMPLE_FUNCTION",
    "RULES",
]
