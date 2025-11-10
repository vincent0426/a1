"""
Code executors for running generated Python code.

Provides:
- Executor: Base class for code execution
- SimpleExecutor: Basic Python exec-based executor
"""

import ast
import asyncio
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Maximum iterations for loops to prevent infinite loops
MAX_LOOP_ITERATIONS = 10000


@dataclass
class CodeOutput:
    """Result of code execution."""
    output: Any
    logs: str
    is_final_answer: bool = False
    error: Optional[str] = None


class Executor:
    """
    Base class for code executors.
    
    Executors are responsible for running generated Python code
    in a controlled environment.
    """
    
    async def execute(self, code: str, tools: Optional[List[Any]] = None) -> CodeOutput:
        """
        Execute Python code and return the result.
        
        Args:
            code: Python code to execute
            tools: Optional list of Tool objects to make available during execution
        
        Returns:
            CodeOutput with result, logs, and any errors
        """
        raise NotImplementedError


class BaseExecutor(Executor):
    """
    Base executor implementation using Python's exec() with async support.
    
    Maintains state between executions and provides access to custom functions.
    Captures print outputs and supports async/await natively.
    
    WARNING: This executor does NOT provide sandboxing. Only use with
    trusted code generation.
    
    Args:
        additional_functions: Dictionary of functions to make available to code
        additional_imports: Dictionary of module imports to make available
    """
    
    def __init__(
        self,
        additional_functions: Optional[Dict[str, Any]] = None,
        additional_imports: Optional[Dict[str, Any]] = None,
    ):
        self.state: Dict[str, Any] = {}
        self.additional_functions = additional_functions or {}
        self.additional_imports = additional_imports or {}
        
        # Initialize state with imports and functions
        self.state.update(self.additional_imports)
        self.state.update(self.additional_functions)
        
        # Track print output
        self.print_buffer: List[str] = []
    
    def _capture_print(self, *args, **kwargs):
        """Capture print() calls."""
        output = " ".join(str(arg) for arg in args)
        self.print_buffer.append(output)
        # Also print to actual stdout for debugging
        print(output, **kwargs)
    
    async def execute(self, code: str, tools: Optional[List[Any]] = None) -> CodeOutput:
        """
        Execute Python code asynchronously.
        
        Args:
            code: Python code to execute
            tools: Optional list of Tool objects to make available during execution
            
        Returns:
            CodeOutput with result, logs, and any errors
        """
        logger.info(f"\n{'='*80}\nEXECUTING CODE\n{'='*80}\n```python\n{code}\n```\n{'-'*80}")
        
        # Clear print buffer
        self.print_buffer = []
        
        # Create execution environment with tools available
        exec_env = self.state.copy()
        exec_env['print'] = self._capture_print
        
        # Add tools to execution environment by name
        if tools:
            for tool in tools:
                exec_env[tool.name] = tool
                # Also add the input/output schemas as classes
                if hasattr(tool, 'input_schema') and tool.input_schema:
                    # Add the schema class itself so code can instantiate it
                    exec_env[tool.input_schema.__name__] = tool.input_schema
                if hasattr(tool, 'output_schema') and tool.output_schema:
                    exec_env[tool.output_schema.__name__] = tool.output_schema
                # Provide common short aliases for tools to match model expectations.
                # e.g., models may call the LLM as 'llm' even though the tool name is
                # 'llm_groq_openai_gpt_oss_20b'. Add an 'llm' alias for any tool
                # whose name contains 'llm'. This keeps JIT execution running the
                # generated code (which is intended to run standalone) without
                # requiring the definition code to have been executed.
                try:
                    lname = tool.name.lower()
                    if 'llm' in lname:
                        exec_env['llm'] = tool
                except Exception:
                    pass
        
        # Compile code - wrap in async function to support top-level await
        try:
            # Extract __future__ imports (must be at the beginning of the file)
            future_imports = []
            remaining_code_lines = []
            for line in code.split('\n'):
                if line.strip().startswith('from __future__'):
                    future_imports.append(line)
                else:
                    remaining_code_lines.append(line)
            
            wrapped_code = ""
            if future_imports:
                wrapped_code += "\n".join(future_imports) + "\n\n"
            
            wrapped_code += "async def __exec_wrapper():\n"
            # Indent each line of the remaining code
            for line in remaining_code_lines:
                wrapped_code += f"    {line}\n"
            wrapped_code += "    return locals()\n"
            
            compiled = compile(wrapped_code, '<generated>', 'exec')
        except SyntaxError as e:
            return CodeOutput(
                output=None,
                logs="",
                is_final_answer=False,
                error=f"Compilation error: {e}"
            )
        
        # Execute in async context
        try:
            # Execute the wrapped function definition
            exec(compiled, exec_env, exec_env)
            
            # Get the wrapper function and call it
            wrapper_func = exec_env['__exec_wrapper']
            result_locals = await wrapper_func()
            
            # Check if the generated code defined an async function that should be called
            # This handles the case where the LLM generates a function instead of just code
            user_defined_async_funcs = [
                (name, func) for name, func in result_locals.items()
                if callable(func) and 
                asyncio.iscoroutinefunction(func) and 
                name not in ['__exec_wrapper'] and
                not name.startswith('_')
            ]
            
            # If there's exactly one user-defined async function AND no output was already set,
            # try to call it to get the result. Otherwise trust the wrapper set the 'output' variable.
            if user_defined_async_funcs and 'output' not in result_locals:
                func_name, user_func = user_defined_async_funcs[0]
                logger.info(f"Calling user-defined async function: {func_name}")
                try:
                    # Try calling with no arguments first
                    result = await user_func()
                    # If the function returns something, use that as the result
                    if result is not None:
                        return CodeOutput(
                            output=result,
                            logs="\n".join(self.print_buffer),
                            is_final_answer=False
                        )
                except TypeError as e:
                    # Function requires arguments - we can't call it without knowing what they are
                    # In this case, we'll fall through and try to find 'output' or 'result'
                    logger.debug(f"Cannot call {func_name} with no arguments: {e}")
                    pass
            
            # Update state with results (excluding the wrapper function itself and tools)
            result_locals.pop('__exec_wrapper', None)
            # Don't persist tools in state or their schemas
            if tools:
                for tool in tools:
                    result_locals.pop(tool.name, None)
                    if hasattr(tool, 'input_schema') and tool.input_schema:
                        result_locals.pop(tool.input_schema.__name__, None)
                    if hasattr(tool, 'output_schema') and tool.output_schema:
                        result_locals.pop(tool.output_schema.__name__, None)
            self.state.update(result_locals)
            
            # Find the result - prefer variables named 'output', then 'result', then last variable
            result = None
            if 'output' in result_locals:
                result = result_locals['output']
            elif 'result' in result_locals:
                result = result_locals['result']
            elif result_locals:
                # Get LAST value that's not a type/class
                for value in reversed(list(result_locals.values())):
                    if not isinstance(value, type):
                        result = value
                        break
            
            logs = "\n".join(self.print_buffer)
            
            logger.info(f"✓ CODE EXECUTED SUCCESSFULLY")
            logger.info(f"Output: {result}")
            logger.info(f"Logs:\n{logs}")
            logger.info(f"{'='*80}")
            
            return CodeOutput(
                output=result,
                logs=logs,
                is_final_answer=False
            )
            
        except Exception as e:
            logs = "\n".join(self.print_buffer)
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"✗ CODE EXECUTION ERROR: {error_msg}")
            logger.error(f"Logs:\n{logs}")
            logger.error(f"{'='*80}")
            
            return CodeOutput(
                output=None,
                logs=logs,
                is_final_answer=False,
                error=error_msg
            )
    
    def send_tools(self, tools: Dict[str, Any]):
        """Update available tools."""
        self.additional_functions.update(tools)
        self.state.update(tools)
    
    def send_variables(self, variables: Dict[str, Any]):
        """Update state variables."""
        self.state.update(variables)


__all__ = [
    "Executor",
    "BaseExecutor",
    "CodeOutput",
]
