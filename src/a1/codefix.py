"""
Code fixing utilities to transform generated code into correct executable code.

Handles common issues in LLM-generated code:
- asyncio.run() calls in async context
- Missing imports
- Incorrect function signatures
"""

import re
import ast
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def fix_asyncio_run(code: str) -> str:
    """
    Replace asyncio.run() calls with await.
    
    LLMs sometimes generate asyncio.run(func()) which fails in async context.
    We need to replace it with: output = await func()
    
    Args:
        code: Python code string
    
    Returns:
        Fixed code with asyncio.run replaced
    """
    # Pattern: asyncio.run(anything())
    # Replace with: output = await anything()
    pattern = r'asyncio\.run\s*\(\s*([^)]+)\s*\(([^)]*)\)\s*\)'
    
    def replacement(match):
        func_name = match.group(1).strip()
        func_args = match.group(2).strip()
        if func_args:
            return f'output = await {func_name}({func_args})'
        else:
            return f'output = await {func_name}()'
    
    fixed = re.sub(pattern, replacement, code)
    
    if fixed != code:
        logger.info("Fixed asyncio.run() call in generated code")
    
    return fixed


def extract_function_name(code: str) -> Optional[str]:
    """
    Extract the name of the first async function definition in code.
    
    Args:
        code: Python code string
    
    Returns:
        Function name or None if no async function found
    """
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef):
                return node.name
        return None
    except SyntaxError:
        return None


def has_function_call(code: str, func_name: str) -> bool:
    """
    Check if code contains a call to the given function.
    
    Args:
        code: Python code string
        func_name: Function name to look for
    
    Returns:
        True if function is called in the code
    """
    # Look for func_name( or await func_name(
    pattern = rf'\b{re.escape(func_name)}\s*\('
    return bool(re.search(pattern, code))


def append_function_call(code: str, func_name: str, call_args: str = "**validated.model_dump()") -> str:
    """
    Append a function call to code if not already present.
    
    Args:
        code: Python code string
        func_name: Function name to call
        call_args: Arguments to pass to function
    
    Returns:
        Code with function call appended
    """
    if has_function_call(code, func_name):
        logger.debug(f"Function {func_name} already called in code")
        return code
    
    logger.info(f"Appending call to {func_name}()")
    return code + f"\n\noutput = await {func_name}({call_args})"


def fix_generated_code(code: str, is_aot: bool = False) -> str:
    """
    Apply all code fixes to generated code.
    
    Args:
        code: Raw generated code
        is_aot: True if this is AOT mode (function expected), False for JIT (code block)
    
    Returns:
        Fixed code ready for execution
    """
    # First, fix asyncio.run() calls
    code = fix_asyncio_run(code)
    
    # For AOT mode with function definitions, append call if needed
    if is_aot:
        func_name = extract_function_name(code)
        if func_name:
            code = append_function_call(code, func_name)
    
    return code
