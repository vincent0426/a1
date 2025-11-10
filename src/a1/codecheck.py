"""
Code validation and verification for generated Python code.

Provides functions to check code syntax and safety before execution.
Includes CFG-based validation to ensure tool ordering respects preconditions.
"""

import ast
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Verify Base Class and Implementations
# ============================================================================

class Verify:
    """
    Base class for code verification strategies.
    
    Verifies that generated code is valid and safe to execute.
    """
    
    def verify(self, code: str, agent: Any) -> Tuple[bool, Optional[str]]:
        """
        Verify generated code.
        
        Args:
            code: Generated Python code
            agent: Agent specification
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        raise NotImplementedError


class BaseVerify(Verify):
    """
    Base verification implementation that checks:
    - Syntax validity
    - No dangerous operations
    - Uses available tools only
    """
    
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        'socket', 'urllib', 'requests', 'http',
        '__import__', 'eval', 'exec', 'compile'
    }
    
    def verify(self, code: str, agent: Any) -> Tuple[bool, Optional[str]]:
        """Verify code is syntactically valid and safe."""
        # Check syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check for dangerous operations
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                if module_name and any(dangerous in module_name for dangerous in self.DANGEROUS_MODULES):
                    return False, f"Dangerous import detected: {module_name}"
            
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ['eval', 'exec', 'compile', '__import__']:
                        return False, f"Dangerous function detected: {func_name}"
        
            # Also check for undefined names (references to names not defined in
            # the module and not part of builtins or the agent's tools/schemas).
            try:
                from .codecheck import NoUndefinedNames
                is_valid, error = NoUndefinedNames().verify(code, agent=agent)
                if not is_valid:
                    return False, error
            except Exception:
                # If undefined-name checking fails for any reason, don't block
                # verification here; downstream verifiers or execution will catch
                # issues. We swallow exceptions to avoid false negatives.
                pass

            return True, None


class IsLoop(Verify):
    """
    Verifies that code follows standard agentic loop pattern.
    
    When this verifier is present, the Runtime.aot() method can skip
    LLM generation and use a templated loop instead.
    
    Expected pattern:
    ```python
    while <condition>:
        result = await llm(...)
        if <terminal_condition>:
            break
    ```
    """
    
    def verify(self, code: str, agent: Any) -> Tuple[bool, Optional[str]]:
        """Check if code is a standard agentic loop."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, "Invalid syntax"
        
        # Look for while loop with LLM call and break condition
        has_while_loop = False
        has_llm_call = False
        has_break = False
        
        for node in ast.walk(tree):
            # Check for while loop (any condition, not just True)
            if isinstance(node, ast.While):
                has_while_loop = True
            
            # Check for LLM/tool calls (check both function name and variable name)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id.lower()
                    if 'llm' in func_name or 'gpt' in func_name or 'openai' in func_name:
                        has_llm_call = True
                elif isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr.lower()
                    if 'llm' in attr_name or 'gpt' in attr_name:
                        has_llm_call = True
            
            # Check for break
            if isinstance(node, ast.Break):
                has_break = True
        
        if has_while_loop and has_llm_call and has_break:
            return True, None
        else:
            missing = []
            if not has_while_loop:
                missing.append("while loop")
            if not has_llm_call:
                missing.append("LLM call")
            if not has_break:
                missing.append("break statement")
            return False, f"Not a standard agentic loop pattern (missing: {', '.join(missing)})"


# ============================================================================
# Helper Functions for Code Checking
# ============================================================================

def check_code_candidate(
    code: str,
    agent: Optional[Any] = None,
    verifiers: Optional[list] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if generated code candidate is valid.
    
    Performs:
    1. Syntax validation
    2. Safety checks (no dangerous imports/functions)
    3. Optional custom verifiers
    
    Args:
        code: Generated Python code
        agent: Optional agent object for context
        verifiers: Optional list of Verify instances to run
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Always run basic verification
    basic_verify = BaseVerify()
    is_valid, error = basic_verify.verify(code, agent)
    if not is_valid:
        return False, error
    
    # Run custom verifiers if provided
    if verifiers:
        for verifier in verifiers:
            is_valid, error = verifier.verify(code, agent)
            if not is_valid:
                return False, error
    
    return True, None


def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check if code has valid Python syntax.
    
    Args:
        code: Python code to check
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def check_dangerous_ops(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check for dangerous operations in code.
    
    Args:
        code: Python code to check
    
    Returns:
        Tuple of (is_safe, error_message)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "Invalid syntax"
    
    dangerous_funcs = {'eval', 'exec', 'compile', '__import__'}
    dangerous_modules = {
        'os', 'sys', 'subprocess', 'shutil',
        'socket', 'urllib', 'requests', 'http'
    }
    
    for node in ast.walk(tree):
        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in dangerous_funcs:
                return False, f"Dangerous function: {node.func.id}"
        
        # Check imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            if module and any(d in module for d in dangerous_modules):
                return False, f"Dangerous import: {module}"
    
    return True, None


class IsFunction(Verify):
    """
    Verify that code contains at least one async function definition that's not a stub.
    
    Used by AOT mode to ensure generated code is a proper function.
    Extracts function metadata (name, args, return type) for validation.
    Ignores stub functions that raise NotImplementedError.
    """
    
    def verify(self, code: str, **kwargs) -> tuple[bool, Optional[str]]:
        """Check if code has at least one async function (not a stub) and extract metadata."""
        # Parse code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Find async function definitions at module level, excluding stubs
        func_defs = []
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef):
                # Check if it's a stub (raises NotImplementedError)
                is_stub = False
                for stmt in node.body:
                    if isinstance(stmt, ast.Raise):
                        if isinstance(stmt.exc, ast.Call):
                            if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "NotImplementedError":
                                is_stub = True
                                break
                
                # Only count non-stub functions
                if not is_stub:
                    func_defs.append(node)
        
        if len(func_defs) == 0:
            return False, "No async function implementation found. AOT mode requires a function (stubs don't count)."
        
        # Check that the function is async (should always be true here)
        func_def = func_defs[0]  # Use the first non-stub function
        if not isinstance(func_def, ast.AsyncFunctionDef):
            return False, "Function must be async (use 'async def')"
        
        # Extract metadata for validation
        func_name = func_def.name
        func_args = [arg.arg for arg in func_def.args.args]
        
        # Get return annotation if present
        return_annotation = None
        if func_def.returns:
            if isinstance(func_def.returns, ast.Name):
                return_annotation = func_def.returns.id
            elif isinstance(func_def.returns, ast.Constant):
                return_annotation = func_def.returns.value
        
        # Log extracted metadata
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Function metadata - name: {func_name}, args: {func_args}, return: {return_annotation}")
        
        return True, None


class QualitativeCriteria(Verify):
    """
    LLM-based verification using natural language criteria.
    
    Prompts an LLM to evaluate code against qualitative criteria (returns boolean).
    Supports multiple samples with parallel execution and majority voting.
    
    Args:
        expression: Natural language criteria description (e.g., "Code is readable and well-structured")
        llm: Tool instance for LLM calls (e.g., LLM("gpt-4o-mini"))
        num_samples: Number of parallel LLM calls to make (default: 1)
        min_samples_for_aggregation: Minimum successful samples needed (default: 1)
        min_pass: Minimum samples that must return True for overall True (default: 1)
    """
    
    def __init__(
        self,
        expression: str,
        llm: Any,  # Tool
        num_samples: int = 1,
        min_samples_for_aggregation: int = 1,
        min_pass: int = 1
    ):
        self.expression = expression
        self.llm = llm
        self.num_samples = num_samples
        self.min_samples_for_aggregation = min_samples_for_aggregation
        self.min_pass = min_pass
    
    def verify(self, code: str, agent: Any) -> Tuple[bool, Optional[str]]:
        """
        Verify code using LLM-based qualitative criteria with optional sampling.
        
        Returns:
            Tuple of (passes_criteria, error_message)
        """
        import asyncio
        
        async def _verify_async():
            # Create evaluation prompt
            prompt = f"""Evaluate the following Python code against this criteria:

Criteria: {self.expression}

Code:
```python
{code}
```

Return ONLY "true" or "false" (lowercase, no quotes, no explanation).
"""
            
            # Run multiple samples in parallel if requested
            if self.num_samples <= 1:
                # Single sample
                result = await self.llm(content=prompt)
                # Handle both string and LLMOutput responses
                if isinstance(result, str):
                    response = result.strip().lower()
                elif hasattr(result, "content"):
                    response = (result.content or "").strip().lower()
                else:
                    response = result.get("content", "").strip().lower()
                
                if response == "true":
                    return True, None
                elif response == "false":
                    return False, f"Code does not meet criteria: {self.expression}"
                else:
                    return False, f"Invalid LLM response: {response}"
            
            # Multiple samples - run in parallel
            tasks = [self.llm(content=prompt) for _ in range(self.num_samples)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Parse results
            valid_responses = []
            for result in results:
                if isinstance(result, Exception):
                    continue
                # Handle both string and LLMOutput responses
                if isinstance(result, str):
                    response = result.strip().lower()
                elif hasattr(result, "content"):
                    response = (result.content or "").strip().lower()
                else:
                    response = result.get("content", "").strip().lower()
                if response in ("true", "false"):
                    valid_responses.append(response == "true")
            
            # Check if we have enough valid samples
            if len(valid_responses) < self.min_samples_for_aggregation:
                return False, f"Insufficient valid LLM responses: {len(valid_responses)}/{self.min_samples_for_aggregation}"
            
            # Count how many passed
            num_passed = sum(valid_responses)
            
            if num_passed >= self.min_pass:
                return True, None
            else:
                return False, f"Code failed criteria ({num_passed}/{len(valid_responses)} samples passed, need {self.min_pass}): {self.expression}"
        
        # Run async verification - handle both sync and async contexts
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # We are in an async context, so we need to await directly
            # Create a new task and wait for it
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _verify_async())
                return future.result()
        except RuntimeError:
            # No event loop running, use asyncio.run
            return asyncio.run(_verify_async())


__all__ = [
    "Verify",
    "BaseVerify",
    "QualitativeCriteria",
    "IsLoop",
    "IsFunction",
    "check_code_candidate",
    "check_syntax",
    "check_dangerous_ops",
]
