"""
Tests for code verification and validation module.
"""

import pytest
from a1.codecheck import (
    BaseVerify,
    IsLoop,
    check_code_candidate,
    check_syntax,
    check_dangerous_ops,
)
from a1 import Agent, Tool, tool
from pydantic import BaseModel


class TestBaseVerify:
    """Test BaseVerify strategy."""
    
    def test_valid_code(self):
        """Test verification of valid code."""
        code = """
x = 1
y = 2
result = x + y
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid
        assert error is None
    
    def test_syntax_error(self):
        """Test detection of syntax errors."""
        code = "this is not valid python!!!"
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Syntax error" in error
    
    def test_dangerous_import_os(self):
        """Test detection of dangerous os import."""
        code = """
import os
os.system("rm -rf /")
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Dangerous import" in error
    
    def test_dangerous_import_subprocess(self):
        """Test detection of dangerous subprocess import."""
        code = """
import subprocess
subprocess.run(["ls"])
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Dangerous import" in error
    
    def test_dangerous_function_eval(self):
        """Test detection of eval function."""
        code = """
result = eval("1 + 1")
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Dangerous function" in error
    
    def test_dangerous_function_exec(self):
        """Test detection of exec function."""
        code = """
exec("print('hello')")
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Dangerous function" in error
    
    def test_safe_imports_allowed(self):
        """Test that safe imports are allowed."""
        code = """
import json
import re
from typing import List
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid
    
    def test_async_await_allowed(self):
        """Test that async/await syntax is allowed."""
        code = """
async def test():
    result = await some_tool(x=1)
    return result
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid


class TestIsLoop:
    """Test IsLoop verifier."""
    
    def test_valid_loop_pattern(self):
        """Test verification of valid agentic loop."""
        code = """
while True:
    result = await llm(prompt="What should I do?")
    if result.done:
        break
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert is_valid
        assert error is None
    
    def test_missing_while_true(self):
        """Test detection of missing while True."""
        code = """
result = await llm(prompt="test")
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Not a standard agentic loop" in error
    
    def test_missing_llm_call(self):
        """Test detection of missing LLM call."""
        code = """
while True:
    x = 1
    break
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
    
    def test_missing_break(self):
        """Test detection of missing break."""
        code = """
while True:
    result = await llm(prompt="test")
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
    
    def test_syntax_error(self):
        """Test handling of syntax errors."""
        code = "while True invalid"
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Invalid syntax" in error


class TestCheckCodeCandidate:
    """Test check_code_candidate function."""
    
    def test_valid_code(self):
        """Test validation of valid code."""
        code = """
x = 1
y = 2
result = x + y
"""
        is_valid, error = check_code_candidate(code)
        assert is_valid
        assert error is None
    
    def test_invalid_syntax(self):
        """Test validation fails on syntax error."""
        code = "this is not valid x ="
        is_valid, error = check_code_candidate(code)
        assert not is_valid
        assert error is not None
    
    def test_dangerous_code(self):
        """Test validation fails on dangerous code."""
        code = """
import os
os.system("rm -rf /")
"""
        is_valid, error = check_code_candidate(code)
        assert not is_valid
        assert error is not None
    
    def test_with_custom_verifiers(self):
        """Test validation with custom verifiers."""
        code = """
while True:
    result = await llm(prompt="test")
    break
"""
        # Should pass basic verification
        is_valid, error = check_code_candidate(code)
        assert is_valid
        
        # Should also pass loop verification
        is_valid, error = check_code_candidate(code, verifiers=[IsLoop()])
        assert is_valid
    
    def test_custom_verifier_failure(self):
        """Test that custom verifier can reject code."""
        code = """
result = await tool(x=1)
"""
        # Should pass basic verification
        is_valid, error = check_code_candidate(code)
        assert is_valid
        
        # Should fail loop verification
        is_valid, error = check_code_candidate(code, verifiers=[IsLoop()])
        assert not is_valid


class TestCheckSyntax:
    """Test check_syntax function."""
    
    def test_valid_syntax(self):
        """Test valid Python syntax."""
        code = "x = 1 + 2"
        is_valid, error = check_syntax(code)
        assert is_valid
        assert error is None
    
    def test_invalid_syntax(self):
        """Test invalid Python syntax."""
        code = "x = 1 +"
        is_valid, error = check_syntax(code)
        assert not is_valid
        assert "Syntax error" in error
        assert "line" in error
    
    def test_multiline_valid(self):
        """Test valid multiline code."""
        code = """
def test():
    x = 1
    return x
"""
        is_valid, error = check_syntax(code)
        assert is_valid
    
    def test_multiline_invalid(self):
        """Test invalid multiline code."""
        code = """
def test()
    x = 1
"""
        is_valid, error = check_syntax(code)
        assert not is_valid


class TestCheckDangerousOps:
    """Test check_dangerous_ops function."""
    
    def test_safe_code(self):
        """Test safe code passes."""
        code = """
import json
x = json.dumps({"a": 1})
"""
        is_safe, error = check_dangerous_ops(code)
        assert is_safe
        assert error is None
    
    def test_dangerous_eval(self):
        """Test eval is detected."""
        code = "x = eval('1+1')"
        is_safe, error = check_dangerous_ops(code)
        assert not is_safe
        assert "eval" in error
    
    def test_dangerous_exec(self):
        """Test exec is detected."""
        code = "exec('print(1)')"
        is_safe, error = check_dangerous_ops(code)
        assert not is_safe
        assert "exec" in error
    
    def test_dangerous_os_import(self):
        """Test os import is detected."""
        code = "import os"
        is_safe, error = check_dangerous_ops(code)
        assert not is_safe
        assert "os" in error
    
    def test_dangerous_subprocess_import(self):
        """Test subprocess import is detected."""
        code = "from subprocess import run"
        is_safe, error = check_dangerous_ops(code)
        assert not is_safe
        assert "subprocess" in error
    
    def test_safe_standard_library(self):
        """Test safe stdlib imports are allowed."""
        code = """
import json
import re
from typing import List
from dataclasses import dataclass
"""
        is_safe, error = check_dangerous_ops(code)
        assert is_safe
    
    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        code = "this is invalid x ="
        is_safe, error = check_dangerous_ops(code)
        assert not is_safe
        assert "Invalid syntax" in error
