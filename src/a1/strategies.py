"""
Strategy classes for code generation, verification, and cost estimation.

This module provides backward compatibility by re-exporting classes from:
- codegen: Code generation strategies  
- codecheck: Code verification strategies
- codecost: Cost estimation strategies
- cfg_builder: Control flow graph builder

For new code, prefer importing directly from the specific modules.
"""

# Re-export from individual modules for backward compatibility
from .codegen import (
    Generate,
    BaseGenerate,
    EXAMPLE_CODE,
    EXAMPLE_FUNCTION,
    RULES,
)

from .codecheck import (
    Verify,
    BaseVerify,
    QualitativeCriteria,
    IsLoop,
    IsFunction,
    check_code_candidate,
    check_syntax,
    check_dangerous_ops,
)

from .codecost import (
    Cost,
    BaseCost,
    QuantitativeCriteria,
    compute_code_cost,
    TOOL_LATENCIES,
    LOOP_MULTIPLIER,
)

from .cfg_builder import (
    CFGBuilder,
    BasicBlock,
)


__all__ = [
    # Generation
    "Generate",
    "BaseGenerate",
    "EXAMPLE_CODE",
    "EXAMPLE_FUNCTION",
    "RULES",
    # Verification
    "Verify",
    "BaseVerify",
    "QualitativeCriteria",
    "IsLoop",
    "IsFunction",
    "check_code_candidate",
    "check_syntax",
    "check_dangerous_ops",
    # Cost
    "Cost",
    "BaseCost",
    "QuantitativeCriteria",
    "compute_code_cost",
    "TOOL_LATENCIES",
    "LOOP_MULTIPLIER",
    # CFG
    "CFGBuilder",
    "BasicBlock",
]


