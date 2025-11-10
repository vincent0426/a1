"""
RAG (Retrieval-Augmented Generation) toolsets.

Provides:
- FileSystemRAG: Tools for file operations (ls, grep, cat)
- SQLRAG: Tools for SQL queries (readonly)
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, create_model

from .models import Tool, ToolSet


# ============================================================================
# FileSystem RAG
# ============================================================================

class LSInput(BaseModel):
    """Input for ls tool."""
    path: str = Field(..., description="Directory path to list")


class LSOutput(BaseModel):
    """Output from ls tool."""
    files: list[str] = Field(..., description="List of files and directories")


class GrepInput(BaseModel):
    """Input for grep tool."""
    pattern: str = Field(..., description="Pattern to search for")
    path: str = Field(..., description="File or directory to search in")
    limit: int = Field(100, description="Maximum number of results")


class GrepOutput(BaseModel):
    """Output from grep tool."""
    matches: list[Dict[str, Any]] = Field(..., description="List of matches with file and line info")


class CatInput(BaseModel):
    """Input for cat tool."""
    path: str = Field(..., description="File path to read")
    limit: int = Field(10000, description="Maximum characters to read")


class CatOutput(BaseModel):
    """Output from cat tool."""
    content: str = Field(..., description="File content")
    truncated: bool = Field(False, description="Whether content was truncated")


def FileSystemRAG(filepath: Union[str, Path]) -> ToolSet:
    """
    Create a RAG toolset for filesystem operations.
    
    Uses fsspec for flexible filesystem access (local, s3, gcs, etc.)
    
    Args:
        filepath: Base path or fsspec URL (e.g., "s3://bucket/prefix")
    
    Returns:
        ToolSet with ls, grep, cat tools
    """
    import fsspec
    
    # Parse filesystem from path
    fs, base_path = fsspec.core.url_to_fs(str(filepath))
    
    # LS tool
    async def ls_execute(path: str) -> Dict[str, Any]:
        """List directory contents."""
        full_path = f"{base_path}/{path}".rstrip("/")
        try:
            files = fs.ls(full_path, detail=False)
            # Make paths relative to base
            rel_files = [f.replace(base_path + "/", "") for f in files]
            return {"files": rel_files}
        except Exception as e:
            return {"files": [], "error": str(e)}
    
    ls_tool = Tool(
        name="ls",
        description="List files and directories",
        input_schema=LSInput,
        output_schema=LSOutput,
        execute=ls_execute
    )
    
    # GREP tool
    async def grep_execute(pattern: str, path: str, limit: int = 100) -> Dict[str, Any]:
        """Search for pattern in files."""
        import re
        
        full_path = f"{base_path}/{path}".rstrip("/")
        matches = []
        
        try:
            # If path is a file, search it
            if fs.isfile(full_path):
                files = [full_path]
            else:
                # If directory, get all files recursively
                files = fs.find(full_path)
            
            pattern_re = re.compile(pattern, re.IGNORECASE)
            
            for file in files:
                if len(matches) >= limit:
                    break
                
                try:
                    content = fs.cat_file(file).decode('utf-8', errors='ignore')
                    for i, line in enumerate(content.split('\n'), 1):
                        if pattern_re.search(line):
                            rel_file = file.replace(base_path + "/", "")
                            matches.append({
                                "file": rel_file,
                                "line": i,
                                "content": line.strip()
                            })
                            if len(matches) >= limit:
                                break
                except Exception:
                    continue
            
            return {"matches": matches}
        except Exception as e:
            return {"matches": [], "error": str(e)}
    
    grep_tool = Tool(
        name="grep",
        description="Search for pattern in files",
        input_schema=GrepInput,
        output_schema=GrepOutput,
        execute=grep_execute
    )
    
    # CAT tool
    async def cat_execute(path: str, limit: int = 10000) -> Dict[str, Any]:
        """Read file contents."""
        full_path = f"{base_path}/{path}".rstrip("/")
        
        try:
            content = fs.cat_file(full_path).decode('utf-8', errors='ignore')
            
            truncated = len(content) > limit
            if truncated:
                content = content[:limit]
            
            return {"content": content, "truncated": truncated}
        except Exception as e:
            return {"content": "", "truncated": False, "error": str(e)}
    
    cat_tool = Tool(
        name="cat",
        description="Read file contents",
        input_schema=CatInput,
        output_schema=CatOutput,
        execute=cat_execute
    )
    
    return ToolSet(
        name=f"filesystem_{Path(filepath).name}",
        description=f"File operations on {filepath}",
        tools=[ls_tool, grep_tool, cat_tool]
    )


# ============================================================================
# SQL RAG
# ============================================================================

class SQLQueryInput(BaseModel):
    """Input for SQL query tool."""
    query: str = Field(..., description="SQL query to execute (SELECT only)")
    limit: int = Field(100, description="Maximum rows to return")


class SQLQueryOutput(BaseModel):
    """Output from SQL query."""
    rows: list[Dict[str, Any]] = Field(..., description="Query results as list of dicts")
    columns: list[str] = Field(..., description="Column names")
    row_count: int = Field(..., description="Number of rows returned")


def SQLRAG(
    connection: Union[str, Any],  # Connection string or DataFrame
    schema: Optional[str] = None
) -> ToolSet:
    """
    Create a RAG toolset for SQL queries.
    
    Supports:
    - SQLAlchemy connection strings
    - pandas DataFrames (creates in-memory SQLite)
    
    Args:
        connection: SQLAlchemy connection string or pandas DataFrame
        schema: Optional schema name (for DataFrames, becomes the table name)
    
    Returns:
        ToolSet with sql query tool (readonly)
    """
    from sqlalchemy import create_engine, text
    import pandas as pd
    
    # Handle DataFrame input
    table_name = None  # Track table name for DataFrames
    if isinstance(connection, pd.DataFrame):
        # Create in-memory SQLite database
        engine = create_engine("sqlite:///:memory:")
        table_name = schema or "data"
        connection.to_sql(table_name, engine, index=False)
    else:
        # Use connection string
        engine = create_engine(connection)
    
    # SQL query tool
    async def sql_execute(query: str, limit: int = 100) -> Dict[str, Any]:
        """Execute SQL query (SELECT only)."""
        # Security: Only allow SELECT queries
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            return {
                "rows": [],
                "columns": [],
                "row_count": 0,
                "error": "Only SELECT queries are allowed"
            }
        
        # Add LIMIT if not present
        if "LIMIT" not in query_upper:
            query = f"{query.rstrip(';')} LIMIT {limit}"
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                rows = [dict(row._mapping) for row in result]
                columns = list(result.keys()) if result.returns_rows else []
                
                return {
                    "rows": rows,
                    "columns": columns,
                    "row_count": len(rows)
                }
        except Exception as e:
            return {
                "rows": [],
                "columns": [],
                "row_count": 0,
                "error": str(e)
            }
    
    sql_tool = Tool(
        name="sql",
        description="Execute SQL query (SELECT only, readonly)",
        input_schema=SQLQueryInput,
        output_schema=SQLQueryOutput,
        execute=sql_execute
    )
    
    return ToolSet(
        name="sql_rag",
        description="SQL query toolset (readonly)",
        tools=[sql_tool]
    )


__all__ = [
    "FileSystemRAG",
    "SQLRAG",
]
