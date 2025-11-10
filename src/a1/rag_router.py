"""
Smart RAG router for syntax sugar access to multiple RAG backends.

Provides flexible access to FileSystemRAG and SQLRAG with automatic routing.
"""

from typing import Union, Optional, Any
from pathlib import Path

from .models import ToolSet
from .rag import FileSystemRAG, SQLRAG


class RAG:
    """
    Syntax sugar RAG that routes to FileSystemRAG or SQLRAG based on input.
    
    Provides convenient methods for file operations and SQL queries,
    automatically routing to the appropriate backend.
    
    Example:
        rag = RAG(
            filesystem_path="./documents",
            dataframe=df  # optional
        )
        
        # Automatically routes to FileSystemRAG
        result = await rag.ls("")
        result = await rag.grep("pattern", "path")
        result = await rag.cat("file.txt")
        
        # Automatically routes to SQLRAG (if connected)
        result = await rag.query("SELECT * FROM table")
    """
    
    def __init__(
        self,
        filesystem_path: Optional[Union[str, Path]] = None,
        dataframe: Optional[Any] = None,
    ):
        """
        Initialize RAG with optional filesystem and SQL backends.
        
        Args:
            filesystem_path: Path for FileSystemRAG (supports fsspec URLs)
            dataframe: pandas DataFrame for SQLRAG
        """
        self.filesystem_rag = None
        self.sql_rag = None
        self.tools = []
        
        if filesystem_path:
            self.filesystem_rag = FileSystemRAG(filesystem_path)
            self.tools.extend(self.filesystem_rag.tools)
        
        if dataframe is not None:
            try:
                import pandas as pd
                if isinstance(dataframe, pd.DataFrame):
                    self.sql_rag = SQLRAG(connection=dataframe)
                    self.tools.extend(self.sql_rag.tools)
            except ImportError:
                pass
    
    async def ls(self, path: str = "") -> dict:
        """List directory contents (routes to FileSystemRAG)."""
        if not self.filesystem_rag:
            raise ValueError("FileSystemRAG not initialized. Pass filesystem_path to RAG()")
        ls_tool = [t for t in self.filesystem_rag.tools if t.name == "ls"][0]
        return await ls_tool.execute(path=path)
    
    async def grep(self, pattern: str, path: str = "", limit: int = 100) -> dict:
        """Search files (routes to FileSystemRAG)."""
        if not self.filesystem_rag:
            raise ValueError("FileSystemRAG not initialized. Pass filesystem_path to RAG()")
        grep_tool = [t for t in self.filesystem_rag.tools if t.name == "grep"][0]
        return await grep_tool.execute(pattern=pattern, path=path, limit=limit)
    
    async def cat(self, path: str, limit: int = 10000) -> dict:
        """Read file (routes to FileSystemRAG)."""
        if not self.filesystem_rag:
            raise ValueError("FileSystemRAG not initialized. Pass filesystem_path to RAG()")
        cat_tool = [t for t in self.filesystem_rag.tools if t.name == "cat"][0]
        return await cat_tool.execute(path=path, limit=limit)
    
    async def query(self, sql: str, limit: int = 100) -> dict:
        """Execute SQL query (routes to SQLRAG)."""
        if not self.sql_rag:
            raise ValueError("SQLRAG not initialized. Pass dataframe to RAG()")
        sql_tool = [t for t in self.sql_rag.tools if t.name == "sql"][0]
        return await sql_tool.execute(query=sql, limit=limit)
    
    def as_toolset(self, name: str = "rag") -> ToolSet:
        """Convert to ToolSet for use in agents."""
        return ToolSet(
            name=name,
            description="RAG router for files and SQL",
            tools=self.tools
        )


__all__ = ["RAG"]
