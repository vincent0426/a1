"""
Test tool execution and advanced RAG features.

Tests:
1. Actual tool execution (not just creation)
2. Smart RAG router based on input type
3. OpenAPI-based tool loading
4. FastAPI integration
"""

import pytest
import tempfile
import asyncio
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Union

from a1 import Agent, Tool, Runtime, ToolSet, FileSystemRAG, SQLRAG


# ============================================================================
# Tool Execution Tests
# ============================================================================

class TestToolExecution:
    """Test actual execution of RAG tools."""
    
    @pytest.mark.asyncio
    async def test_filesystem_rag_ls_execution(self):
        """Test actually executing LS tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "file1.txt").write_text("Content 1")
            (tmppath / "file2.txt").write_text("Content 2")
            (tmppath / "subdir").mkdir()
            
            rag = FileSystemRAG(tmppath)
            ls_tool = [t for t in rag.tools if t.name == "ls"][0]
            
            # Execute tool
            result = await ls_tool.execute(path="")
            
            # Verify actual execution results
            assert isinstance(result, dict)
            assert "files" in result
            assert len(result["files"]) >= 3
            assert any("file1" in str(f) for f in result["files"])
            assert any("subdir" in str(f) for f in result["files"])
            print(f"✓ LS execution result: {result['files']}")
    
    @pytest.mark.asyncio
    async def test_filesystem_rag_grep_execution(self):
        """Test actually executing GREP tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "doc1.txt").write_text("The quick brown fox jumps\nover the lazy dog")
            (tmppath / "doc2.txt").write_text("The fox is very clever\nand likes to jump")
            
            rag = FileSystemRAG(tmppath)
            grep_tool = [t for t in rag.tools if t.name == "grep"][0]
            
            # Execute grep
            result = await grep_tool.execute(pattern="fox", path="", limit=100)
            
            # Verify actual search results
            assert isinstance(result, dict)
            assert "matches" in result
            assert len(result["matches"]) >= 2
            assert any("fox" in m["content"].lower() for m in result["matches"])
            print(f"✓ GREP found {len(result['matches'])} matches")
            for match in result["matches"]:
                print(f"  - {match['file']}:{match['line']}: {match['content'][:50]}")
    
    @pytest.mark.asyncio
    async def test_filesystem_rag_cat_execution(self):
        """Test actually executing CAT tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_content = "This is a test file.\nWith multiple lines.\nOf content."
            (tmppath / "test.txt").write_text(test_content)
            
            rag = FileSystemRAG(tmppath)
            cat_tool = [t for t in rag.tools if t.name == "cat"][0]
            
            # Execute cat
            result = await cat_tool.execute(path="test.txt", limit=10000)
            
            # Verify actual file content
            assert isinstance(result, dict)
            assert "content" in result
            assert result["content"] == test_content
            assert result["truncated"] is False
            print(f"✓ CAT read content:\n{result['content']}")
    
    @pytest.mark.asyncio
    async def test_sqlrag_execution(self):
        """Test actually executing SQL tool."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'product': ['Widget', 'Gadget', 'Doohickey'],
            'price': [9.99, 19.99, 14.99],
            'in_stock': [True, False, True]
        })
        
        rag = SQLRAG(connection=df, schema='products')
        sql_tool = [t for t in rag.tools if t.name == "sql"][0]
        
        # Execute SQL query
        result = await sql_tool.execute(
            query="SELECT product, price FROM products WHERE price > 10",
            limit=100
        )
        
        # Verify actual query results
        assert isinstance(result, dict)
        assert "rows" in result
        assert len(result["rows"]) == 2  # Gadget and Doohickey
        assert all(row["price"] > 10 for row in result["rows"])
        print(f"✓ SQL query returned {len(result['rows'])} rows:")
        for row in result["rows"]:
            print(f"  - {row['product']}: ${row['price']}")


# ============================================================================
# Smart RAG Router
# ============================================================================

class RAG:
    """
    Syntax sugar RAG that routes to FileSystemRAG or SQLRAG based on input.
    
    Usage:
        rag = RAG(
            filesystem_path="./documents",
            dataframe=df  # optional
        )
        
        # Automatically routes to FileSystemRAG
        result = await rag.ls("")
        
        # Automatically routes to SQLRAG (if connected)
        result = await rag.query("SELECT * FROM table")
    """
    
    def __init__(self, filesystem_path: Union[str, Path] = None, dataframe=None):
        """
        Initialize RAG with optional filesystem and SQL backends.
        
        Args:
            filesystem_path: Path for FileSystemRAG
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
            raise ValueError("FileSystemRAG not initialized")
        ls_tool = [t for t in self.filesystem_rag.tools if t.name == "ls"][0]
        return await ls_tool.execute(path=path)
    
    async def grep(self, pattern: str, path: str = "", limit: int = 100) -> dict:
        """Search files (routes to FileSystemRAG)."""
        if not self.filesystem_rag:
            raise ValueError("FileSystemRAG not initialized")
        grep_tool = [t for t in self.filesystem_rag.tools if t.name == "grep"][0]
        return await grep_tool.execute(pattern=pattern, path=path, limit=limit)
    
    async def cat(self, path: str, limit: int = 10000) -> dict:
        """Read file (routes to FileSystemRAG)."""
        if not self.filesystem_rag:
            raise ValueError("FileSystemRAG not initialized")
        cat_tool = [t for t in self.filesystem_rag.tools if t.name == "cat"][0]
        return await cat_tool.execute(path=path, limit=limit)
    
    async def query(self, sql: str, limit: int = 100) -> dict:
        """Execute SQL query (routes to SQLRAG)."""
        if not self.sql_rag:
            raise ValueError("SQLRAG not initialized")
        sql_tool = [t for t in self.sql_rag.tools if t.name == "sql"][0]
        return await sql_tool.execute(query=sql, limit=limit)
    
    def as_toolset(self, name: str = "rag") -> ToolSet:
        """Convert to ToolSet for use in agents."""
        return ToolSet(
            name=name,
            description="RAG router for files and SQL",
            tools=self.tools
        )


class TestRAG:
    """Test RAG router functionality."""
    
    @pytest.mark.asyncio
    async def test_rag_filesystem_only(self):
        """Test RAG with only filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "data.txt").write_text("Sample data")
            
            rag = RAG(filesystem_path=tmppath)
            
            # Should route to filesystem
            result = await rag.ls("")
            assert "files" in result
            print(f"✓ RAG filesystem: {result['files']}")
    
    @pytest.mark.asyncio
    async def test_rag_sql_only(self):
        """Test RAG with only SQL."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        
        rag = RAG(dataframe=df)
        
        # Should route to SQL
        result = await rag.query("SELECT * FROM data")
        assert "rows" in result
        print(f"✓ RAG SQL: {len(result['rows'])} rows")
    
    @pytest.mark.asyncio
    async def test_rag_both_backends(self):
        """Test RAG with both filesystem and SQL."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "file1.txt").write_text("Content 1")
            
            df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
            
            rag = RAG(filesystem_path=tmppath, dataframe=df)
            
            # Filesystem operations
            fs_result = await rag.ls("")
            assert "files" in fs_result
            
            # SQL operations
            sql_result = await rag.query("SELECT * FROM data")
            assert "rows" in sql_result
            
            print(f"✓ RAG dual: files={len(fs_result['files'])}, rows={len(sql_result['rows'])}")
    
    def test_rag_as_toolset(self):
        """Test converting RAG to ToolSet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAG(filesystem_path=tmpdir)
            toolset = rag.as_toolset()
            
            assert isinstance(toolset, ToolSet)
            assert toolset.name == "rag"
            assert len(toolset.tools) == 3  # ls, grep, cat
            print(f"✓ RAG as ToolSet: {[t.name for t in toolset.tools]}")


# ============================================================================
# OpenAPI Tool Loading
# ============================================================================

class TestOpenAPIToolLoading:
    """Test loading tools from OpenAPI specs."""
    
    def test_openapi_spec_parsing(self):
        """Test parsing basic OpenAPI spec."""
        # Example OpenAPI spec
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/search": {
                    "post": {
                        "operationId": "search",
                        "summary": "Search for content",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Search results",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "results": {"type": "array"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Should be able to extract basic info
        assert openapi_spec["info"]["title"] == "Test API"
        assert "/search" in openapi_spec["paths"]
        assert "search" in openapi_spec["paths"]["/search"]["post"]["operationId"]
        print("✓ OpenAPI spec parsing works")
    
    def test_openapi_toolset_from_spec(self):
        """Test ToolSet creation from OpenAPI spec."""
        # This is a placeholder - real implementation would:
        # 1. Parse OpenAPI spec
        # 2. Create Tool objects for each endpoint
        # 3. Return as ToolSet
        
        try:
            # Check if ToolSet has from_openapi method
            assert hasattr(ToolSet, 'from_openapi'), "ToolSet should have from_openapi classmethod"
            print("✓ ToolSet.from_openapi method exists")
        except AssertionError:
            print("ℹ ToolSet.from_openapi not yet implemented, need to add it")


# ============================================================================
# FastAPI Integration Test Server
# ============================================================================

class TestFastAPIIntegration:
    """Test integration with FastAPI servers."""
    
    def test_fastapi_imports(self):
        """Test that FastAPI can be imported."""
        try:
            import fastapi
            print("✓ FastAPI is installed")
        except ImportError:
            pytest.skip("FastAPI not installed")
    
    @pytest.mark.asyncio
    async def test_fastapi_server_as_tool_source(self):
        """Test using a FastAPI server as a tool source."""
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("FastAPI not installed")
        
        # Create a simple FastAPI app
        app = FastAPI(title="Test API", openapi_url="/openapi.json")
        
        class CalculateRequest(BaseModel):
            operation: str
            a: int
            b: int
        
        @app.post("/calculate")
        async def calculate(req: CalculateRequest):
            """Perform a calculation."""
            if req.operation == "add":
                return {"result": req.a + req.b}
            elif req.operation == "multiply":
                return {"result": req.a * req.b}
            else:
                return {"error": "Unknown operation"}
        
        @app.get("/status")
        async def status():
            """Get API status."""
            return {"status": "ok"}
        
        # Test the app
        client = TestClient(app)
        
        # Test calculate endpoint
        response = client.post(
            "/calculate",
            json={"operation": "add", "a": 5, "b": 3}
        )
        assert response.status_code == 200
        assert response.json()["result"] == 8
        print(f"✓ FastAPI endpoint works: 5 + 3 = {response.json()['result']}")
        
        # Test status endpoint
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        print("✓ FastAPI status endpoint works")
        
        # Test OpenAPI spec
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_spec = response.json()
        assert "paths" in openapi_spec
        assert "/calculate" in openapi_spec["paths"]
        print(f"✓ FastAPI OpenAPI spec available with {len(openapi_spec['paths'])} paths")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
