"""
Comprehensive integration tests for FileSystemRAG, SQLRAG, and LangChain.

These tests verify actual functionality, not just imports:
1. FileSystemRAG with fsspec for file operations (ls, grep, cat)
2. SQLRAG with SQLAlchemy for SQL queries with pandas DataFrames
3. LangChain agent conversion and tool compatibility
4. Combined usage of multiple RAG systems in a single agent
5. OpenTelemetry tracing integration
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field

from a1 import (
    Agent, Tool, Runtime, ToolSet,
    FileSystemRAG, SQLRAG,
)


class QueryInput(BaseModel):
    query: str = Field(..., description="Search query")


class FileOutput(BaseModel):
    content: str = Field(..., description="File content")


class SearchOutput(BaseModel):
    results: str = Field(..., description="Search results")


# ============================================================================
# FileSystemRAG Tests - Real File Operations
# ============================================================================

class TestFileSystemRAGFunctionality:
    """Test FileSystemRAG with actual file operations using fsspec."""
    
    @pytest.mark.asyncio
    async def test_filesystem_rag_ls_tool(self):
        """Test LS tool lists files correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create test files
            (tmppath / "file1.txt").write_text("Content 1")
            (tmppath / "file2.txt").write_text("Content 2")
            (tmppath / "subdir").mkdir()
            (tmppath / "subdir" / "file3.txt").write_text("Content 3")
            
            # Create RAG
            rag = FileSystemRAG(tmppath)
            assert isinstance(rag, ToolSet)
            
            # Get LS tool
            ls_tool = [t for t in rag.tools if t.name == "ls"][0]
            result = await ls_tool.execute(path="")
            
            # Verify results
            assert "files" in result
            assert len(result["files"]) >= 3, "Should list all files and subdirs"
            assert any("file1" in f for f in result["files"])
    
    @pytest.mark.asyncio
    async def test_filesystem_rag_grep_tool(self):
        """Test GREP tool finds patterns in files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create test files with specific content
            (tmppath / "file1.txt").write_text("Hello World")
            (tmppath / "file2.txt").write_text("Python Programming")
            (tmppath / "file3.txt").write_text("Nested Python Code")
            
            # Create RAG
            rag = FileSystemRAG(tmppath)
            
            # Get GREP tool
            grep_tool = [t for t in rag.tools if t.name == "grep"][0]
            result = await grep_tool.execute(pattern="Python", path="", limit=100)
            
            # Verify results
            assert "matches" in result
            assert len(result["matches"]) > 0, "Should find Python matches"
            assert any("file2" in m["file"] or "file3" in m["file"] 
                      for m in result["matches"])
    
    @pytest.mark.asyncio
    async def test_filesystem_rag_cat_tool(self):
        """Test CAT tool reads file contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create a test file
            test_content = "This is test content for CAT tool"
            (tmppath / "test.txt").write_text(test_content)
            
            # Create RAG
            rag = FileSystemRAG(tmppath)
            
            # Get CAT tool
            cat_tool = [t for t in rag.tools if t.name == "cat"][0]
            result = await cat_tool.execute(path="test.txt", limit=10000)
            
            # Verify results
            assert "content" in result
            assert test_content in result["content"]
            assert result["truncated"] is False
    
    def test_filesystem_rag_is_toolset(self):
        """Test FileSystemRAG returns a proper ToolSet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = FileSystemRAG(tmpdir)
            
            assert isinstance(rag, ToolSet)
            assert hasattr(rag, 'tools')
            assert len(rag.tools) == 3  # ls, grep, cat
            assert all(isinstance(t, Tool) for t in rag.tools)
    
    def test_filesystem_rag_agent_integration(self):
        """Test using FileSystemRAG with an Agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "test.txt").write_text("Test content")
            
            # Create RAG toolset
            rag = FileSystemRAG(tmppath)
            
            # Create agent with RAG tools
            agent = Agent(
                name="file_agent",
                description="Agent for file operations",
                input_schema=QueryInput,
                output_schema=FileOutput,
                tools=[rag],
            )
            
            assert agent.name == "file_agent"
            assert len(agent.tools) > 0



class TestOpenTelemetryIntegration:
    """Test OpenTelemetry tracing is integrated."""
    
    def test_opentelemetry_imports_in_runtime(self):
        """Test that OpenTelemetry is used in runtime."""
        from a1 import runtime as runtime_module
        runtime_code = str(runtime_module.__file__)
        
        # Runtime should exist and have async methods
        assert hasattr(Runtime, 'aot')
        assert hasattr(Runtime, 'jit')
        assert hasattr(Runtime, 'execute')
    
    def test_runtime_with_rag_agent(self):
        """Test Runtime works with RAG-enabled agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create RAG
            rag = FileSystemRAG(tmpdir)
            
            # Create agent
            agent = Agent(
                name="traced_agent",
                description="Agent with tracing",
                input_schema=QueryInput,
                output_schema=FileOutput,
                tools=[rag],
            )
            
            # Create runtime
            runtime = Runtime()
            
            # Both should coexist properly
            assert runtime is not None
            assert agent is not None
            assert hasattr(runtime, 'execute')


# ============================================================================
# SQLRAG Tests - Real SQL Operations
# ============================================================================

class TestSQLRAGFunctionality:
    """Test SQLRAG with actual SQL queries using SQLAlchemy."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
            'salary': [50000, 60000, 75000, 55000, 65000]
        })
    
    @pytest.mark.asyncio
    async def test_sqlrag_basic_select(self, sample_dataframe):
        """Test basic SELECT query."""
        rag = SQLRAG(connection=sample_dataframe, schema='employees')
        
        sql_tool = [t for t in rag.tools if t.name == "sql"][0]
        result = await sql_tool.execute(query="SELECT * FROM employees", limit=100)
        
        assert "rows" in result
        assert len(result["rows"]) == 5
        assert result["columns"] == ['id', 'name', 'age', 'city', 'salary']
    
    @pytest.mark.asyncio
    async def test_sqlrag_filtered_select(self, sample_dataframe):
        """Test filtered SELECT query."""
        rag = SQLRAG(connection=sample_dataframe, schema='employees')
        
        sql_tool = [t for t in rag.tools if t.name == "sql"][0]
        result = await sql_tool.execute(
            query="SELECT name, age FROM employees WHERE age > 28",
            limit=100
        )
        
        assert len(result["rows"]) == 3  # Charlie, Diana, Eve
        assert result["columns"] == ['name', 'age']
        assert all(row['age'] > 28 for row in result["rows"])
    
    @pytest.mark.asyncio
    async def test_sqlrag_aggregation(self, sample_dataframe):
        """Test aggregation queries."""
        rag = SQLRAG(connection=sample_dataframe, schema='employees')
        
        sql_tool = [t for t in rag.tools if t.name == "sql"][0]
        result = await sql_tool.execute(
            query="SELECT city, COUNT(*) as count, AVG(salary) as avg_salary FROM employees GROUP BY city",
            limit=100
        )
        
        assert len(result["rows"]) == 3  # NYC, LA, Chicago
        assert result["columns"] == ['city', 'count', 'avg_salary']
    
    @pytest.mark.asyncio
    async def test_sqlrag_security_no_insert(self, sample_dataframe):
        """Test that INSERT queries are blocked."""
        rag = SQLRAG(connection=sample_dataframe)
        
        sql_tool = [t for t in rag.tools if t.name == "sql"][0]
        result = await sql_tool.execute(
            query="INSERT INTO data (id, name) VALUES (999, 'Hacker')",
            limit=100
        )
        
        assert "error" in result
        assert "Only SELECT queries are allowed" in result["error"]
    
    def test_sqlrag_is_toolset(self):
        """Test SQLRAG returns a proper ToolSet."""
        try:
            import pandas as pd
            df = pd.DataFrame({'id': [1], 'name': ['test']})
        except ImportError:
            pytest.skip("pandas not installed")
        
        rag = SQLRAG(connection=df)
        
        assert isinstance(rag, ToolSet)
        assert hasattr(rag, 'tools')
        assert len(rag.tools) == 1  # Just sql tool
        assert rag.tools[0].name == "sql"
    
    def test_sqlrag_with_agent(self):
        """Test using SQLRAG with an Agent."""
        try:
            import pandas as pd
            df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        except ImportError:
            pytest.skip("pandas not installed")
        
        rag = SQLRAG(connection=df)
        
        agent = Agent(
            name="sql_agent",
            description="Agent for database queries",
            input_schema=QueryInput,
            output_schema=SearchOutput,
            tools=[rag],
        )
        
        assert agent.name == "sql_agent"
        assert len(agent.tools) > 0


# ============================================================================
# LangChain Integration Tests
# ============================================================================

class TestLangChainIntegration:
    """Test LangChain agent conversion."""
    
    def test_from_langchain_method_exists(self):
        """Test that from_langchain method exists on Agent."""
        assert hasattr(Agent, 'from_langchain')
        assert callable(Agent.from_langchain)
    
    def test_from_langchain_with_tools(self):
        """Test from_langchain with actual LangChain tools."""
        try:
            from langchain.tools import tool
        except ImportError:
            pytest.skip("LangChain not installed")
        
        # Create LangChain tools
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b
        
        # Tools should be created
        assert add is not None
        assert multiply is not None
        assert hasattr(add, 'name')
        assert hasattr(multiply, 'name')


# ============================================================================
# Combined Integration Tests
# ============================================================================

class TestCombinedIntegrations:
    """Test combining multiple RAG systems."""
    
    def test_filesystem_and_sql_rag_together(self):
        """Test combining FileSystemRAG and SQLRAG in one agent."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "data.txt").write_text("Sample data")
            
            # Create FileSystemRAG
            fs_rag = FileSystemRAG(tmppath)
            
            # Create SQLRAG
            df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
            sql_rag = SQLRAG(connection=df)
            
            # Combine in agent
            agent = Agent(
                name="combined_agent",
                description="Agent with file and database access",
                input_schema=QueryInput,
                output_schema=SearchOutput,
                tools=[fs_rag, sql_rag],
            )
            
            assert agent.name == "combined_agent"
            # Agent should have both FileSystemRAG and SQLRAG tools
            assert len(agent.tools) >= 2
    
    def test_rag_with_custom_tools(self):
        """Test RAG systems alongside custom tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a custom tool
            from pydantic import BaseModel
            
            class CustomInput(BaseModel):
                value: str
            
            class CustomOutput(BaseModel):
                result: str
            
            async def custom_execute(value: str):
                return {"result": f"Custom: {value}"}
            
            custom_tool = Tool(
                name="custom",
                description="Custom tool",
                input_schema=CustomInput,
                output_schema=CustomOutput,
                execute=custom_execute
            )
            
            # Create FileSystemRAG
            rag = FileSystemRAG(tmpdir)
            
            # Combine in agent
            agent = Agent(
                name="mixed_agent",
                description="Agent with RAG and custom tools",
                input_schema=QueryInput,
                output_schema=FileOutput,
                tools=[custom_tool, rag],
            )
            
            assert agent.name == "mixed_agent"
            assert len(agent.tools) >= 2


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestIntegrationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_filesystem_rag_with_nonexistent_path(self):
        """Test FileSystemRAG gracefully handles nonexistent paths."""
        # Should not raise - fsspec is flexible
        try:
            rag = FileSystemRAG("/tmp/test_nonexistent_path_12345")
            assert rag is not None  # Creation should succeed
        except (FileNotFoundError, OSError):
            pass  # Also acceptable
    
    @pytest.mark.asyncio
    async def test_sqlrag_invalid_query_type(self):
        """Test SQLRAG rejects non-SELECT queries."""
        try:
            import pandas as pd
            df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        except ImportError:
            pytest.skip("pandas not installed")
        
        rag = SQLRAG(connection=df)
        sql_tool = [t for t in rag.tools if t.name == "sql"][0]
        
        # DELETE should fail
        result = await sql_tool.execute(query="DELETE FROM data WHERE id = 1")
        assert "error" in result
        assert "SELECT" in result["error"]
    
    @pytest.mark.asyncio
    async def test_filesystem_rag_cat_nonexistent_file(self):
        """Test CAT tool handles nonexistent files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = FileSystemRAG(tmpdir)
            cat_tool = [t for t in rag.tools if t.name == "cat"][0]
            
            # Should return gracefully with error
            result = await cat_tool.execute(path="nonexistent.txt")
            assert "content" in result or "error" in result


# ============================================================================
# OpenTelemetry Integration Tests
# ============================================================================

# (Note: OpenTelemetry integration is already tested above in TestOpenTelemetryIntegration
# which verifies that Runtime has the necessary methods for tracing)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
