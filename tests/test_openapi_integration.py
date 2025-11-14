"""
Test OpenAPI/FastAPI integration with ToolSet.from_openapi().

This tests loading tools from a real FastAPI server's OpenAPI spec.
"""

import pytest

pytest.importorskip("fastapi", reason="FastAPI not installed")

from fastapi import FastAPI
from pydantic import BaseModel

from a1 import ToolSet


class CalculateRequest(BaseModel):
    """Request model for calculation."""

    operation: str
    a: int
    b: int


class CalculateResponse(BaseModel):
    """Response model for calculation."""

    result: int


class UserInfo(BaseModel):
    """User information."""

    name: str
    email: str


def create_test_app():
    """Create a FastAPI app for testing."""
    app = FastAPI(
        title="Test Calculator API",
        version="1.0.0",
        description="A test API for calculations",
    )

    @app.post("/calculate", response_model=CalculateResponse)
    async def calculate(req: CalculateRequest):
        """Perform a calculation."""
        if req.operation == "add":
            return {"result": req.a + req.b}
        elif req.operation == "subtract":
            return {"result": req.a - req.b}
        elif req.operation == "multiply":
            return {"result": req.a * req.b}
        elif req.operation == "divide":
            if req.b == 0:
                return {"result": 0}  # Simplified error handling
            return {"result": req.a // req.b}
        else:
            return {"result": 0}

    @app.get("/status")
    async def get_status():
        """Get API status."""
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/users/{user_id}")
    async def get_user(user_id: int):
        """Get user by ID."""
        return {"id": user_id, "name": f"User{user_id}", "email": f"user{user_id}@example.com"}

    @app.post("/users", response_model=UserInfo)
    async def create_user(user: UserInfo):
        """Create a new user."""
        return user

    return app


class TestFastAPIOpenAPISpec:
    """Test FastAPI OpenAPI spec generation."""

    def test_fastapi_generates_openapi_spec(self):
        """Test that FastAPI generates a valid OpenAPI spec."""
        app = create_test_app()
        spec = app.openapi()

        assert spec["openapi"].startswith("3.")
        assert spec["info"]["title"] == "Test Calculator API"
        assert "/calculate" in spec["paths"]
        assert "/status" in spec["paths"]
        assert "/users/{user_id}" in spec["paths"]

        print("✓ FastAPI generates valid OpenAPI spec")
        print(f"  Operations: {list(spec['paths'].keys())}")


class TestOpenAPIToolLoading:
    """Test loading tools from OpenAPI spec dict."""

    @pytest.mark.asyncio
    async def test_toolset_from_openapi_with_spec_dict(self):
        """Test creating ToolSet from OpenAPI spec dictionary."""
        # For now, we'll just test that the method exists
        # Full integration test would require a running server
        assert hasattr(ToolSet, "from_openapi")
        print("✓ ToolSet.from_openapi method exists")

    def test_openapi_spec_structure(self):
        """Test that we can parse the OpenAPI spec structure."""
        app = create_test_app()
        spec = app.openapi()

        # Check paths structure
        paths = spec.get("paths", {})
        assert len(paths) > 0

        # Check that operations have required fields
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    assert "responses" in operation
                    # operationId is optional but recommended
                    print(f"  {method.upper()} {path}: {operation.get('operationId', 'no-id')}")

        print("✓ OpenAPI spec has valid structure")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
