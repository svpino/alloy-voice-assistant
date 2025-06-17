"""Validation tests to ensure the testing infrastructure is properly set up."""

import sys
from pathlib import Path

import pytest


class TestSetupValidation:
    """Validate that the testing infrastructure is properly configured."""

    def test_python_version(self):
        """Ensure Python version is 3.8 or higher."""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"

    def test_project_structure(self):
        """Verify the project structure is correct."""
        project_root = Path(__file__).parent.parent
        
        # Check main project files exist
        assert (project_root / "assistant.py").exists()
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "README.md").exists()
        
        # Check test structure
        assert (project_root / "tests").is_dir()
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "conftest.py").exists()
        assert (project_root / "tests" / "unit").is_dir()
        assert (project_root / "tests" / "integration").is_dir()

    def test_pytest_markers(self):
        """Verify custom pytest markers are available."""
        # The markers are defined in pyproject.toml and will be available when running tests
        # This test just verifies they can be used (tested in the next three methods)
        assert True

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        assert True

    def test_fixtures_available(self, temp_dir, mock_env_vars, sample_config):
        """Verify that common fixtures are available."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_env_vars fixture
        mock_env_vars(TEST_VAR="test_value")
        
        # Test sample_config fixture
        assert isinstance(sample_config, dict)
        assert "openai_api_key" in sample_config

    def test_imports(self):
        """Verify all required packages can be imported."""
        try:
            import pytest
            import pytest_cov
            import pytest_mock
        except ImportError as e:
            pytest.fail(f"Failed to import testing dependency: {e}")

    def test_coverage_configuration(self):
        """Verify coverage is properly configured."""
        project_root = Path(__file__).parent.parent
        pyproject = project_root / "pyproject.toml"
        
        assert pyproject.exists()
        
        with open(pyproject, 'r') as f:
            content = f.read()
            assert "[tool.coverage.run]" in content
            assert "[tool.coverage.report]" in content
            assert "cov-fail-under=80" in content