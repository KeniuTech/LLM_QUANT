"""Test cases for template version management."""
import pytest

from app.llm.templates import PromptTemplate
from app.llm.version import TemplateVersion, TemplateVersionManager


def test_template_version_creation():
    """Test creating and managing template versions."""
    template = PromptTemplate(
        id="test",
        name="Test Template",
        description="A test template",
        template="Test {var}",
        variables=["var"]
    )
    
    version = TemplateVersion.create(
        template=template,
        version="1.0.0",
        metadata={"author": "test"}
    )
    
    assert version.id == "test"
    assert version.version == "1.0.0"
    assert not version.is_active
    assert version.metadata["author"] == "test"


def test_version_manager():
    """Test version manager operations."""
    manager = TemplateVersionManager()
    
    template = PromptTemplate(
        id="test",
        name="Test Template",
        description="A test template",
        template="Test {var}",
        variables=["var"]
    )

    # Add version
    v1 = manager.add_version(template, "1.0.0")
    assert not v1.is_active

    # Add and activate version
    v2 = manager.add_version(template, "2.0.0", activate=True)
    assert v2.is_active
    assert not v1.is_active

    # Get version
    assert manager.get_version("test", "1.0.0") == v1
    assert manager.get_version("test", "2.0.0") == v2

    # List versions
    versions = manager.list_versions("test")
    assert len(versions) == 2

    # Get active version
    active = manager.get_active_version("test")
    assert active == v2

    # Export and import
    exported = manager.export_versions("test")
    
    new_manager = TemplateVersionManager()
    new_manager.import_versions(exported)
    
    imported = new_manager.get_version("test", "2.0.0")
    assert imported.version == "2.0.0"
    assert imported.is_active


def test_version_validation():
    """Test version validation checks."""
    manager = TemplateVersionManager()
    
    template = PromptTemplate(
        id="test",
        name="Test Template", 
        description="A test template",
        template="Test {var}",
        variables=["var"]
    )

    # Test duplicate version
    manager.add_version(template, "1.0.0")
    with pytest.raises(ValueError):
        manager.add_version(template, "1.0.0")

    # Test invalid version
    with pytest.raises(ValueError):
        manager.activate_version("test", "invalid")
