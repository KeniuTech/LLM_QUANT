"""Test cases for LLM template management."""
import pytest

from app.llm.templates import PromptTemplate, TemplateRegistry


def test_prompt_template_validation():
    """Test template validation logic."""
    # Valid template
    template = PromptTemplate(
        id="test",
        name="Test Template",
        description="A test template",
        template="Hello {name}!",
        variables=["name"]
    )
    assert not template.validate()

    # Missing variable
    template = PromptTemplate(
        id="test",
        name="Test Template",
        description="A test template",
        template="Hello {name}!",
        variables=["name", "missing"]
    )
    errors = template.validate()
    assert len(errors) == 1
    assert "missing" in errors[0]

    # Empty required context
    template = PromptTemplate(
        id="test",
        name="Test Template",
        description="A test template",
        template="Hello {name}!",
        variables=["name"],
        required_context=["", "name"]
    )
    errors = template.validate()
    assert len(errors) == 1
    assert "Empty required context" in errors[0]

    # Empty validation rule
    template = PromptTemplate(
        id="test",
        name="Test Template",
        description="A test template",
        template="Hello {name}!",
        variables=["name"],
        validation_rules=["len(name) > 0", ""]
    )
    errors = template.validate()
    assert len(errors) == 1
    assert "Empty validation rule" in errors[0]


def test_prompt_template_format():
    """Test template formatting."""
    template = PromptTemplate(
        id="test",
        name="Test Template",
        description="A test template",
        template="Hello {name}!",
        variables=["name"],
        required_context=["name"],
        max_length=10
    )

    # Valid context
    result = template.format({"name": "World"})
    assert result == "Hello Wor..."

    # Missing required context
    with pytest.raises(ValueError) as exc:
        template.format({})
    assert "Missing required context" in str(exc.value)

    # Missing variable
    with pytest.raises(ValueError) as exc:
        template.format({"wrong": "value"})
    assert "Missing template variable" in str(exc.value)


def test_template_registry():
    """Test template registry operations."""
    TemplateRegistry.clear()

    # Register valid template
    template = PromptTemplate(
        id="test",
        name="Test Template",
        description="A test template",
        template="Hello {name}!",
        variables=["name"]
    )
    TemplateRegistry.register(template)
    assert TemplateRegistry.get("test") == template

    # Register invalid template
    invalid = PromptTemplate(
        id="invalid",
        name="Invalid Template",
        description="An invalid template",
        template="Hello {name}!",
        variables=["wrong"]
    )
    with pytest.raises(ValueError) as exc:
        TemplateRegistry.register(invalid)
    assert "Invalid template" in str(exc.value)

    # List templates
    templates = TemplateRegistry.list()
    assert len(templates) == 1
    assert templates[0].id == "test"

    # Load from JSON
    json_str = '''
    {
        "json_test": {
            "name": "JSON Test",
            "description": "Test template from JSON",
            "template": "Hello {name}!",
            "variables": ["name"]
        }
    }
    '''
    TemplateRegistry.load_from_json(json_str)
    assert TemplateRegistry.get("json_test") is not None

    # Invalid JSON
    with pytest.raises(ValueError) as exc:
        TemplateRegistry.load_from_json("invalid json")
    assert "Invalid JSON" in str(exc.value)

    # Non-object JSON
    with pytest.raises(ValueError) as exc:
        TemplateRegistry.load_from_json("[1, 2, 3]")
    assert "JSON root must be an object" in str(exc.value)


def test_default_templates():
    """Test default template registration."""
    TemplateRegistry.clear()
    from app.llm.templates import DEFAULT_TEMPLATES

    # Verify default templates are loaded
    assert len(TemplateRegistry.list()) > 0

    # Check specific templates
    dept_base = TemplateRegistry.get("department_base")
    assert dept_base is not None
    assert "部门基础模板" in dept_base.name

    momentum = TemplateRegistry.get("momentum_dept")
    assert momentum is not None
    assert "动量研究部门" in momentum.name

    # Validate template content
    assert all("{" + var + "}" in dept_base.template for var in dept_base.variables)
    assert all("{" + var + "}" in momentum.template for var in momentum.variables)

    # Test template usage
    context = {
        "title": "测试部门",
        "ts_code": "000001.SZ",
        "trade_date": "20251005",
        "description": "测试描述",
        "instruction": "测试指令",
        "data_scope": "daily,daily_basic",
        "features": "特征1,特征2",
        "market_snapshot": "市场数据1,市场数据2",
        "supplements": "补充数据"
    }
    result = dept_base.format(context)
    assert "测试部门" in result
    assert "000001.SZ" in result
    assert "20251005" in result
