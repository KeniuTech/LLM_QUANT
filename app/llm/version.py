"""Template version management and validation."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .templates import PromptTemplate

LOGGER = logging.getLogger(__name__)
LOG_EXTRA = {"stage": "template_version"}


@dataclass
class TemplateVersion:
    """A versioned template configuration."""

    id: str
    version: str
    created_at: str
    template: PromptTemplate
    metadata: Dict[str, Any]
    is_active: bool = False

    @classmethod
    def create(cls, template: PromptTemplate, version: str, 
               metadata: Optional[Dict[str, Any]] = None) -> TemplateVersion:
        """Create a new template version."""
        return cls(
            id=template.id,
            version=version,
            created_at=datetime.now().isoformat(),
            template=template,
            metadata=metadata or {},
            is_active=False
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "version": self.version,
            "created_at": self.created_at,
            "template": asdict(self.template),
            "metadata": self.metadata,
            "is_active": self.is_active
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TemplateVersion:
        """Create from dictionary format."""
        template_data = data["template"]
        template = PromptTemplate(
            id=template_data["id"],
            name=template_data["name"],
            description=template_data["description"],
            template=template_data["template"],
            variables=template_data["variables"],
            max_length=template_data.get("max_length", 4000),
            required_context=template_data.get("required_context"),
            validation_rules=template_data.get("validation_rules")
        )
        return cls(
            id=data["id"],
            version=data["version"],
            created_at=data["created_at"],
            template=template,
            metadata=data["metadata"],
            is_active=data.get("is_active", False)
        )


class TemplateVersionManager:
    """Manages template versioning and deployment."""

    def __init__(self):
        """Initialize version manager."""
        self._versions: Dict[str, Dict[str, TemplateVersion]] = {}
        self._active_versions: Dict[str, str] = {}

    def add_version(self, template: PromptTemplate, version: str,
                   metadata: Optional[Dict[str, Any]] = None,
                   activate: bool = False) -> TemplateVersion:
        """Add a new template version."""
        if template.id not in self._versions:
            self._versions[template.id] = {}

        versions = self._versions[template.id]
        if version in versions:
            raise ValueError(f"Version {version} already exists for template {template.id}")

        template_version = TemplateVersion.create(
            template=template,
            version=version,
            metadata=metadata
        )

        versions[version] = template_version
        if activate:
            self.activate_version(template.id, version)

        return template_version

    def get_version(self, template_id: str, version: str) -> Optional[TemplateVersion]:
        """Get a specific template version."""
        return self._versions.get(template_id, {}).get(version)

    def list_versions(self, template_id: str) -> List[TemplateVersion]:
        """List all versions of a template."""
        return list(self._versions.get(template_id, {}).values())

    def get_active_version(self, template_id: str) -> Optional[TemplateVersion]:
        """Get the active version of a template."""
        active_version = self._active_versions.get(template_id)
        if active_version:
            return self.get_version(template_id, active_version)
        return None

    def activate_version(self, template_id: str, version: str) -> None:
        """Activate a specific template version."""
        if template_id not in self._versions:
            raise ValueError(f"Template {template_id} not found")
        
        versions = self._versions[template_id]
        if version not in versions:
            raise ValueError(f"Version {version} not found for template {template_id}")

        # Deactivate current active version
        current_active = self._active_versions.get(template_id)
        if current_active and current_active in versions:
            versions[current_active].is_active = False

        # Activate new version
        versions[version].is_active = True
        self._active_versions[template_id] = version

    def export_versions(self, template_id: str) -> str:
        """Export all versions of a template to JSON."""
        if template_id not in self._versions:
            raise ValueError(f"Template {template_id} not found")

        versions = self._versions[template_id]
        data = {
            "template_id": template_id,
            "active_version": self._active_versions.get(template_id),
            "versions": {
                version: ver.to_dict()
                for version, ver in versions.items()
            }
        }
        return json.dumps(data, indent=2)

    def import_versions(self, json_str: str) -> None:
        """Import template versions from JSON."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

        template_id = data.get("template_id")
        if not template_id:
            raise ValueError("Missing template_id in JSON")

        versions = data.get("versions", {})
        if not versions:
            raise ValueError("No versions found in JSON")

        # Clear existing versions for this template
        self._versions[template_id] = {}

        # Import versions
        for version, ver_data in versions.items():
            template_version = TemplateVersion.from_dict(ver_data)
            self._versions[template_id][version] = template_version

        # Set active version if specified
        active_version = data.get("active_version")
        if active_version:
            self.activate_version(template_id, active_version)
