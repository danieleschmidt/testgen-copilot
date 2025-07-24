"""Security rules configuration management for TestGen Copilot."""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

# YAML support is optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .logging_config import get_security_logger


@dataclass
class SecurityRule:
    """Definition of a security rule."""
    
    name: str
    pattern: str
    message: str
    severity: str = "medium"  # low, medium, high, critical
    enabled: bool = True
    requires_shell: bool = False  # Special handling for subprocess calls
    check_dynamic_args: bool = False  # Check for non-constant arguments


@dataclass 
class SecurityRuleSet:
    """Collection of security rules with metadata."""
    
    version: str
    description: str
    rules: List[SecurityRule]
    
    def get_enabled_rules(self) -> List[SecurityRule]:
        """Get only enabled rules."""
        return [rule for rule in self.rules if rule.enabled]
    
    def get_rules_by_severity(self, severity: str) -> List[SecurityRule]:
        """Get rules filtered by severity level."""
        return [rule for rule in self.rules if rule.severity == severity and rule.enabled]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SecurityRuleSet':
        """Create from dictionary."""
        rules = [SecurityRule(**rule_data) for rule_data in data.get('rules', [])]
        return cls(
            version=data.get('version', '1.0'),
            description=data.get('description', ''),
            rules=rules
        )


class SecurityRulesManager:
    """Manages loading and validation of security rules from external configuration."""
    
    DEFAULT_RULES_FILE = "security_rules.json"
    FALLBACK_RULES_FILE = "security_rules.yaml"
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.logger = get_security_logger()
        self.config_path = self._resolve_config_path(config_path)
        self._rule_set: Optional[SecurityRuleSet] = None
    
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolve the configuration file path."""
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            else:
                self.logger.warning("Specified config path does not exist", {
                    "config_path": str(config_path)
                })
                return None
        
        # Look for default config files in common locations
        search_paths = [
            Path.cwd() / self.DEFAULT_RULES_FILE,
            Path.cwd() / self.FALLBACK_RULES_FILE,
            Path.cwd() / "config" / self.DEFAULT_RULES_FILE,
            Path.cwd() / ".testgen" / self.DEFAULT_RULES_FILE,
            Path.home() / ".testgen" / self.DEFAULT_RULES_FILE,
        ]
        
        for path in search_paths:
            if path.exists():
                self.logger.debug("Found security rules config", {
                    "config_path": str(path)
                })
                return path
        
        # No config file found
        self.logger.debug("No security rules config file found, using defaults")
        return None
    
    def load_rules(self) -> SecurityRuleSet:
        """Load security rules from configuration file or defaults."""
        if self._rule_set is not None:
            return self._rule_set
        
        if self.config_path and self.config_path.exists():
            try:
                self._rule_set = self._load_from_file(self.config_path)
                self.logger.info("Loaded security rules from config", {
                    "config_path": str(self.config_path),
                    "rule_count": len(self._rule_set.rules),
                    "enabled_count": len(self._rule_set.get_enabled_rules())
                })
                return self._rule_set
            except Exception as e:
                self.logger.error("Failed to load security rules config", {
                    "config_path": str(self.config_path),
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                # Fall back to defaults
        
        # Use default rules
        self._rule_set = self._get_default_rules()
        self.logger.info("Using default security rules", {
            "rule_count": len(self._rule_set.rules)
        })
        return self._rule_set
    
    def _load_from_file(self, file_path: Path) -> SecurityRuleSet:
        """Load rules from JSON or YAML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError("PyYAML is required to load YAML configuration files")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a JSON/YAML object")
        
        if 'rules' not in data:
            raise ValueError("Config file must contain 'rules' key")
        
        return SecurityRuleSet.from_dict(data)
    
    def _get_default_rules(self) -> SecurityRuleSet:
        """Get default security rules (same as current hardcoded rules)."""
        default_rules = [
            SecurityRule(
                name="eval_usage",
                pattern="eval",
                message="Use of eval() can lead to code execution vulnerabilities",
                severity="critical"
            ),
            SecurityRule(
                name="exec_usage", 
                pattern="exec",
                message="Use of exec() can lead to code execution vulnerabilities",
                severity="critical"
            ),
            SecurityRule(
                name="os_system",
                pattern="os.system",
                message="os.system() can be unsafe; prefer subprocess without shell=True",
                severity="high",
                check_dynamic_args=True
            ),
            SecurityRule(
                name="subprocess_call",
                pattern="subprocess.call",
                message="subprocess.call with shell=True can be unsafe",
                severity="high",
                requires_shell=True,
                check_dynamic_args=True
            ),
            SecurityRule(
                name="subprocess_popen",
                pattern="subprocess.Popen",
                message="subprocess.Popen with shell=True can be unsafe",
                severity="high",
                requires_shell=True,
                check_dynamic_args=True
            ),
            SecurityRule(
                name="subprocess_run",
                pattern="subprocess.run",
                message="subprocess.run with shell=True can be unsafe",
                severity="high",
                requires_shell=True,
                check_dynamic_args=True
            ),
            SecurityRule(
                name="pickle_load",
                pattern="pickle.load",
                message="Deserializing with pickle can be unsafe with untrusted data",
                severity="medium"
            ),
            SecurityRule(
                name="pickle_loads",
                pattern="pickle.loads", 
                message="Deserializing with pickle can be unsafe with untrusted data",
                severity="medium"
            ),
            SecurityRule(
                name="yaml_load",
                pattern="yaml.load",
                message="yaml.load is unsafe; use yaml.safe_load instead",
                severity="high"
            ),
            SecurityRule(
                name="tempfile_mktemp",
                pattern="tempfile.mktemp",
                message="tempfile.mktemp is insecure; use NamedTemporaryFile",
                severity="medium"
            )
        ]
        
        return SecurityRuleSet(
            version="1.0",
            description="Default TestGen Copilot security rules",
            rules=default_rules
        )
    
    def save_default_config(self, output_path: Union[str, Path]) -> None:
        """Save default configuration to a file for user customization."""
        output_file = Path(output_path)
        
        rule_set = self._get_default_rules()
        data = rule_set.to_dict()
        
        if output_file.suffix.lower() in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise ImportError("PyYAML is required to save YAML configuration files")
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2, sort_keys=False)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Saved default security rules config", {
            "output_path": str(output_file),
            "format": "YAML" if output_file.suffix.lower() in ['.yaml', '.yml'] else "JSON"
        })
    
    def get_dangerous_calls_dict(self) -> Dict[str, str]:
        """Get rules in the format expected by the current SecurityScanner."""
        rule_set = self.load_rules()
        return {rule.pattern: rule.message for rule in rule_set.get_enabled_rules()}
    
    def get_rules_requiring_shell(self) -> List[str]:
        """Get patterns for rules that require shell=True checking."""
        rule_set = self.load_rules()
        return [rule.pattern for rule in rule_set.get_enabled_rules() if rule.requires_shell]
    
    def get_rules_checking_dynamic_args(self) -> List[str]:
        """Get patterns for rules that need dynamic argument checking."""
        rule_set = self.load_rules()
        return [rule.pattern for rule in rule_set.get_enabled_rules() if rule.check_dynamic_args]