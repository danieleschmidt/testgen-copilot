# ADR-002: Security Rule Engine Design

## Status
Accepted

## Context
The security scanning component needs to detect various types of vulnerabilities across multiple programming languages. We need a flexible, extensible rule engine that can be easily maintained and updated.

## Decision
We will implement a rule-based security engine with the following design:

1. **Rule Definition Format**: YAML-based rule definitions for easy maintenance
2. **Pattern Matching**: AST-based pattern matching with regex fallback
3. **Severity Classification**: CVSS-inspired severity scoring system
4. **Extensible Architecture**: Plugin system for custom security rules
5. **Configuration Management**: Hierarchical configuration with rule enablement controls

## Rationale

### YAML Rule Definitions
- **Readability**: Non-technical stakeholders can review and contribute rules
- **Version Control**: Easy to track changes and review rule modifications
- **Collaboration**: Security experts can contribute without Python knowledge
- **Maintenance**: Simple format reduces maintenance overhead

### AST-Based Pattern Matching
- **Accuracy**: Reduces false positives compared to regex-only approaches
- **Context Awareness**: Can analyze code structure and data flow
- **Language Support**: Consistent approach across different programming languages
- **Performance**: More efficient than string-based pattern matching

### Severity Scoring
- **Standardization**: Industry-standard approach to vulnerability classification
- **Prioritization**: Helps developers focus on critical issues first
- **Integration**: Compatible with existing security tools and workflows
- **Customization**: Organizations can adjust scoring based on their risk profile

## Rule Engine Architecture

```python
class SecurityRule:
    def __init__(self, rule_definition):
        self.id = rule_definition['id']
        self.name = rule_definition['name']
        self.severity = rule_definition['severity']
        self.patterns = rule_definition['patterns']
        self.mitigation = rule_definition['mitigation']
    
    def evaluate(self, ast_node, context):
        # Pattern matching logic
        pass

class SecurityEngine:
    def __init__(self):
        self.rules = self.load_rules()
    
    def scan(self, source_code, language):
        ast_tree = parse_ast(source_code, language)
        findings = []
        for rule in self.rules:
            findings.extend(rule.evaluate(ast_tree))
        return findings
```

## Rule Definition Schema

```yaml
# Example security rule
id: "python-sql-injection-001"
name: "SQL Injection via String Concatenation"
severity: "HIGH"
language: "python"
categories: ["injection", "database"]
cwe_id: "CWE-89"
owasp_category: "A03:2021 – Injection"

patterns:
  - type: "ast_pattern"
    pattern: |
      cursor.execute(
        BinOp(
          left=Str(),
          op=Add(),
          right=Name()
        )
      )
  - type: "regex_pattern"
    pattern: "cursor\\.execute\\(.*\\+.*\\)"

description: |
  SQL queries constructed using string concatenation are vulnerable
  to SQL injection attacks when user input is not properly sanitized.

mitigation: |
  Use parameterized queries or prepared statements instead:
  cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

examples:
  vulnerable: |
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
  
  secure: |
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

references:
  - "https://owasp.org/www-community/attacks/SQL_Injection"
  - "https://cwe.mitre.org/data/definitions/89.html"
```

## Configuration Management

```json
{
  "security_rules": {
    "enabled": true,
    "rule_sets": ["owasp-top-10", "cwe-top-25"],
    "custom_rules_path": "./security/custom-rules",
    "rule_overrides": {
      "python-sql-injection-001": {
        "enabled": true,
        "severity": "CRITICAL"
      }
    },
    "exclude_patterns": [
      "tests/**/*",
      "vendor/**/*"
    ]
  }
}
```

## Consequences

### Positive
- Easy to add new security rules without code changes
- Community can contribute security rules
- Consistent vulnerability detection across languages
- Integration with existing security tools and standards
- Clear audit trail for security scanning decisions

### Negative
- Additional complexity in rule parsing and evaluation
- Potential performance impact for complex AST pattern matching
- Requires domain expertise to write effective rules

### Mitigation Strategies
- Provide rule writing documentation and examples
- Implement rule validation and testing framework
- Cache parsed rules and AST trees for performance
- Provide pre-built rule sets for common vulnerability types

## Implementation Plan
1. Define core rule schema and validation
2. Implement AST pattern matching engine
3. Create initial rule set for OWASP Top 10
4. Add configuration management layer
5. Implement rule testing framework
6. Create documentation and contribution guidelines

## Related ADRs
- ADR-001: Python CLI Architecture Decision
- ADR-003: Test Generation Strategy
- ADR-005: Multi-language Support Strategy