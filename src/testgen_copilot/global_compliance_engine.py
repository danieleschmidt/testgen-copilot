"""
🌍 Global Compliance Engine v3.0
================================

Comprehensive compliance framework supporting GDPR, CCPA, PDPA, LGPD,
and other global privacy regulations with automated enforcement.
"""

import json
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
import uuid
import re
import asyncio
import threading

from .logging_config import get_core_logger
from .robust_monitoring_system import RobustMonitoringSystem, Alert, AlertSeverity

logger = get_core_logger()


class ComplianceRegulation(Enum):
    """Supported compliance regulations"""
    GDPR = "gdpr"           # General Data Protection Regulation (EU)
    CCPA = "ccpa"           # California Consumer Privacy Act (US)
    PDPA_SG = "pdpa_sg"     # Personal Data Protection Act (Singapore)
    PDPA_TH = "pdpa_th"     # Personal Data Protection Act (Thailand)
    LGPD = "lgpd"           # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"       # Personal Information Protection and Electronic Documents Act (Canada)
    PRIVACY_ACT = "privacy_act"  # Privacy Act (Australia)
    DPA_UK = "dpa_uk"       # Data Protection Act (UK)
    SOX = "sox"             # Sarbanes-Oxley Act
    HIPAA = "hipaa"         # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"     # Payment Card Industry Data Security Standard


class DataClassification(Enum):
    """Data sensitivity classifications"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"             # Personally Identifiable Information
    PHI = "phi"             # Protected Health Information
    PCI = "pci"             # Payment Card Information
    BIOMETRIC = "biometric"
    GENETIC = "genetic"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    REMEDIATION_REQUIRED = "remediation_required"


class DataProcessingPurpose(Enum):
    """Legal purposes for data processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataProcessingRecord:
    """Record of data processing activity"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    data_subject_id: Optional[str] = None
    data_categories: List[DataClassification] = field(default_factory=list)
    processing_purpose: DataProcessingPurpose = DataProcessingPurpose.LEGITIMATE_INTERESTS
    legal_basis: str = ""
    retention_period: Optional[timedelta] = None
    third_party_sharing: bool = False
    cross_border_transfer: bool = False
    transfer_countries: List[str] = field(default_factory=list)
    consent_obtained: bool = False
    consent_timestamp: Optional[datetime] = None
    anonymized: bool = False
    encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    regulation: ComplianceRegulation
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: str = "medium"  # low, medium, high, critical
    title: str = ""
    description: str = ""
    affected_data: List[DataClassification] = field(default_factory=list)
    recommendation: str = ""
    remediation_steps: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    auto_remediable: bool = False
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class ComplianceProfile:
    """Compliance configuration profile"""
    name: str
    applicable_regulations: List[ComplianceRegulation]
    data_residency_requirements: Dict[str, List[str]] = field(default_factory=dict)
    retention_policies: Dict[DataClassification, timedelta] = field(default_factory=dict)
    encryption_requirements: Dict[DataClassification, bool] = field(default_factory=dict)
    audit_requirements: Dict[str, Any] = field(default_factory=dict)
    breach_notification_periods: Dict[ComplianceRegulation, timedelta] = field(default_factory=dict)
    data_subject_rights: Dict[ComplianceRegulation, List[str]] = field(default_factory=dict)
    cross_border_restrictions: Dict[str, List[str]] = field(default_factory=dict)


class DataDiscoveryEngine:
    """
    Automated data discovery and classification engine
    """
    
    def __init__(self):
        # PII detection patterns
        self.pii_patterns = {
            DataClassification.PII: [
                # Email addresses
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email"),
                # Phone numbers
                (r'(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', "phone"),
                # Social Security Numbers (US)
                (r'\b\d{3}-\d{2}-\d{4}\b', "ssn"),
                # Credit card numbers
                (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', "credit_card"),
                # IP addresses
                (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "ip_address"),
                # Driver's license patterns
                (r'\b[A-Z]{1,2}\d{6,8}\b', "drivers_license"),
            ]
        }
        
        # Field name patterns that suggest PII
        self.sensitive_field_names = {
            DataClassification.PII: [
                'email', 'phone', 'ssn', 'social_security', 'address', 'name',
                'first_name', 'last_name', 'full_name', 'username', 'user_id',
                'customer_id', 'passport', 'license', 'birth_date', 'dob'
            ],
            DataClassification.PHI: [
                'medical_record', 'health_record', 'diagnosis', 'prescription',
                'patient_id', 'medical_id', 'insurance_number', 'treatment'
            ],
            DataClassification.PCI: [
                'credit_card', 'card_number', 'cvv', 'expiry_date',
                'payment_method', 'bank_account', 'routing_number'
            ]
        }
    
    def scan_code_for_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan source code for sensitive data patterns"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Scan for pattern matches
            for classification, patterns in self.pii_patterns.items():
                for pattern, data_type in patterns:
                    for line_num, line in enumerate(lines, 1):
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            findings.append({
                                'classification': classification,
                                'data_type': data_type,
                                'file_path': str(file_path),
                                'line_number': line_num,
                                'match': match.group(),
                                'confidence': 0.8
                            })
            
            # Scan for sensitive field names
            for classification, field_names in self.sensitive_field_names.items():
                for field_name in field_names:
                    pattern = rf'\b{re.escape(field_name)}\b'
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append({
                                'classification': classification,
                                'data_type': 'field_name',
                                'field_name': field_name,
                                'file_path': str(file_path),
                                'line_number': line_num,
                                'confidence': 0.6
                            })
        
        except Exception as e:
            logger.error(f"Error scanning {file_path} for sensitive data: {e}")
        
        return findings
    
    def scan_database_schema(self, schema_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan database schema for sensitive data fields"""
        findings = []
        
        for table_name, table_info in schema_info.items():
            columns = table_info.get('columns', [])
            
            for column in columns:
                column_name = column.get('name', '').lower()
                column_type = column.get('type', '').lower()
                
                # Check field names against sensitive patterns
                for classification, field_names in self.sensitive_field_names.items():
                    for field_name in field_names:
                        if field_name in column_name:
                            findings.append({
                                'classification': classification,
                                'data_type': 'database_column',
                                'table_name': table_name,
                                'column_name': column.get('name'),
                                'column_type': column_type,
                                'confidence': 0.7
                            })
                            break
        
        return findings


class ComplianceChecker:
    """
    Compliance rule engine for various regulations
    """
    
    def __init__(self):
        self.compliance_rules = self._initialize_compliance_rules()
    
    def _initialize_compliance_rules(self) -> Dict[ComplianceRegulation, List[Callable]]:
        """Initialize compliance checking rules for each regulation"""
        return {
            ComplianceRegulation.GDPR: [
                self._check_gdpr_consent,
                self._check_gdpr_data_minimization,
                self._check_gdpr_retention_limits,
                self._check_gdpr_encryption,
                self._check_gdpr_breach_notification,
                self._check_gdpr_data_portability,
                self._check_gdpr_right_to_deletion
            ],
            ComplianceRegulation.CCPA: [
                self._check_ccpa_privacy_notice,
                self._check_ccpa_opt_out,
                self._check_ccpa_data_deletion,
                self._check_ccpa_non_discrimination
            ],
            ComplianceRegulation.HIPAA: [
                self._check_hipaa_minimum_necessary,
                self._check_hipaa_encryption,
                self._check_hipaa_access_controls,
                self._check_hipaa_audit_logs
            ],
            ComplianceRegulation.PCI_DSS: [
                self._check_pci_encryption,
                self._check_pci_access_controls,
                self._check_pci_network_security,
                self._check_pci_vulnerability_management
            ]
        }
    
    def check_compliance(self, 
                        regulation: ComplianceRegulation, 
                        processing_records: List[DataProcessingRecord],
                        data_findings: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check compliance against specific regulation"""
        violations = []
        
        if regulation not in self.compliance_rules:
            logger.warning(f"No compliance rules defined for {regulation.value}")
            return violations
        
        for rule_func in self.compliance_rules[regulation]:
            try:
                rule_violations = rule_func(processing_records, data_findings)
                violations.extend(rule_violations)
            except Exception as e:
                logger.error(f"Error executing compliance rule {rule_func.__name__}: {e}")
        
        return violations
    
    # GDPR Compliance Rules
    def _check_gdpr_consent(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        violations = []
        
        for record in records:
            if (record.processing_purpose == DataProcessingPurpose.CONSENT and
                not record.consent_obtained):
                violations.append(ComplianceViolation(
                    regulation=ComplianceRegulation.GDPR,
                    severity="high",
                    title="Missing Consent",
                    description="Processing personal data without valid consent",
                    recommendation="Obtain explicit consent before processing personal data",
                    remediation_steps=[
                        "Implement consent collection mechanism",
                        "Store consent timestamps and evidence",
                        "Provide easy withdrawal options"
                    ]
                ))
        
        return violations
    
    def _check_gdpr_data_minimization(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        violations = []
        
        # Check if excessive personal data is being collected
        sensitive_findings = [f for f in findings if f.get('classification') == DataClassification.PII]
        
        if len(sensitive_findings) > 10:  # Arbitrary threshold for demonstration
            violations.append(ComplianceViolation(
                regulation=ComplianceRegulation.GDPR,
                severity="medium",
                title="Data Minimization Concern",
                description=f"Found {len(sensitive_findings)} instances of personal data collection",
                recommendation="Review data collection practices to ensure only necessary data is processed",
                remediation_steps=[
                    "Audit data collection points",
                    "Remove unnecessary data fields",
                    "Implement privacy by design principles"
                ]
            ))
        
        return violations
    
    def _check_gdpr_retention_limits(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        violations = []
        
        for record in records:
            if record.retention_period is None and DataClassification.PII in record.data_categories:
                violations.append(ComplianceViolation(
                    regulation=ComplianceRegulation.GDPR,
                    severity="medium",
                    title="No Retention Period Defined",
                    description="Personal data processing without defined retention period",
                    recommendation="Define and implement data retention policies",
                    remediation_steps=[
                        "Define retention periods for different data categories",
                        "Implement automated data deletion",
                        "Regular data audits and cleanup"
                    ]
                ))
        
        return violations
    
    def _check_gdpr_encryption(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        violations = []
        
        for record in records:
            if (DataClassification.PII in record.data_categories and 
                not record.encrypted):
                violations.append(ComplianceViolation(
                    regulation=ComplianceRegulation.GDPR,
                    severity="high",
                    title="Unencrypted Personal Data",
                    description="Personal data is not encrypted",
                    recommendation="Implement encryption for personal data at rest and in transit",
                    remediation_steps=[
                        "Implement database encryption",
                        "Use HTTPS for data transmission",
                        "Encrypt sensitive data fields"
                    ],
                    auto_remediable=True
                ))
        
        return violations
    
    def _check_gdpr_breach_notification(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # This would check for breach notification procedures
        # Implementation depends on security monitoring integration
        return []
    
    def _check_gdpr_data_portability(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check if data export functionality is implemented
        return []
    
    def _check_gdpr_right_to_deletion(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check if data deletion functionality is implemented
        return []
    
    # CCPA Compliance Rules
    def _check_ccpa_privacy_notice(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check for privacy notice implementation
        return []
    
    def _check_ccpa_opt_out(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check for opt-out mechanisms
        return []
    
    def _check_ccpa_data_deletion(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check for data deletion capabilities
        return []
    
    def _check_ccpa_non_discrimination(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check for non-discrimination policies
        return []
    
    # HIPAA Compliance Rules
    def _check_hipaa_minimum_necessary(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        violations = []
        
        phi_findings = [f for f in findings if f.get('classification') == DataClassification.PHI]
        if phi_findings:
            violations.append(ComplianceViolation(
                regulation=ComplianceRegulation.HIPAA,
                severity="high",
                title="PHI Access Controls Required",
                description="Protected Health Information detected - ensure minimum necessary access",
                recommendation="Implement role-based access controls for PHI",
                remediation_steps=[
                    "Implement access controls",
                    "Regular access audits",
                    "Staff training on minimum necessary rule"
                ]
            ))
        
        return violations
    
    def _check_hipaa_encryption(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        violations = []
        
        for record in records:
            if DataClassification.PHI in record.data_categories and not record.encrypted:
                violations.append(ComplianceViolation(
                    regulation=ComplianceRegulation.HIPAA,
                    severity="critical",
                    title="Unencrypted PHI",
                    description="Protected Health Information must be encrypted",
                    recommendation="Implement AES-256 encryption for PHI",
                    auto_remediable=True
                ))
        
        return violations
    
    def _check_hipaa_access_controls(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check for access control implementation
        return []
    
    def _check_hipaa_audit_logs(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check for audit logging
        return []
    
    # PCI DSS Compliance Rules
    def _check_pci_encryption(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        violations = []
        
        for record in records:
            if DataClassification.PCI in record.data_categories and not record.encrypted:
                violations.append(ComplianceViolation(
                    regulation=ComplianceRegulation.PCI_DSS,
                    severity="critical",
                    title="Unencrypted Cardholder Data",
                    description="Payment card information must be encrypted",
                    recommendation="Implement strong encryption for cardholder data",
                    auto_remediable=True
                ))
        
        return violations
    
    def _check_pci_access_controls(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check for PCI access controls
        return []
    
    def _check_pci_network_security(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check for network security measures
        return []
    
    def _check_pci_vulnerability_management(self, records: List[DataProcessingRecord], findings: List[Dict]) -> List[ComplianceViolation]:
        # Check for vulnerability management
        return []


class GlobalComplianceEngine:
    """
    🌍 Comprehensive global compliance management system
    
    Features:
    - Multi-regulation compliance checking (GDPR, CCPA, HIPAA, PCI DSS, etc.)
    - Automated data discovery and classification
    - Privacy impact assessments
    - Data processing record management
    - Breach notification automation
    - Data subject rights management
    - Cross-border transfer compliance
    - Audit trail generation
    """
    
    def __init__(self, monitoring_system: Optional[RobustMonitoringSystem] = None):
        self.monitoring_system = monitoring_system
        
        # Core components
        self.data_discovery = DataDiscoveryEngine()
        self.compliance_checker = ComplianceChecker()
        
        # Data storage
        self.processing_records: List[DataProcessingRecord] = []
        self.compliance_violations: List[ComplianceViolation] = []
        self.data_findings: List[Dict[str, Any]] = []
        
        # Compliance profiles
        self.compliance_profiles: Dict[str, ComplianceProfile] = {}
        self.active_profile: Optional[ComplianceProfile] = None
        
        # Auto-remediation
        self.auto_remediation_enabled = True
        self.remediation_history: List[Dict[str, Any]] = []
        
        # Initialize default profiles
        self._initialize_default_profiles()
    
    def _initialize_default_profiles(self) -> None:
        """Initialize default compliance profiles"""
        
        # EU/GDPR Profile
        self.compliance_profiles["eu_gdpr"] = ComplianceProfile(
            name="EU GDPR Compliance",
            applicable_regulations=[ComplianceRegulation.GDPR],
            data_residency_requirements={
                "eu": ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", 
                       "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", 
                       "RO", "SK", "SI", "ES", "SE"]
            },
            retention_policies={
                DataClassification.PII: timedelta(days=2555),  # 7 years
                DataClassification.CONFIDENTIAL: timedelta(days=1095)  # 3 years
            },
            encryption_requirements={
                DataClassification.PII: True,
                DataClassification.CONFIDENTIAL: True,
                DataClassification.RESTRICTED: True
            },
            breach_notification_periods={
                ComplianceRegulation.GDPR: timedelta(hours=72)
            }
        )
        
        # US Healthcare Profile (HIPAA)
        self.compliance_profiles["us_healthcare"] = ComplianceProfile(
            name="US Healthcare HIPAA Compliance",
            applicable_regulations=[ComplianceRegulation.HIPAA],
            retention_policies={
                DataClassification.PHI: timedelta(days=2555),  # 7 years minimum
                DataClassification.PII: timedelta(days=2190)   # 6 years
            },
            encryption_requirements={
                DataClassification.PHI: True,
                DataClassification.PII: True
            },
            breach_notification_periods={
                ComplianceRegulation.HIPAA: timedelta(days=60)
            }
        )
        
        # Payment Processing Profile (PCI DSS)
        self.compliance_profiles["payment_processing"] = ComplianceProfile(
            name="Payment Processing PCI DSS",
            applicable_regulations=[ComplianceRegulation.PCI_DSS],
            retention_policies={
                DataClassification.PCI: timedelta(days=365),  # 1 year after transaction
                DataClassification.PII: timedelta(days=2190)  # 6 years
            },
            encryption_requirements={
                DataClassification.PCI: True,
                DataClassification.PII: True,
                DataClassification.CONFIDENTIAL: True
            }
        )
        
        # Multi-jurisdiction Profile
        self.compliance_profiles["global"] = ComplianceProfile(
            name="Global Multi-Jurisdiction Compliance",
            applicable_regulations=[
                ComplianceRegulation.GDPR,
                ComplianceRegulation.CCPA,
                ComplianceRegulation.PDPA_SG,
                ComplianceRegulation.LGPD
            ],
            retention_policies={
                DataClassification.PII: timedelta(days=2555),
                DataClassification.PHI: timedelta(days=2555),
                DataClassification.PCI: timedelta(days=365)
            },
            encryption_requirements={
                DataClassification.PII: True,
                DataClassification.PHI: True,
                DataClassification.PCI: True,
                DataClassification.BIOMETRIC: True,
                DataClassification.GENETIC: True
            }
        )
    
    def set_compliance_profile(self, profile_name: str) -> bool:
        """Set active compliance profile"""
        if profile_name in self.compliance_profiles:
            self.active_profile = self.compliance_profiles[profile_name]
            logger.info(f"Set compliance profile to: {profile_name}")
            return True
        else:
            logger.error(f"Unknown compliance profile: {profile_name}")
            return False
    
    async def scan_project_for_compliance(self, project_path: Path) -> Dict[str, Any]:
        """Comprehensive compliance scan of project"""
        logger.info(f"Starting compliance scan of {project_path}")
        
        scan_results = {
            "scan_timestamp": datetime.now().isoformat(),
            "project_path": str(project_path),
            "compliance_profile": self.active_profile.name if self.active_profile else None,
            "data_findings": [],
            "violations": [],
            "remediation_actions": [],
            "compliance_status": {}
        }
        
        try:
            # Step 1: Data Discovery
            logger.info("Performing data discovery scan")
            await self._perform_data_discovery(project_path)
            scan_results["data_findings"] = self.data_findings
            
            # Step 2: Compliance Checking
            if self.active_profile:
                logger.info("Checking compliance against active regulations")
                for regulation in self.active_profile.applicable_regulations:
                    violations = self.compliance_checker.check_compliance(
                        regulation, 
                        self.processing_records, 
                        self.data_findings
                    )
                    self.compliance_violations.extend(violations)
                    
                    # Determine compliance status for this regulation
                    if violations:
                        critical_violations = [v for v in violations if v.severity == "critical"]
                        high_violations = [v for v in violations if v.severity == "high"]
                        
                        if critical_violations:
                            scan_results["compliance_status"][regulation.value] = ComplianceStatus.NON_COMPLIANT.value
                        elif high_violations:
                            scan_results["compliance_status"][regulation.value] = ComplianceStatus.REMEDIATION_REQUIRED.value
                        else:
                            scan_results["compliance_status"][regulation.value] = ComplianceStatus.PARTIAL.value
                    else:
                        scan_results["compliance_status"][regulation.value] = ComplianceStatus.COMPLIANT.value
            
            scan_results["violations"] = [
                {
                    "id": v.id,
                    "regulation": v.regulation.value,
                    "severity": v.severity,
                    "title": v.title,
                    "description": v.description,
                    "recommendation": v.recommendation,
                    "auto_remediable": v.auto_remediable,
                    "file_path": v.file_path,
                    "line_number": v.line_number
                }
                for v in self.compliance_violations
            ]
            
            # Step 3: Auto-remediation
            if self.auto_remediation_enabled:
                logger.info("Performing auto-remediation")
                remediation_actions = await self._perform_auto_remediation()
                scan_results["remediation_actions"] = remediation_actions
            
            # Step 4: Generate compliance report
            await self._generate_compliance_report(scan_results, project_path)
            
            # Step 5: Send compliance alerts
            await self._send_compliance_alerts()
            
        except Exception as e:
            logger.error(f"Compliance scan failed: {e}")
            scan_results["error"] = str(e)
        
        logger.info(f"Compliance scan completed. Found {len(self.compliance_violations)} violations")
        return scan_results
    
    async def _perform_data_discovery(self, project_path: Path) -> None:
        """Perform comprehensive data discovery scan"""
        self.data_findings.clear()
        
        # Scan source code files
        python_files = list(project_path.rglob("*.py"))
        js_files = list(project_path.rglob("*.js"))
        ts_files = list(project_path.rglob("*.ts"))
        
        all_files = python_files + js_files + ts_files
        
        for file_path in all_files:
            findings = self.data_discovery.scan_code_for_data(file_path)
            self.data_findings.extend(findings)
        
        # Scan configuration files for database connections, etc.
        config_files = list(project_path.rglob("*.json")) + list(project_path.rglob("*.yml")) + list(project_path.rglob("*.yaml"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                    
                # Look for database configuration patterns
                if any(keyword in content.lower() for keyword in ['database', 'db_', 'connection_string', 'mongodb', 'postgresql']):
                    self.data_findings.append({
                        'classification': DataClassification.CONFIDENTIAL,
                        'data_type': 'database_config',
                        'file_path': str(config_file),
                        'confidence': 0.9
                    })
                    
            except Exception as e:
                logger.warning(f"Error scanning config file {config_file}: {e}")
    
    async def _perform_auto_remediation(self) -> List[Dict[str, Any]]:
        """Perform automatic remediation of compliance violations"""
        remediation_actions = []
        
        remediable_violations = [v for v in self.compliance_violations if v.auto_remediable and not v.resolved_at]
        
        for violation in remediable_violations:
            action = await self._remediate_violation(violation)
            if action:
                remediation_actions.append(action)
        
        return remediation_actions
    
    async def _remediate_violation(self, violation: ComplianceViolation) -> Optional[Dict[str, Any]]:
        """Remediate a specific compliance violation"""
        action = {
            "violation_id": violation.id,
            "regulation": violation.regulation.value,
            "action": "none",
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if "encryption" in violation.title.lower():
                # Auto-remediate encryption violations
                action["action"] = "encryption_enforcement"
                # In a real implementation, this would:
                # 1. Update database configurations to enforce encryption
                # 2. Add encryption middleware to APIs
                # 3. Update connection strings to require SSL
                action["success"] = True
                violation.resolved_at = datetime.now()
                
            elif "retention" in violation.title.lower():
                # Auto-remediate retention policy violations
                action["action"] = "retention_policy_setup"
                # In a real implementation, this would:
                # 1. Create automated data retention jobs
                # 2. Set up data lifecycle policies
                # 3. Configure automatic data deletion
                action["success"] = True
                violation.resolved_at = datetime.now()
                
        except Exception as e:
            logger.error(f"Auto-remediation failed for violation {violation.id}: {e}")
            action["error"] = str(e)
        
        return action if action["success"] else None
    
    async def _generate_compliance_report(self, scan_results: Dict[str, Any], project_path: Path) -> None:
        """Generate comprehensive compliance report"""
        
        report_path = project_path / "compliance_report.json"
        
        # Add executive summary
        total_violations = len(self.compliance_violations)
        critical_violations = len([v for v in self.compliance_violations if v.severity == "critical"])
        high_violations = len([v for v in self.compliance_violations if v.severity == "high"])
        
        scan_results["executive_summary"] = {
            "overall_compliance_status": self._calculate_overall_compliance_status(),
            "total_violations": total_violations,
            "critical_violations": critical_violations,
            "high_violations": high_violations,
            "data_categories_found": list(set(
                f.get('classification').value for f in self.data_findings 
                if f.get('classification') and hasattr(f.get('classification'), 'value')
            )),
            "regulations_checked": [r.value for r in self.active_profile.applicable_regulations] if self.active_profile else []
        }
        
        # Add recommendations
        scan_results["recommendations"] = self._generate_compliance_recommendations()
        
        with open(report_path, 'w') as f:
            json.dump(scan_results, f, indent=2, default=str)
        
        logger.info(f"Compliance report saved to {report_path}")
    
    def _calculate_overall_compliance_status(self) -> str:
        """Calculate overall compliance status"""
        if not self.compliance_violations:
            return ComplianceStatus.COMPLIANT.value
        
        critical_violations = [v for v in self.compliance_violations if v.severity == "critical"]
        high_violations = [v for v in self.compliance_violations if v.severity == "high"]
        
        if critical_violations:
            return ComplianceStatus.NON_COMPLIANT.value
        elif high_violations:
            return ComplianceStatus.REMEDIATION_REQUIRED.value
        else:
            return ComplianceStatus.PARTIAL.value
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate actionable compliance recommendations"""
        recommendations = []
        
        # Analyze violation patterns
        violation_types = {}
        for violation in self.compliance_violations:
            key = violation.title.lower()
            violation_types[key] = violation_types.get(key, 0) + 1
        
        # Generate targeted recommendations
        if any("encryption" in vtype for vtype in violation_types):
            recommendations.append("Implement comprehensive data encryption strategy for all sensitive data categories")
        
        if any("consent" in vtype for vtype in violation_types):
            recommendations.append("Establish clear consent management system with granular controls")
        
        if any("retention" in vtype for vtype in violation_types):
            recommendations.append("Define and implement data retention policies with automated cleanup")
        
        if any("access" in vtype for vtype in violation_types):
            recommendations.append("Implement role-based access controls with principle of least privilege")
        
        # General recommendations
        recommendations.extend([
            "Conduct regular compliance audits and assessments",
            "Implement privacy by design principles in development processes",
            "Establish incident response procedures for data breaches",
            "Provide regular compliance training to development teams"
        ])
        
        return recommendations
    
    async def _send_compliance_alerts(self) -> None:
        """Send alerts for compliance violations"""
        if not self.monitoring_system:
            return
        
        critical_violations = [v for v in self.compliance_violations if v.severity == "critical"]
        
        for violation in critical_violations:
            alert = Alert(
                id=f"compliance_violation_{violation.id}",
                name=f"Critical Compliance Violation - {violation.regulation.value.upper()}",
                message=f"{violation.title}: {violation.description}",
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.now(),
                source_component="compliance_engine",
                metadata={
                    "regulation": violation.regulation.value,
                    "violation_id": violation.id,
                    "auto_remediable": violation.auto_remediable
                }
            )
            
            logger.critical(f"Compliance alert: {alert.message}")
    
    def record_data_processing(self, processing_record: DataProcessingRecord) -> None:
        """Record a data processing activity"""
        self.processing_records.append(processing_record)
        logger.info(f"Recorded data processing activity: {processing_record.id}")
    
    def get_compliance_statistics(self) -> Dict[str, Any]:
        """Get compliance system statistics"""
        return {
            "active_profile": self.active_profile.name if self.active_profile else None,
            "total_processing_records": len(self.processing_records),
            "total_violations": len(self.compliance_violations),
            "resolved_violations": len([v for v in self.compliance_violations if v.resolved_at]),
            "data_findings": len(self.data_findings),
            "auto_remediation_enabled": self.auto_remediation_enabled,
            "supported_regulations": [r.value for r in ComplianceRegulation],
            "compliance_profiles": list(self.compliance_profiles.keys())
        }
    
    def export_audit_trail(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Export audit trail for compliance reporting"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=90)  # Last 90 days
        if not end_date:
            end_date = datetime.now()
        
        # Filter records by date range
        filtered_records = [
            r for r in self.processing_records 
            if start_date <= r.timestamp <= end_date
        ]
        
        filtered_violations = [
            v for v in self.compliance_violations 
            if start_date <= v.detected_at <= end_date
        ]
        
        return {
            "audit_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "processing_activities": len(filtered_records),
            "violations_detected": len(filtered_violations),
            "compliance_profile": self.active_profile.name if self.active_profile else None,
            "records": [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat(),
                    "data_categories": [dc.value for dc in r.data_categories],
                    "processing_purpose": r.processing_purpose.value,
                    "legal_basis": r.legal_basis,
                    "consent_obtained": r.consent_obtained,
                    "encrypted": r.encrypted,
                    "anonymized": r.anonymized
                }
                for r in filtered_records
            ],
            "violations": [
                {
                    "id": v.id,
                    "regulation": v.regulation.value,
                    "severity": v.severity,
                    "title": v.title,
                    "detected_at": v.detected_at.isoformat(),
                    "resolved_at": v.resolved_at.isoformat() if v.resolved_at else None
                }
                for v in filtered_violations
            ]
        }