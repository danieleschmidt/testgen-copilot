"""
Regulatory compliance and data governance framework for TestGen-Copilot.

Provides compliance checks, data governance, privacy protection, and regulatory
adherence for global deployment in regulated industries.
"""

import hashlib
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

from .logging_config import get_core_logger


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    GDPR = "GDPR"           # General Data Protection Regulation (EU)
    CCPA = "CCPA"           # California Consumer Privacy Act (US)
    HIPAA = "HIPAA"         # Health Insurance Portability and Accountability Act (US)
    SOX = "SOX"             # Sarbanes-Oxley Act (US)
    PCI_DSS = "PCI_DSS"     # Payment Card Industry Data Security Standard
    ISO27001 = "ISO27001"   # Information Security Management
    NIST = "NIST"           # National Institute of Standards and Technology
    PIPEDA = "PIPEDA"       # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "LGPD"           # Lei Geral de Proteção de Dados (Brazil)
    PDPA_SG = "PDPA_SG"     # Personal Data Protection Act (Singapore)
    PDPA_TH = "PDPA_TH"     # Personal Data Protection Act (Thailand)
    POPIA = "POPIA"         # Protection of Personal Information Act (South Africa)


class DataClassification(str, Enum):
    """Data classification levels."""
    PUBLIC = "PUBLIC"                   # Publicly available information
    INTERNAL = "INTERNAL"               # Internal company information
    CONFIDENTIAL = "CONFIDENTIAL"       # Confidential business information
    RESTRICTED = "RESTRICTED"           # Highly sensitive restricted information
    PERSONAL = "PERSONAL"               # Personally identifiable information
    SENSITIVE_PERSONAL = "SENSITIVE_PERSONAL"  # Sensitive personal information
    FINANCIAL = "FINANCIAL"             # Financial information
    HEALTH = "HEALTH"                   # Health information
    BIOMETRIC = "BIOMETRIC"             # Biometric information


class ProcessingPurpose(str, Enum):
    """Data processing purposes for compliance."""
    TESTING = "TESTING"                 # Test generation and analysis
    SECURITY_ANALYSIS = "SECURITY_ANALYSIS"  # Security vulnerability scanning
    PERFORMANCE_MONITORING = "PERFORMANCE_MONITORING"  # Performance metrics
    QUALITY_ASSURANCE = "QUALITY_ASSURANCE"  # Code quality assessment
    ANALYTICS = "ANALYTICS"             # Usage analytics
    TROUBLESHOOTING = "TROUBLESHOOTING"  # Error diagnosis and resolution
    RESEARCH = "RESEARCH"               # Research and development
    COMPLIANCE_MONITORING = "COMPLIANCE_MONITORING"  # Compliance verification


class RetentionPeriod(str, Enum):
    """Data retention periods."""
    IMMEDIATE = "IMMEDIATE"             # Delete immediately after processing
    SESSION = "SESSION"                 # Retain for session duration
    DAY_1 = "DAY_1"                    # Retain for 1 day
    DAYS_7 = "DAYS_7"                  # Retain for 7 days
    DAYS_30 = "DAYS_30"                # Retain for 30 days
    DAYS_90 = "DAYS_90"                # Retain for 90 days
    YEAR_1 = "YEAR_1"                  # Retain for 1 year
    YEARS_7 = "YEARS_7"                # Retain for 7 years (regulatory requirement)
    PERMANENT = "PERMANENT"             # Permanent retention


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for compliance auditing."""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_subject_id: Optional[str] = None
    data_classification: DataClassification = DataClassification.INTERNAL
    processing_purpose: ProcessingPurpose = ProcessingPurpose.TESTING
    legal_basis: str = "Legitimate interest"
    retention_period: RetentionPeriod = RetentionPeriod.DAYS_30
    data_controller: str = "TestGen Copilot"
    data_processor: str = "TestGen Copilot System"
    geographic_location: str = "US"
    encryption_used: bool = True
    pseudonymization_used: bool = False
    consent_obtained: bool = False
    consent_id: Optional[str] = None
    processing_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceRule:
    """A compliance rule definition."""
    id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    severity: str  # "critical", "high", "medium", "low"
    automated_check: Optional[Callable[[Any], bool]] = None
    remediation_guidance: str = ""
    applicable_data_types: Set[DataClassification] = field(default_factory=set)
    geographic_scope: Set[str] = field(default_factory=set)  # ISO country codes


class PrivacyControlsManager:
    """Manages privacy controls and data protection."""

    def __init__(self):
        self.logger = get_core_logger()
        self._data_processing_log: List[DataProcessingRecord] = []
        self._consent_records: Dict[str, Dict[str, Any]] = {}
        self._data_retention_policies: Dict[DataClassification, RetentionPeriod] = {}
        self._lock = threading.RLock()

        self._init_default_retention_policies()

    def _init_default_retention_policies(self) -> None:
        """Initialize default data retention policies."""
        self._data_retention_policies = {
            DataClassification.PUBLIC: RetentionPeriod.PERMANENT,
            DataClassification.INTERNAL: RetentionPeriod.YEAR_1,
            DataClassification.CONFIDENTIAL: RetentionPeriod.DAYS_90,
            DataClassification.RESTRICTED: RetentionPeriod.DAYS_30,
            DataClassification.PERSONAL: RetentionPeriod.DAYS_30,
            DataClassification.SENSITIVE_PERSONAL: RetentionPeriod.DAYS_7,
            DataClassification.FINANCIAL: RetentionPeriod.YEARS_7,
            DataClassification.HEALTH: RetentionPeriod.DAYS_30,
            DataClassification.BIOMETRIC: RetentionPeriod.IMMEDIATE,
        }

    def log_data_processing(self, record: DataProcessingRecord) -> None:
        """Log a data processing activity."""
        with self._lock:
            self._data_processing_log.append(record)

            # Apply retention policy
            retention_period = self._data_retention_policies.get(
                record.data_classification, RetentionPeriod.DAYS_30
            )
            record.retention_period = retention_period

            self.logger.info("Data processing logged", {
                "record_id": record.id,
                "data_classification": record.data_classification.value,
                "processing_purpose": record.processing_purpose.value,
                "retention_period": retention_period.value
            })

    def record_consent(self, data_subject_id: str, purposes: List[ProcessingPurpose],
                      consent_details: Dict[str, Any]) -> str:
        """Record user consent for data processing."""
        consent_id = str(uuid4())

        with self._lock:
            self._consent_records[consent_id] = {
                "id": consent_id,
                "data_subject_id": data_subject_id,
                "purposes": [p.value for p in purposes],
                "granted_at": datetime.now(timezone.utc),
                "valid": True,
                "details": consent_details,
                "withdrawn_at": None
            }

        self.logger.info("Consent recorded", {
            "consent_id": consent_id,
            "data_subject_id": data_subject_id,
            "purposes": [p.value for p in purposes]
        })

        return consent_id

    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw previously granted consent."""
        with self._lock:
            if consent_id in self._consent_records:
                self._consent_records[consent_id]["valid"] = False
                self._consent_records[consent_id]["withdrawn_at"] = datetime.now(timezone.utc)

                self.logger.info("Consent withdrawn", {"consent_id": consent_id})
                return True

        return False

    def has_valid_consent(self, data_subject_id: str, purpose: ProcessingPurpose) -> bool:
        """Check if valid consent exists for a processing purpose."""
        with self._lock:
            for consent in self._consent_records.values():
                if (consent["data_subject_id"] == data_subject_id and
                    consent["valid"] and
                    purpose.value in consent["purposes"]):
                    return True

        return False

    def pseudonymize_identifier(self, identifier: str, salt: str = "") -> str:
        """Pseudonymize a personal identifier using hashing."""
        # Use SHA-256 with salt for pseudonymization
        combined = f"{identifier}:{salt}:testgen_copilot"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]

    def classify_data_content(self, content: str) -> Set[DataClassification]:
        """Automatically classify data content based on patterns."""
        classifications = set()

        # Email patterns
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content):
            classifications.add(DataClassification.PERSONAL)

        # Credit card patterns
        if re.search(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', content):
            classifications.add(DataClassification.FINANCIAL)

        # Social Security Number patterns (US)
        if re.search(r'\b\d{3}-?\d{2}-?\d{4}\b', content):
            classifications.add(DataClassification.SENSITIVE_PERSONAL)

        # Phone number patterns
        if re.search(r'\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', content):
            classifications.add(DataClassification.PERSONAL)

        # IP address patterns
        if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', content):
            classifications.add(DataClassification.INTERNAL)

        # If no patterns matched, default to internal
        if not classifications:
            classifications.add(DataClassification.INTERNAL)

        return classifications

    def get_retention_policy(self, data_classification: DataClassification) -> RetentionPeriod:
        """Get retention policy for a data classification."""
        return self._data_retention_policies.get(data_classification, RetentionPeriod.DAYS_30)

    def cleanup_expired_data(self) -> int:
        """Clean up data that has exceeded its retention period."""
        current_time = datetime.now(timezone.utc)
        expired_count = 0

        with self._lock:
            # Clean up processing logs
            retention_cutoffs = {
                RetentionPeriod.IMMEDIATE: current_time,
                RetentionPeriod.SESSION: current_time - timedelta(hours=1),
                RetentionPeriod.DAY_1: current_time - timedelta(days=1),
                RetentionPeriod.DAYS_7: current_time - timedelta(days=7),
                RetentionPeriod.DAYS_30: current_time - timedelta(days=30),
                RetentionPeriod.DAYS_90: current_time - timedelta(days=90),
                RetentionPeriod.YEAR_1: current_time - timedelta(days=365),
                RetentionPeriod.YEARS_7: current_time - timedelta(days=2555),  # 7 years
            }

            # Filter out expired records
            initial_count = len(self._data_processing_log)
            self._data_processing_log = [
                record for record in self._data_processing_log
                if record.retention_period == RetentionPeriod.PERMANENT or
                   record.timestamp > retention_cutoffs.get(record.retention_period, current_time)
            ]
            expired_count = initial_count - len(self._data_processing_log)

        if expired_count > 0:
            self.logger.info("Expired data cleaned up", {"records_deleted": expired_count})

        return expired_count

    def export_data_for_subject(self, data_subject_id: str) -> Dict[str, Any]:
        """Export all data related to a data subject (GDPR Article 15)."""
        with self._lock:
            # Find processing records
            processing_records = [
                {
                    "id": record.id,
                    "timestamp": record.timestamp.isoformat(),
                    "purpose": record.processing_purpose.value,
                    "classification": record.data_classification.value,
                    "retention_period": record.retention_period.value,
                    "details": record.processing_details
                }
                for record in self._data_processing_log
                if record.data_subject_id == data_subject_id
            ]

            # Find consent records
            consent_records = [
                {
                    "id": consent["id"],
                    "granted_at": consent["granted_at"].isoformat(),
                    "purposes": consent["purposes"],
                    "valid": consent["valid"],
                    "withdrawn_at": consent["withdrawn_at"].isoformat() if consent["withdrawn_at"] else None
                }
                for consent in self._consent_records.values()
                if consent["data_subject_id"] == data_subject_id
            ]

            return {
                "data_subject_id": data_subject_id,
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_records": processing_records,
                "consent_records": consent_records
            }

    def delete_subject_data(self, data_subject_id: str) -> bool:
        """Delete all data for a data subject (Right to be Forgotten - GDPR Article 17)."""
        with self._lock:
            # Remove processing records
            initial_count = len(self._data_processing_log)
            self._data_processing_log = [
                record for record in self._data_processing_log
                if record.data_subject_id != data_subject_id
            ]

            # Remove consent records
            consents_to_remove = [
                consent_id for consent_id, consent in self._consent_records.items()
                if consent["data_subject_id"] == data_subject_id
            ]

            for consent_id in consents_to_remove:
                del self._consent_records[consent_id]

            deleted_count = initial_count - len(self._data_processing_log) + len(consents_to_remove)

            self.logger.info("Subject data deleted", {
                "data_subject_id": data_subject_id,
                "records_deleted": deleted_count
            })

            return deleted_count > 0


class ComplianceEngine:
    """Main compliance engine for regulatory adherence."""

    def __init__(self):
        self.logger = get_core_logger()
        self.privacy_controls = PrivacyControlsManager()
        self._compliance_rules: Dict[ComplianceFramework, List[ComplianceRule]] = {}
        self._active_frameworks: Set[ComplianceFramework] = set()
        self._geographic_jurisdiction: Set[str] = {"US"}  # Default to US
        self._lock = threading.RLock()

        # Initialize compliance rules
        self._init_compliance_rules()

    def _init_compliance_rules(self) -> None:
        """Initialize compliance rules for different frameworks."""
        # GDPR Rules
        gdpr_rules = [
            ComplianceRule(
                id="GDPR_ART_6_LEGAL_BASIS",
                framework=ComplianceFramework.GDPR,
                title="Legal Basis for Processing",
                description="Processing must have a valid legal basis under GDPR Article 6",
                requirement="Ensure legal basis exists for all personal data processing",
                severity="critical",
                applicable_data_types={DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL},
                geographic_scope={"EU", "UK"}
            ),
            ComplianceRule(
                id="GDPR_ART_25_DATA_PROTECTION_BY_DESIGN",
                framework=ComplianceFramework.GDPR,
                title="Data Protection by Design and Default",
                description="Implement appropriate technical measures for data protection",
                requirement="Technical measures must protect personal data by design and default",
                severity="high",
                applicable_data_types={DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL},
                geographic_scope={"EU", "UK"}
            ),
            ComplianceRule(
                id="GDPR_ART_32_SECURITY",
                framework=ComplianceFramework.GDPR,
                title="Security of Processing",
                description="Implement appropriate technical and organizational security measures",
                requirement="Ensure appropriate security measures including encryption and pseudonymization",
                severity="critical",
                applicable_data_types={DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL},
                geographic_scope={"EU", "UK"}
            )
        ]

        # CCPA Rules
        ccpa_rules = [
            ComplianceRule(
                id="CCPA_TRANSPARENCY",
                framework=ComplianceFramework.CCPA,
                title="Transparency in Data Collection",
                description="Inform consumers about personal information collection and use",
                requirement="Provide clear notice of personal information collection at or before collection",
                severity="high",
                applicable_data_types={DataClassification.PERSONAL},
                geographic_scope={"US-CA"}
            ),
            ComplianceRule(
                id="CCPA_RIGHT_TO_DELETE",
                framework=ComplianceFramework.CCPA,
                title="Right to Delete Personal Information",
                description="Allow consumers to request deletion of their personal information",
                requirement="Implement deletion capabilities for consumer personal information",
                severity="high",
                applicable_data_types={DataClassification.PERSONAL},
                geographic_scope={"US-CA"}
            )
        ]

        # HIPAA Rules
        hipaa_rules = [
            ComplianceRule(
                id="HIPAA_SAFEGUARDS",
                framework=ComplianceFramework.HIPAA,
                title="Administrative, Physical, and Technical Safeguards",
                description="Implement appropriate safeguards for protected health information",
                requirement="Ensure appropriate safeguards for PHI handling",
                severity="critical",
                applicable_data_types={DataClassification.HEALTH},
                geographic_scope={"US"}
            ),
            ComplianceRule(
                id="HIPAA_MINIMUM_NECESSARY",
                framework=ComplianceFramework.HIPAA,
                title="Minimum Necessary Standard",
                description="Limit access to minimum necessary PHI",
                requirement="Apply minimum necessary standards for PHI access and disclosure",
                severity="high",
                applicable_data_types={DataClassification.HEALTH},
                geographic_scope={"US"}
            )
        ]

        self._compliance_rules = {
            ComplianceFramework.GDPR: gdpr_rules,
            ComplianceFramework.CCPA: ccpa_rules,
            ComplianceFramework.HIPAA: hipaa_rules,
            # Additional frameworks would be added here
        }

    def enable_compliance_framework(self, framework: Union[ComplianceFramework, str]) -> None:
        """Enable a compliance framework."""
        if isinstance(framework, str):
            framework = ComplianceFramework(framework)

        with self._lock:
            self._active_frameworks.add(framework)

        self.logger.info("Compliance framework enabled", {"framework": framework.value})

    def disable_compliance_framework(self, framework: ComplianceFramework) -> None:
        """Disable a compliance framework."""
        with self._lock:
            self._active_frameworks.discard(framework)

        self.logger.info("Compliance framework disabled", {"framework": framework.value})

    def set_geographic_jurisdiction(self, jurisdictions: Set[str]) -> None:
        """Set geographic jurisdictions for compliance (ISO country codes)."""
        self._geographic_jurisdiction = jurisdictions

        self.logger.info("Geographic jurisdiction updated", {"jurisdictions": list(jurisdictions)})

    def check_compliance(self, data_classification: DataClassification,
                        processing_purpose: ProcessingPurpose,
                        geographic_location: str = "US") -> Dict[str, Any]:
        """Check compliance for a data processing activity."""
        violations = []
        recommendations = []
        applicable_rules = []

        with self._lock:
            for framework in self._active_frameworks:
                rules = self._compliance_rules.get(framework, [])

                for rule in rules:
                    # Check if rule applies to this data type and location
                    if (data_classification in rule.applicable_data_types and
                        (not rule.geographic_scope or geographic_location in rule.geographic_scope)):

                        applicable_rules.append({
                            "id": rule.id,
                            "framework": rule.framework.value,
                            "title": rule.title,
                            "severity": rule.severity
                        })

                        # Run automated check if available
                        if rule.automated_check:
                            try:
                                if not rule.automated_check({"classification": data_classification, "purpose": processing_purpose}):
                                    violations.append({
                                        "rule_id": rule.id,
                                        "framework": rule.framework.value,
                                        "severity": rule.severity,
                                        "description": rule.description,
                                        "requirement": rule.requirement,
                                        "remediation": rule.remediation_guidance
                                    })
                            except Exception as e:
                                self.logger.warning("Compliance check failed", {
                                    "rule_id": rule.id,
                                    "error": str(e)
                                })

        # Generate recommendations
        if data_classification in {DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL}:
            recommendations.extend([
                "Consider pseudonymization or anonymization where possible",
                "Implement data minimization principles",
                "Ensure appropriate retention periods are applied",
                "Document legal basis for processing"
            ])

        if data_classification == DataClassification.HEALTH:
            recommendations.extend([
                "Implement HIPAA-compliant access controls",
                "Apply minimum necessary standards",
                "Ensure appropriate encryption for PHI"
            ])

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendations": recommendations,
            "applicable_rules": applicable_rules,
            "data_classification": data_classification.value,
            "processing_purpose": processing_purpose.value,
            "geographic_location": geographic_location,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive compliance report."""
        with self._lock:
            # Processing activity summary
            processing_summary = {}
            for record in self.privacy_controls._data_processing_log:
                key = f"{record.data_classification.value}_{record.processing_purpose.value}"
                processing_summary[key] = processing_summary.get(key, 0) + 1

            # Consent summary
            consent_summary = {
                "total_consents": len(self.privacy_controls._consent_records),
                "active_consents": sum(1 for c in self.privacy_controls._consent_records.values() if c["valid"]),
                "withdrawn_consents": sum(1 for c in self.privacy_controls._consent_records.values() if not c["valid"])
            }

            return {
                "report_generated_at": datetime.now(timezone.utc).isoformat(),
                "active_frameworks": [f.value for f in self._active_frameworks],
                "geographic_jurisdictions": list(self._geographic_jurisdiction),
                "processing_activity_summary": processing_summary,
                "consent_summary": consent_summary,
                "retention_policies": {
                    k.value: v.value
                    for k, v in self.privacy_controls._data_retention_policies.items()
                },
                "total_processing_records": len(self.privacy_controls._data_processing_log),
                "data_classifications_in_use": list(set(
                    record.data_classification.value
                    for record in self.privacy_controls._data_processing_log
                ))
            }

    def process_data_subject_request(self, request_type: str, data_subject_id: str) -> Dict[str, Any]:
        """Process a data subject rights request."""
        if request_type.lower() == "export":
            data = self.privacy_controls.export_data_for_subject(data_subject_id)

            return {
                "request_type": "export",
                "data_subject_id": data_subject_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "data": data
            }

        elif request_type.lower() == "delete":
            deleted = self.privacy_controls.delete_subject_data(data_subject_id)

            return {
                "request_type": "delete",
                "data_subject_id": data_subject_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed" if deleted else "no_data_found",
                "data_deleted": deleted
            }

        else:
            return {
                "request_type": request_type,
                "data_subject_id": data_subject_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "unsupported_request_type"
            }


# Global compliance engine instance
_compliance_engine: Optional[ComplianceEngine] = None
_engine_lock = threading.Lock()


def get_compliance_engine() -> ComplianceEngine:
    """Get the global compliance engine instance."""
    global _compliance_engine

    if _compliance_engine is None:
        with _engine_lock:
            if _compliance_engine is None:
                _compliance_engine = ComplianceEngine()

    return _compliance_engine


def log_data_processing(data_classification: DataClassification,
                       processing_purpose: ProcessingPurpose,
                       data_subject_id: Optional[str] = None,
                       additional_details: Optional[Dict[str, Any]] = None) -> str:
    """Log a data processing activity for compliance."""
    engine = get_compliance_engine()

    record = DataProcessingRecord(
        data_subject_id=data_subject_id,
        data_classification=data_classification,
        processing_purpose=processing_purpose,
        processing_details=additional_details or {}
    )

    engine.privacy_controls.log_data_processing(record)
    return record.id


def check_data_compliance(data_classification: DataClassification,
                         processing_purpose: ProcessingPurpose,
                         geographic_location: str = "US") -> Dict[str, Any]:
    """Check compliance for data processing."""
    return get_compliance_engine().check_compliance(
        data_classification, processing_purpose, geographic_location
    )
