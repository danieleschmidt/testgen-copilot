"""Tests for Generation 4 global-first features - i18n, compliance, and multi-region."""

import pytest
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from testgen_copilot.internationalization import (
    LocalizationManager, SupportedLocales, get_localization_manager, set_global_locale, t
)
from testgen_copilot.compliance import (
    ComplianceEngine, ComplianceFramework, DataClassification, ProcessingPurpose,
    RetentionPeriod, DataProcessingRecord, get_compliance_engine, log_data_processing
)
from testgen_copilot.multi_region import (
    MultiRegionManager, Region, RegionConfig, DataResidencyRequirement,
    get_multi_region_manager, store_data_globally
)


class TestInternationalization:
    """Test internationalization and localization features."""
    
    def test_localization_manager_initialization(self):
        """Test LocalizationManager initialization."""
        manager = LocalizationManager()
        
        assert manager.get_current_locale() == SupportedLocales.EN_US
        assert len(manager._locale_info) > 0
        assert SupportedLocales.EN_US.value in manager._translations
        
    def test_set_locale(self):
        """Test setting different locales."""
        manager = LocalizationManager()
        
        # Test valid locale
        result = manager.set_locale(SupportedLocales.ES_ES)
        assert result is True
        assert manager.get_current_locale() == SupportedLocales.ES_ES
        
        # Test invalid locale string
        result = manager.set_locale("invalid_locale")
        assert result is False
        
    def test_translation_basic(self):
        """Test basic translation functionality."""
        manager = LocalizationManager()
        
        # Test existing key
        translated = manager.translate("app.name")
        assert translated == "TestGen Copilot"
        
        # Test non-existent key (should return key itself)
        translated = manager.translate("non.existent.key")
        assert translated == "non.existent.key"
        
    def test_translation_with_parameters(self):
        """Test translation with parameter substitution."""
        manager = LocalizationManager()
        
        # Test parameter substitution
        translated = manager.translate("cli.generating_tests", filename="test.py")
        assert "test.py" in translated
        
        # Test shorthand t() function
        translated = manager.t("cli.error", error="File not found")
        assert "File not found" in translated
        
    def test_date_time_formatting(self):
        """Test locale-specific date and time formatting."""
        manager = LocalizationManager()
        test_date = datetime(2023, 12, 25, 15, 30, 45, tzinfo=timezone.utc)
        
        # US format
        manager.set_locale(SupportedLocales.EN_US)
        date_str = manager.format_date(test_date)
        assert "12/25/2023" in date_str or "2023" in date_str
        
        # German format
        manager.set_locale(SupportedLocales.DE_DE)
        date_str = manager.format_date(test_date)
        assert "25.12.2023" in date_str or "2023" in date_str
        
    def test_number_formatting(self):
        """Test locale-specific number formatting."""
        manager = LocalizationManager()
        test_number = 1234.56
        
        # US format (should have comma separator)
        manager.set_locale(SupportedLocales.EN_US)
        formatted = manager.format_number(test_number)
        assert "1,234" in formatted
        
        # Format as currency
        currency = manager.format_currency(test_number)
        assert "$" in currency
        
    def test_rtl_locale_detection(self):
        """Test right-to-left locale detection."""
        manager = LocalizationManager()
        
        # Arabic is RTL
        manager.set_locale(SupportedLocales.AR_SA)
        assert manager.is_rtl_locale() is True
        
        # English is LTR
        manager.set_locale(SupportedLocales.EN_US)
        assert manager.is_rtl_locale() is False
        
    def test_supported_locales(self):
        """Test getting supported locales list."""
        manager = LocalizationManager()
        locales = manager.get_supported_locales()
        
        assert len(locales) > 0
        assert any(locale.code == "en_US" for locale in locales)
        
    def test_global_functions(self):
        """Test global localization functions."""
        # Test global locale setting
        result = set_global_locale(SupportedLocales.FR_FR)
        assert result is True
        
        # Test global translation function
        translated = t("app.name")
        assert translated == "TestGen Copilot"
        
        # Test global manager access
        manager = get_localization_manager()
        assert manager.get_current_locale() == SupportedLocales.FR_FR


class TestCompliance:
    """Test compliance and data governance features."""
    
    def test_compliance_engine_initialization(self):
        """Test ComplianceEngine initialization."""
        engine = ComplianceEngine()
        
        assert len(engine._compliance_rules) > 0
        assert ComplianceFramework.GDPR in engine._compliance_rules
        assert ComplianceFramework.CCPA in engine._compliance_rules
        
    def test_enable_disable_frameworks(self):
        """Test enabling and disabling compliance frameworks."""
        engine = ComplianceEngine()
        
        # Enable GDPR
        engine.enable_compliance_framework(ComplianceFramework.GDPR)
        assert ComplianceFramework.GDPR in engine._active_frameworks
        
        # Disable GDPR
        engine.disable_compliance_framework(ComplianceFramework.GDPR)
        assert ComplianceFramework.GDPR not in engine._active_frameworks
        
    def test_data_processing_logging(self):
        """Test data processing logging."""
        engine = ComplianceEngine()
        
        # Log data processing
        record = DataProcessingRecord(
            data_classification=DataClassification.PERSONAL,
            processing_purpose=ProcessingPurpose.TESTING
        )
        engine.privacy_controls.log_data_processing(record)
        
        assert len(engine.privacy_controls._data_processing_log) > 0
        assert record in engine.privacy_controls._data_processing_log
        
    def test_consent_management(self):
        """Test consent recording and withdrawal."""
        engine = ComplianceEngine()
        
        # Record consent
        consent_id = engine.privacy_controls.record_consent(
            "user123",
            [ProcessingPurpose.TESTING, ProcessingPurpose.ANALYTICS],
            {"consent_method": "web_form"}
        )
        
        assert consent_id is not None
        assert len(engine.privacy_controls._consent_records) > 0
        
        # Check consent exists
        has_consent = engine.privacy_controls.has_valid_consent(
            "user123", ProcessingPurpose.TESTING
        )
        assert has_consent is True
        
        # Withdraw consent
        withdrawn = engine.privacy_controls.withdraw_consent(consent_id)
        assert withdrawn is True
        
        # Check consent no longer valid
        has_consent = engine.privacy_controls.has_valid_consent(
            "user123", ProcessingPurpose.TESTING
        )
        assert has_consent is False
        
    def test_data_classification(self):
        """Test automatic data classification."""
        engine = ComplianceEngine()
        
        # Test email classification
        classifications = engine.privacy_controls.classify_data_content(
            "Contact us at support@example.com"
        )
        assert DataClassification.PERSONAL in classifications
        
        # Test credit card classification
        classifications = engine.privacy_controls.classify_data_content(
            "Card number: 4532 1234 5678 9012"
        )
        assert DataClassification.FINANCIAL in classifications
        
        # Test generic content
        classifications = engine.privacy_controls.classify_data_content(
            "This is just some regular text"
        )
        assert DataClassification.INTERNAL in classifications
        
    def test_data_subject_rights(self):
        """Test data subject rights (export/delete)."""
        engine = ComplianceEngine()
        
        # Add some data for a subject
        record = DataProcessingRecord(
            data_subject_id="user456",
            data_classification=DataClassification.PERSONAL,
            processing_purpose=ProcessingPurpose.TESTING
        )
        engine.privacy_controls.log_data_processing(record)
        
        # Export data
        exported = engine.privacy_controls.export_data_for_subject("user456")
        assert exported["data_subject_id"] == "user456"
        assert len(exported["processing_records"]) > 0
        
        # Delete data
        deleted = engine.privacy_controls.delete_subject_data("user456")
        assert deleted is True
        
        # Verify deletion
        exported_after = engine.privacy_controls.export_data_for_subject("user456")
        assert len(exported_after["processing_records"]) == 0
        
    def test_compliance_checking(self):
        """Test compliance checking for data processing."""
        engine = ComplianceEngine()
        engine.enable_compliance_framework(ComplianceFramework.GDPR)
        
        # Check compliance for personal data
        result = engine.check_compliance(
            DataClassification.PERSONAL,
            ProcessingPurpose.TESTING,
            "EU"
        )
        
        assert "compliant" in result
        assert "violations" in result
        assert "recommendations" in result
        assert result["data_classification"] == "PERSONAL"
        
    def test_compliance_report(self):
        """Test compliance report generation."""
        engine = ComplianceEngine()
        engine.enable_compliance_framework(ComplianceFramework.GDPR)
        
        # Add some test data
        record = DataProcessingRecord(
            data_classification=DataClassification.PERSONAL,
            processing_purpose=ProcessingPurpose.TESTING
        )
        engine.privacy_controls.log_data_processing(record)
        
        # Generate report
        report = engine.generate_compliance_report()
        
        assert "report_generated_at" in report
        assert "active_frameworks" in report
        assert "processing_activity_summary" in report
        assert ComplianceFramework.GDPR.value in report["active_frameworks"]
        
    def test_global_compliance_functions(self):
        """Test global compliance functions."""
        # Test data processing logging
        record_id = log_data_processing(
            DataClassification.INTERNAL,
            ProcessingPurpose.TESTING,
            data_subject_id="test_user"
        )
        
        assert record_id is not None
        
        # Test compliance engine access
        engine = get_compliance_engine()
        assert engine is not None
        assert len(engine.privacy_controls._data_processing_log) > 0


class TestMultiRegion:
    """Test multi-region deployment features."""
    
    def test_multi_region_manager_initialization(self):
        """Test MultiRegionManager initialization."""
        manager = MultiRegionManager()
        
        assert len(manager._regions) > 0
        assert manager.get_current_region() is not None
        assert Region.US_EAST_1 in manager._regions
        
    def test_region_configuration(self):
        """Test region configuration."""
        manager = MultiRegionManager()
        
        # Configure a new region
        config = RegionConfig(
            region=Region.AP_SOUTHEAST_1,
            name="Asia Pacific Test",
            country_code="SG",
            jurisdiction="Singapore",
            data_residency=DataResidencyRequirement.STRICT
        )
        
        manager.configure_region(config)
        assert Region.AP_SOUTHEAST_1 in manager._regions
        assert manager._regions[Region.AP_SOUTHEAST_1].data_residency == DataResidencyRequirement.STRICT
        
    def test_current_region_management(self):
        """Test current region setting and getting."""
        manager = MultiRegionManager()
        
        # Set valid region
        result = manager.set_current_region(Region.EU_WEST_1)
        assert result is True
        assert manager.get_current_region() == Region.EU_WEST_1
        
        # Try to set disabled region
        config = manager._regions[Region.EU_WEST_1]
        config.enabled = False
        
        result = manager.set_current_region(Region.EU_WEST_1)
        assert result is False
        
    def test_data_storage_with_compliance(self):
        """Test data storage with compliance considerations."""
        manager = MultiRegionManager()
        
        # Store personal data
        location = manager.store_data_with_compliance(
            "test_data_1",
            {"user": "john", "email": "john@example.com"},
            DataClassification.PERSONAL,
            "EU"
        )
        
        assert location is not None
        assert location.data_id == "test_data_1"
        assert location.data_classification == DataClassification.PERSONAL
        
    def test_data_retrieval(self):
        """Test data retrieval from optimal region."""
        manager = MultiRegionManager()
        
        # Store data first
        location = manager.store_data_with_compliance(
            "test_data_2",
            {"content": "test content"},
            DataClassification.INTERNAL,
            "US"
        )
        
        # Retrieve data
        content, region = manager.retrieve_data("test_data_2", "US")
        
        assert content is not None
        assert region is not None
        assert "test_data_2" in str(content) or region in manager._regions
        
    def test_region_status(self):
        """Test region status reporting."""
        manager = MultiRegionManager()
        
        status = manager.get_region_status()
        
        assert "current_region" in status
        assert "total_regions" in status
        assert "enabled_regions" in status
        assert "regions" in status
        assert len(status["regions"]) > 0
        
    def test_compliance_regions(self):
        """Test getting compliance-specific regions."""
        manager = MultiRegionManager()
        
        # Configure EU region with GDPR compliance
        config = manager._regions[Region.EU_WEST_1]
        config.compliance_frameworks.add(ComplianceFramework.GDPR)
        
        # Get GDPR compliant regions
        gdpr_regions = manager.get_compliance_regions(ComplianceFramework.GDPR)
        
        assert Region.EU_WEST_1 in gdpr_regions
        
    def test_data_placement_validation(self):
        """Test data placement validation."""
        manager = MultiRegionManager()
        
        # Validate personal data in EU region
        validation = manager.validate_data_placement(
            DataClassification.PERSONAL,
            Region.EU_WEST_1
        )
        
        assert "valid" in validation
        assert validation["data_classification"] == "PERSONAL"
        assert validation["target_region"] == "eu-west-1"
        
    def test_region_failover(self):
        """Test region failover handling."""
        manager = MultiRegionManager()
        
        # Test failover for region with backup
        backup_region = manager.handle_region_failover(Region.US_EAST_1)
        
        # Should return backup region or None
        assert backup_region is None or backup_region in manager._regions
        
    def test_global_multi_region_functions(self):
        """Test global multi-region functions."""
        # Test global data storage
        location = store_data_globally(
            "global_test_data",
            {"test": "data"},
            DataClassification.INTERNAL,
            "US"
        )
        
        assert location is not None
        assert location.data_id == "global_test_data"
        
        # Test global manager access
        manager = get_multi_region_manager()
        assert manager is not None
        assert len(manager._regions) > 0


class TestIntegration:
    """Integration tests for Generation 4 features."""
    
    def test_compliance_with_i18n(self):
        """Test compliance reporting with internationalization."""
        # Set German locale
        set_global_locale(SupportedLocales.DE_DE)
        
        # Generate compliance report
        engine = get_compliance_engine()
        engine.enable_compliance_framework(ComplianceFramework.GDPR)
        
        report = engine.generate_compliance_report()
        assert report is not None
        
        # Translate compliance messages
        translated = t("security.scan_started")
        assert translated is not None
        
    def test_multi_region_with_compliance(self):
        """Test multi-region deployment with compliance requirements."""
        manager = get_multi_region_manager()
        engine = get_compliance_engine()
        
        # Enable GDPR compliance
        engine.enable_compliance_framework(ComplianceFramework.GDPR)
        
        # Store GDPR-subject data
        location = manager.store_data_with_compliance(
            "gdpr_data",
            {"user_id": "eu_user", "preferences": "settings"},
            DataClassification.PERSONAL,
            "EU"
        )
        
        assert location is not None
        
        # Check compliance for this data
        compliance_check = engine.check_compliance(
            DataClassification.PERSONAL,
            ProcessingPurpose.TESTING,
            "EU"
        )
        
        assert compliance_check is not None
        assert "compliant" in compliance_check
        
    def test_end_to_end_global_workflow(self):
        """Test complete global-first workflow."""
        # 1. Set locale for German user
        set_global_locale(SupportedLocales.DE_DE)
        
        # 2. Enable compliance frameworks
        engine = get_compliance_engine()
        engine.enable_compliance_framework(ComplianceFramework.GDPR)
        
        # 3. Store data in appropriate region
        manager = get_multi_region_manager()
        location = manager.store_data_with_compliance(
            "global_workflow_data",
            {"user_data": "sensitive information"},
            DataClassification.PERSONAL,
            "DE"  # German user
        )
        
        # 4. Log data processing for compliance
        record_id = log_data_processing(
            DataClassification.PERSONAL,
            ProcessingPurpose.TESTING,
            data_subject_id="de_user_123"
        )
        
        # 5. Generate localized compliance report
        report = engine.generate_compliance_report()
        
        # 6. Verify everything is working
        assert location is not None
        assert record_id is not None
        assert report is not None
        assert t("app.name") == "TestGen Copilot"
        
        # 7. Test data subject rights
        exported = engine.privacy_controls.export_data_for_subject("de_user_123")
        assert exported["data_subject_id"] == "de_user_123"
        
        # 8. Get region status
        status = manager.get_region_status()
        assert "current_region" in status
        
    def test_performance_with_global_features(self):
        """Test performance impact of global features."""
        import time
        
        # Time localization operations
        start_time = time.time()
        
        for i in range(100):
            t(f"test.key.{i % 10}")
            
        localization_time = time.time() - start_time
        assert localization_time < 1.0  # Should be fast
        
        # Time compliance operations
        start_time = time.time()
        
        engine = get_compliance_engine()
        for i in range(50):
            engine.check_compliance(
                DataClassification.INTERNAL,
                ProcessingPurpose.TESTING,
                "US"
            )
            
        compliance_time = time.time() - start_time
        assert compliance_time < 2.0  # Should be reasonable
        
        # Time multi-region operations
        start_time = time.time()
        
        manager = get_multi_region_manager()
        for i in range(20):
            manager.validate_data_placement(
                DataClassification.INTERNAL,
                Region.US_EAST_1
            )
            
        multi_region_time = time.time() - start_time
        assert multi_region_time < 1.0  # Should be fast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])