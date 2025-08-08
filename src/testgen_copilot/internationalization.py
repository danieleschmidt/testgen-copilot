"""
Internationalization (i18n) support for TestGen-Copilot.

Provides multi-language support, locale handling, and cultural compliance
for global deployment and accessibility.
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import threading
import locale

from .logging_config import get_core_logger


class SupportedLocales(str, Enum):
    """Supported locales for TestGen-Copilot."""
    EN_US = "en_US"  # English (United States)
    EN_GB = "en_GB"  # English (United Kingdom) 
    ES_ES = "es_ES"  # Spanish (Spain)
    ES_MX = "es_MX"  # Spanish (Mexico)
    FR_FR = "fr_FR"  # French (France)
    DE_DE = "de_DE"  # German (Germany)
    IT_IT = "it_IT"  # Italian (Italy)
    PT_BR = "pt_BR"  # Portuguese (Brazil)
    JA_JP = "ja_JP"  # Japanese (Japan)
    KO_KR = "ko_KR"  # Korean (South Korea)
    ZH_CN = "zh_CN"  # Chinese (Simplified)
    ZH_TW = "zh_TW"  # Chinese (Traditional)
    RU_RU = "ru_RU"  # Russian (Russia)
    AR_SA = "ar_SA"  # Arabic (Saudi Arabia)
    HI_IN = "hi_IN"  # Hindi (India)
    NL_NL = "nl_NL"  # Dutch (Netherlands)
    SV_SE = "sv_SE"  # Swedish (Sweden)
    NO_NO = "no_NO"  # Norwegian (Norway)
    DA_DK = "da_DK"  # Danish (Denmark)
    FI_FI = "fi_FI"  # Finnish (Finland)
    PL_PL = "pl_PL"  # Polish (Poland)
    TR_TR = "tr_TR"  # Turkish (Turkey)
    TH_TH = "th_TH"  # Thai (Thailand)
    VI_VN = "vi_VN"  # Vietnamese (Vietnam)


@dataclass
class LocaleInfo:
    """Information about a locale."""
    code: str
    name: str
    native_name: str
    language: str
    country: str
    rtl: bool = False  # Right-to-left text direction
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "1,234.56"
    currency_symbol: str = "$"
    currency_code: str = "USD"


class LocalizationManager:
    """Manages localization and internationalization features."""
    
    def __init__(self):
        self.logger = get_core_logger()
        self._current_locale = SupportedLocales.EN_US
        self._translations: Dict[str, Dict[str, str]] = {}
        self._locale_info: Dict[str, LocaleInfo] = {}
        self._lock = threading.RLock()
        
        # Initialize locale information
        self._init_locale_info()
        
        # Load default translations
        self._load_translations()
    
    def _init_locale_info(self) -> None:
        """Initialize locale information for all supported locales."""
        self._locale_info = {
            SupportedLocales.EN_US: LocaleInfo(
                "en_US", "English (US)", "English (US)", 
                "English", "United States", False, "%m/%d/%Y", "%I:%M:%S %p",
                "1,234.56", "$", "USD"
            ),
            SupportedLocales.EN_GB: LocaleInfo(
                "en_GB", "English (UK)", "English (UK)",
                "English", "United Kingdom", False, "%d/%m/%Y", "%H:%M:%S",
                "1,234.56", "£", "GBP"
            ),
            SupportedLocales.ES_ES: LocaleInfo(
                "es_ES", "Spanish (Spain)", "Español (España)",
                "Spanish", "Spain", False, "%d/%m/%Y", "%H:%M:%S",
                "1.234,56", "€", "EUR"
            ),
            SupportedLocales.FR_FR: LocaleInfo(
                "fr_FR", "French (France)", "Français (France)",
                "French", "France", False, "%d/%m/%Y", "%H:%M:%S",
                "1 234,56", "€", "EUR"
            ),
            SupportedLocales.DE_DE: LocaleInfo(
                "de_DE", "German (Germany)", "Deutsch (Deutschland)",
                "German", "Germany", False, "%d.%m.%Y", "%H:%M:%S",
                "1.234,56", "€", "EUR"
            ),
            SupportedLocales.JA_JP: LocaleInfo(
                "ja_JP", "Japanese (Japan)", "日本語 (日本)",
                "Japanese", "Japan", False, "%Y/%m/%d", "%H:%M:%S",
                "1,234.56", "¥", "JPY"
            ),
            SupportedLocales.ZH_CN: LocaleInfo(
                "zh_CN", "Chinese (Simplified)", "中文 (简体)",
                "Chinese", "China", False, "%Y年%m月%d日", "%H:%M:%S",
                "1,234.56", "¥", "CNY"
            ),
            SupportedLocales.AR_SA: LocaleInfo(
                "ar_SA", "Arabic (Saudi Arabia)", "العربية (السعودية)",
                "Arabic", "Saudi Arabia", True, "%d/%m/%Y", "%H:%M:%S",
                "1,234.56", "ر.س", "SAR"
            ),
            # Add more locales as needed
        }
    
    def _load_translations(self) -> None:
        """Load translation files for all supported locales."""
        translations_dir = Path(__file__).parent / "locales"
        
        # Create default translations if directory doesn't exist
        if not translations_dir.exists():
            translations_dir.mkdir(exist_ok=True)
            self._create_default_translations(translations_dir)
        
        # Load translation files
        for locale_code in SupportedLocales:
            translation_file = translations_dir / f"{locale_code.value}.json"
            
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self._translations[locale_code.value] = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load translations for {locale_code.value}", {
                        "error": str(e),
                        "file": str(translation_file)
                    })
                    # Fall back to English
                    self._translations[locale_code.value] = self._translations.get(
                        SupportedLocales.EN_US.value, {}
                    )
            else:
                # Create default translation file
                self._create_translation_file(translation_file, locale_code.value)
    
    def _create_default_translations(self, translations_dir: Path) -> None:
        """Create default translation files."""
        default_translations = {
            # Core application messages
            "app.name": "TestGen Copilot",
            "app.description": "AI-powered test generation and security analysis tool",
            "app.version": "Version",
            
            # Command line interface
            "cli.generating_tests": "Generating tests for {filename}...",
            "cli.tests_generated": "Tests generated successfully",
            "cli.coverage_analysis": "Analyzing code coverage...",
            "cli.security_scan": "Performing security scan...",
            "cli.analysis_complete": "Analysis complete",
            "cli.error": "Error occurred: {error}",
            "cli.file_not_found": "File not found: {filename}",
            "cli.invalid_config": "Invalid configuration file",
            
            # Test generation
            "test.generation.started": "Test generation started",
            "test.generation.completed": "Test generation completed",
            "test.generation.failed": "Test generation failed",
            "test.class_generated": "Generated test class: {class_name}",
            "test.method_generated": "Generated test method: {method_name}",
            "test.edge_cases": "Including edge case tests",
            "test.error_handling": "Including error handling tests",
            "test.integration": "Including integration tests",
            
            # Security analysis
            "security.scan_started": "Security scan started",
            "security.scan_completed": "Security scan completed",
            "security.vulnerability_found": "Vulnerability found: {vulnerability}",
            "security.critical_issue": "Critical security issue detected",
            "security.recommendations": "Security recommendations",
            "security.no_issues": "No security issues found",
            
            # Coverage analysis
            "coverage.analyzing": "Analyzing code coverage...",
            "coverage.result": "Coverage: {percentage}%",
            "coverage.target_met": "Coverage target met",
            "coverage.target_missed": "Coverage target not met",
            "coverage.missing_lines": "Missing coverage for lines: {lines}",
            
            # Quality assessment
            "quality.score": "Quality score: {score}%",
            "quality.excellent": "Excellent test quality",
            "quality.good": "Good test quality",
            "quality.needs_improvement": "Test quality needs improvement",
            
            # File operations
            "file.reading": "Reading file: {filename}",
            "file.writing": "Writing file: {filename}",
            "file.parsing": "Parsing file: {filename}",
            "file.analysis": "Analyzing file structure",
            
            # Error messages
            "error.file_not_readable": "Cannot read file: {filename}",
            "error.invalid_syntax": "Invalid syntax in file: {filename}",
            "error.permission_denied": "Permission denied: {filename}",
            "error.network_error": "Network error: {error}",
            "error.unexpected": "Unexpected error occurred",
            
            # Success messages
            "success.operation_completed": "Operation completed successfully",
            "success.tests_created": "Test files created successfully",
            "success.analysis_saved": "Analysis results saved",
            
            # Configuration
            "config.loading": "Loading configuration...",
            "config.loaded": "Configuration loaded successfully",
            "config.invalid": "Invalid configuration",
            "config.default": "Using default configuration",
            
            # Performance and scaling
            "perf.optimization_started": "Performance optimization started",
            "perf.cache_hit": "Cache hit for {key}",
            "perf.cache_miss": "Cache miss for {key}",
            "perf.scaling_up": "Scaling up resources",
            "perf.scaling_down": "Scaling down resources",
            "perf.load_balanced": "Load balanced to worker {worker_id}",
            
            # Monitoring and health
            "monitor.health_check": "Performing health check",
            "monitor.system_healthy": "System is healthy",
            "monitor.alert_triggered": "Alert triggered: {alert}",
            "monitor.metric_recorded": "Metric recorded: {metric}",
            
            # Date and time formats
            "format.date_short": "{month}/{day}/{year}",
            "format.date_long": "{day} {month} {year}",
            "format.time_short": "{hour}:{minute}",
            "format.time_long": "{hour}:{minute}:{second}",
            "format.datetime": "{date} at {time}",
        }
        
        # Create English translation file
        en_file = translations_dir / f"{SupportedLocales.EN_US.value}.json"
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, indent=2, ensure_ascii=False)
        
        # Store in memory
        self._translations[SupportedLocales.EN_US.value] = default_translations
        
        self.logger.info("Created default translation files", {
            "translations_dir": str(translations_dir),
            "translations_count": len(default_translations)
        })
    
    def _create_translation_file(self, translation_file: Path, locale_code: str) -> None:
        """Create a translation file for a specific locale."""
        # Start with English translations as base
        base_translations = self._translations.get(
            SupportedLocales.EN_US.value, {}
        ).copy()
        
        # Add locale-specific translations here
        # This would typically be done by professional translators
        # For now, we use the English version with locale markers
        
        try:
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(base_translations, f, indent=2, ensure_ascii=False)
            
            self._translations[locale_code] = base_translations
            
            self.logger.info("Created translation file", {
                "locale": locale_code,
                "file": str(translation_file)
            })
            
        except Exception as e:
            self.logger.error("Failed to create translation file", {
                "locale": locale_code,
                "file": str(translation_file),
                "error": str(e)
            })
    
    def set_locale(self, locale_code: Union[str, SupportedLocales]) -> bool:
        """Set the current locale."""
        if isinstance(locale_code, str):
            # Try to find matching enum value
            try:
                locale_code = SupportedLocales(locale_code)
            except ValueError:
                self.logger.warning(f"Unsupported locale: {locale_code}")
                return False
        
        with self._lock:
            self._current_locale = locale_code
            
            # Try to set system locale if possible
            try:
                system_locale = f"{locale_code.value}.UTF-8"
                locale.setlocale(locale.LC_ALL, system_locale)
            except locale.Error:
                # Fallback to C locale
                try:
                    locale.setlocale(locale.LC_ALL, "C.UTF-8")
                except locale.Error:
                    pass  # Use whatever is available
            
            self.logger.info("Locale changed", {
                "new_locale": locale_code.value,
                "locale_name": self._locale_info.get(locale_code.value, {}).name
            })
            
        return True
    
    def get_current_locale(self) -> SupportedLocales:
        """Get the current locale."""
        return self._current_locale
    
    def get_locale_info(self, locale_code: Optional[Union[str, SupportedLocales]] = None) -> Optional[LocaleInfo]:
        """Get information about a locale."""
        if locale_code is None:
            locale_code = self._current_locale
            
        if isinstance(locale_code, SupportedLocales):
            locale_code = locale_code.value
            
        return self._locale_info.get(locale_code)
    
    def translate(self, key: str, locale_code: Optional[Union[str, SupportedLocales]] = None, **kwargs) -> str:
        """Translate a message key to the current or specified locale."""
        if locale_code is None:
            locale_code = self._current_locale
            
        if isinstance(locale_code, SupportedLocales):
            locale_code = locale_code.value
            
        translations = self._translations.get(locale_code, {})
        
        # Get the translation
        message = translations.get(key, key)  # Fall back to key if not found
        
        # Handle parameter substitution
        if kwargs:
            try:
                message = message.format(**kwargs)
            except (KeyError, ValueError) as e:
                self.logger.warning("Translation parameter error", {
                    "key": key,
                    "locale": locale_code,
                    "error": str(e)
                })
                # Return the untranslated message with parameters if formatting fails
                return f"{message} {kwargs}"
        
        return message
    
    def t(self, key: str, **kwargs) -> str:
        """Shorthand for translate()."""
        return self.translate(key, **kwargs)
    
    def format_date(self, date: datetime, locale_code: Optional[Union[str, SupportedLocales]] = None) -> str:
        """Format a date according to locale conventions."""
        locale_info = self.get_locale_info(locale_code)
        if locale_info:
            try:
                return date.strftime(locale_info.date_format)
            except:
                pass
        
        # Default format
        return date.strftime("%Y-%m-%d")
    
    def format_time(self, time: datetime, locale_code: Optional[Union[str, SupportedLocales]] = None) -> str:
        """Format a time according to locale conventions."""
        locale_info = self.get_locale_info(locale_code)
        if locale_info:
            try:
                return time.strftime(locale_info.time_format)
            except:
                pass
        
        # Default format
        return time.strftime("%H:%M:%S")
    
    def format_datetime(self, dt: datetime, locale_code: Optional[Union[str, SupportedLocales]] = None) -> str:
        """Format a datetime according to locale conventions."""
        date_str = self.format_date(dt, locale_code)
        time_str = self.format_time(dt, locale_code)
        
        return self.translate("format.datetime", locale_code=locale_code, date=date_str, time=time_str)
    
    def format_number(self, number: float, locale_code: Optional[Union[str, SupportedLocales]] = None) -> str:
        """Format a number according to locale conventions."""
        locale_info = self.get_locale_info(locale_code)
        
        if locale_info:
            # Simple number formatting based on locale
            if "," in locale_info.number_format and "." in locale_info.number_format:
                # European style: 1.234,56
                if locale_info.number_format.index(",") > locale_info.number_format.index("."):
                    return f"{number:,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".")
                
            # US style: 1,234.56 (default)
            return f"{number:,.2f}"
        
        return f"{number:,.2f}"
    
    def format_currency(self, amount: float, locale_code: Optional[Union[str, SupportedLocales]] = None) -> str:
        """Format currency according to locale conventions."""
        locale_info = self.get_locale_info(locale_code)
        number_str = self.format_number(amount, locale_code)
        
        if locale_info:
            return f"{locale_info.currency_symbol}{number_str}"
        
        return f"${number_str}"
    
    def get_supported_locales(self) -> List[LocaleInfo]:
        """Get list of all supported locales."""
        return list(self._locale_info.values())
    
    def is_rtl_locale(self, locale_code: Optional[Union[str, SupportedLocales]] = None) -> bool:
        """Check if the locale uses right-to-left text direction."""
        locale_info = self.get_locale_info(locale_code)
        return locale_info.rtl if locale_info else False
    
    def get_translation_completeness(self, locale_code: Union[str, SupportedLocales]) -> float:
        """Get the percentage of translations completed for a locale."""
        if isinstance(locale_code, SupportedLocales):
            locale_code = locale_code.value
            
        base_translations = self._translations.get(SupportedLocales.EN_US.value, {})
        locale_translations = self._translations.get(locale_code, {})
        
        if not base_translations:
            return 0.0
            
        total_keys = len(base_translations)
        translated_keys = sum(
            1 for key in base_translations 
            if key in locale_translations and locale_translations[key] != base_translations[key]
        )
        
        return (translated_keys / total_keys) * 100 if total_keys > 0 else 0.0


# Global localization manager instance
_localization_manager: Optional[LocalizationManager] = None
_manager_lock = threading.Lock()


def get_localization_manager() -> LocalizationManager:
    """Get the global localization manager instance."""
    global _localization_manager
    
    if _localization_manager is None:
        with _manager_lock:
            if _localization_manager is None:
                _localization_manager = LocalizationManager()
    
    return _localization_manager


def set_global_locale(locale_code: Union[str, SupportedLocales]) -> bool:
    """Set the global locale for the application."""
    return get_localization_manager().set_locale(locale_code)


def t(key: str, **kwargs) -> str:
    """Global translation function."""
    return get_localization_manager().translate(key, **kwargs)


def format_localized_datetime(dt: datetime, locale_code: Optional[Union[str, SupportedLocales]] = None) -> str:
    """Format datetime with localization."""
    return get_localization_manager().format_datetime(dt, locale_code)


def format_localized_currency(amount: float, locale_code: Optional[Union[str, SupportedLocales]] = None) -> str:
    """Format currency with localization."""
    return get_localization_manager().format_currency(amount, locale_code)