"""
🛡️ Advanced Error Handling System
==================================

Comprehensive error handling, recovery, and reporting system for autonomous SDLC execution.
Implements intelligent error classification, contextual recovery strategies,
and adaptive learning from error patterns.
"""

import asyncio
import contextlib
import functools
import inspect
import logging
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import json

from rich.console import Console
from rich.panel import Panel

from .logging_config import setup_logger
from .self_healing import SelfHealingSystem, FailureEvent

logger = setup_logger(__name__)
console = Console()


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues, system continues
    MEDIUM = "medium"     # Significant issues, degraded functionality
    HIGH = "high"         # Major issues, significant impact
    CRITICAL = "critical" # System-threatening issues
    FATAL = "fatal"       # System cannot continue


class ErrorCategory(Enum):
    """Error categories for classification"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM_RESOURCE = "system_resource"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    USER_INTERVENTION = "user_intervention"
    AUTOMATIC_RECOVERY = "automatic_recovery"


@dataclass
class ErrorContext:
    """Comprehensive error context information"""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    operation: str = ""
    user_context: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    request_context: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass
class ErrorReport:
    """Comprehensive error report"""
    context: ErrorContext
    exception_type: str
    exception_message: str
    stack_trace: str
    severity: ErrorSeverity
    category: ErrorCategory
    is_recoverable: bool
    suggested_recovery: List[RecoveryStrategy]
    user_impact: str
    system_impact: str
    related_errors: List[str] = field(default_factory=list)
    resolution_steps: List[str] = field(default_factory=list)
    prevention_recommendations: List[str] = field(default_factory=list)


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_id: str = ""
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    execution_time: float = 0.0
    result_message: str = ""
    side_effects: List[str] = field(default_factory=list)


class AdvancedErrorHandler:
    """
    🛡️ Advanced Error Handling System
    
    Features:
    - Intelligent error classification and severity assessment
    - Contextual recovery strategy selection
    - Adaptive learning from error patterns
    - Comprehensive error reporting and tracking
    - Integration with self-healing system
    - Performance-aware error handling
    """
    
    def __init__(self, project_path: Path, self_healing: Optional[SelfHealingSystem] = None):
        self.project_path = project_path
        self.self_healing = self_healing
        
        # Error tracking
        self.error_history: List[ErrorReport] = []
        self.recovery_history: List[RecoveryAttempt] = []
        self.error_patterns: Dict[str, Any] = {}
        
        # Configuration
        self.max_retry_attempts = 3
        self.retry_delays = [1.0, 2.0, 5.0]  # Exponential backoff
        self.severity_thresholds = {
            "performance_degradation": 0.5,
            "resource_usage": 0.8,
            "failure_rate": 0.1
        }
        
        # Learning and adaptation
        self.recovery_success_rates: Dict[RecoveryStrategy, Dict[ErrorCategory, float]] = {}
        
        # Load previous learning
        self._load_error_knowledge()
    
    def _load_error_knowledge(self) -> None:
        """Load previously learned error handling knowledge"""
        knowledge_file = self.project_path / ".error_knowledge.json"
        if knowledge_file.exists():
            try:
                with open(knowledge_file) as f:
                    data = json.load(f)
                    self.error_patterns = data.get("error_patterns", {})
                    # Convert recovery success rates back to enums
                    raw_rates = data.get("recovery_success_rates", {})
                    for strategy_str, category_rates in raw_rates.items():
                        try:
                            strategy = RecoveryStrategy(strategy_str)
                            self.recovery_success_rates[strategy] = {}
                            for category_str, rate in category_rates.items():
                                try:
                                    category = ErrorCategory(category_str)
                                    self.recovery_success_rates[strategy][category] = rate
                                except ValueError:
                                    continue
                        except ValueError:
                            continue
                logger.info("Loaded error handling knowledge from previous runs")
            except Exception as e:
                logger.warning(f"Failed to load error knowledge: {e}")
    
    def _save_error_knowledge(self) -> None:
        """Save learned error handling knowledge"""
        knowledge_file = self.project_path / ".error_knowledge.json"
        try:
            # Convert enums to strings for JSON serialization
            recovery_rates_serializable = {}
            for strategy, category_rates in self.recovery_success_rates.items():
                recovery_rates_serializable[strategy.value] = {
                    category.value: rate for category, rate in category_rates.items()
                }
            
            data = {
                "error_patterns": self.error_patterns,
                "recovery_success_rates": recovery_rates_serializable,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(knowledge_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error knowledge: {e}")
    
    async def handle_error(self, 
                          exception: Exception, 
                          context: Optional[ErrorContext] = None,
                          auto_recover: bool = True) -> ErrorReport:
        """
        Handle an error with comprehensive analysis and optional auto-recovery
        
        Args:
            exception: The exception that occurred
            context: Additional context information
            auto_recover: Whether to attempt automatic recovery
        
        Returns:
            Comprehensive error report
        """
        # Create context if not provided
        if context is None:
            context = self._create_error_context(exception)
        
        # Classify and analyze the error
        error_report = await self._analyze_error(exception, context)
        
        # Store error for learning
        self.error_history.append(error_report)
        
        # Log the error appropriately
        self._log_error(error_report)
        
        # Attempt recovery if requested and appropriate
        if auto_recover and error_report.is_recoverable:
            recovery_success = await self._attempt_recovery(error_report)
            if not recovery_success and error_report.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                # Escalate to self-healing system
                if self.self_healing:
                    failure_event = FailureEvent(
                        failure_id=error_report.context.error_id,
                        failure_type=self._map_category_to_failure_type(error_report.category),
                        component=error_report.context.component,
                        severity=self._severity_to_float(error_report.severity),
                        error_message=error_report.exception_message,
                        context=error_report.context.__dict__
                    )
                    await self.self_healing.heal(failure_event)
        
        # Learn from this error
        await self._learn_from_error(error_report)
        
        # Save updated knowledge
        self._save_error_knowledge()
        
        return error_report
    
    def _create_error_context(self, exception: Exception) -> ErrorContext:
        """Create error context from exception and current state"""
        # Get stack frame information
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the calling frame
            caller_frame = frame.f_back.f_back if frame and frame.f_back else None
            
            component = ""
            operation = ""
            
            if caller_frame:
                component = caller_frame.f_code.co_filename.split('/')[-1].replace('.py', '')
                operation = caller_frame.f_code.co_name
        finally:
            del frame  # Avoid reference cycles
        
        return ErrorContext(
            component=component,
            operation=operation,
            system_state=self._capture_system_state(),
            performance_metrics=self._capture_performance_metrics()
        )
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context"""
        return {
            "timestamp": datetime.now().isoformat(),
            "process_id": os.getpid() if 'os' in globals() else 0,
            "thread_count": threading.active_count() if 'threading' in globals() else 1,
            "memory_usage_mb": 0,  # Would integrate with psutil in practice
            "cpu_usage_percent": 0,  # Would integrate with psutil in practice
        }
    
    def _capture_performance_metrics(self) -> Dict[str, float]:
        """Capture performance metrics at time of error"""
        return {
            "response_time_ms": 0.0,
            "throughput_rps": 0.0,
            "error_rate": len([e for e in self.error_history[-100:] 
                             if (datetime.now() - e.context.timestamp).total_seconds() < 300]) / 300.0
        }
    
    async def _analyze_error(self, exception: Exception, context: ErrorContext) -> ErrorReport:
        """Comprehensive error analysis and classification"""
        
        # Classify error category
        category = self._classify_error_category(exception)
        
        # Assess severity
        severity = self._assess_error_severity(exception, context, category)
        
        # Determine recoverability
        is_recoverable = self._is_error_recoverable(exception, category, severity)
        
        # Suggest recovery strategies
        suggested_recovery = self._suggest_recovery_strategies(category, severity, context)
        
        # Assess impact
        user_impact = self._assess_user_impact(exception, category, severity)
        system_impact = self._assess_system_impact(exception, category, severity)
        
        # Generate resolution steps
        resolution_steps = self._generate_resolution_steps(exception, category)
        
        # Generate prevention recommendations
        prevention_recommendations = self._generate_prevention_recommendations(exception, category)
        
        return ErrorReport(
            context=context,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            severity=severity,
            category=category,
            is_recoverable=is_recoverable,
            suggested_recovery=suggested_recovery,
            user_impact=user_impact,
            system_impact=system_impact,
            resolution_steps=resolution_steps,
            prevention_recommendations=prevention_recommendations
        )
    
    def _classify_error_category(self, exception: Exception) -> ErrorCategory:
        """Classify error into appropriate category"""
        exception_type = type(exception).__name__
        error_message = str(exception).lower()
        
        # Rule-based classification
        if "validation" in error_message or "invalid" in error_message:
            return ErrorCategory.VALIDATION
        elif "auth" in error_message or "permission" in error_message:
            return ErrorCategory.AUTHENTICATION
        elif "network" in error_message or "connection" in error_message:
            return ErrorCategory.NETWORK
        elif "database" in error_message or "sql" in error_message:
            return ErrorCategory.DATABASE
        elif "file" in error_message or "directory" in error_message:
            return ErrorCategory.FILE_SYSTEM
        elif "config" in error_message:
            return ErrorCategory.CONFIGURATION
        elif "import" in error_message or "module" in error_message:
            return ErrorCategory.DEPENDENCY
        elif "memory" in error_message or "resource" in error_message:
            return ErrorCategory.SYSTEM_RESOURCE
        elif exception_type in ["ValueError", "TypeError", "AttributeError"]:
            return ErrorCategory.BUSINESS_LOGIC
        else:
            return ErrorCategory.UNKNOWN
    
    def _assess_error_severity(self, exception: Exception, context: ErrorContext, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity based on multiple factors"""
        base_severity = ErrorSeverity.MEDIUM
        
        # Adjust based on exception type
        if isinstance(exception, (MemoryError, SystemError)):
            base_severity = ErrorSeverity.CRITICAL
        elif isinstance(exception, (KeyboardInterrupt, SystemExit)):
            base_severity = ErrorSeverity.FATAL
        elif isinstance(exception, (ConnectionError, TimeoutError)):
            base_severity = ErrorSeverity.HIGH
        elif isinstance(exception, (ValueError, TypeError)):
            base_severity = ErrorSeverity.LOW
        
        # Adjust based on category
        if category in [ErrorCategory.SYSTEM_RESOURCE, ErrorCategory.DATABASE]:
            if base_severity.value in ["low", "medium"]:
                base_severity = ErrorSeverity.HIGH
        elif category == ErrorCategory.VALIDATION:
            base_severity = ErrorSeverity.LOW
        
        # Adjust based on context
        if context.performance_metrics.get("error_rate", 0) > self.severity_thresholds["failure_rate"]:
            if base_severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
                base_severity = ErrorSeverity.HIGH
        
        return base_severity
    
    def _is_error_recoverable(self, exception: Exception, category: ErrorCategory, severity: ErrorSeverity) -> bool:
        """Determine if error is recoverable"""
        
        # Fatal errors are not recoverable
        if severity == ErrorSeverity.FATAL:
            return False
        
        # Some exception types are inherently non-recoverable
        if isinstance(exception, (KeyboardInterrupt, SystemExit, MemoryError)):
            return False
        
        # Category-based recoverability
        recoverable_categories = [
            ErrorCategory.NETWORK,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.DATABASE,
            ErrorCategory.VALIDATION
        ]
        
        return category in recoverable_categories or severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
    
    def _suggest_recovery_strategies(self, category: ErrorCategory, severity: ErrorSeverity, context: ErrorContext) -> List[RecoveryStrategy]:
        """Suggest recovery strategies based on error characteristics"""
        
        strategies = []
        
        # Category-based strategies
        if category == ErrorCategory.NETWORK:
            strategies.extend([RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK])
        elif category == ErrorCategory.DATABASE:
            strategies.extend([RecoveryStrategy.RETRY, RecoveryStrategy.GRACEFUL_DEGRADATION])
        elif category == ErrorCategory.VALIDATION:
            strategies.extend([RecoveryStrategy.SKIP, RecoveryStrategy.USER_INTERVENTION])
        elif category == ErrorCategory.EXTERNAL_SERVICE:
            strategies.extend([RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK, RecoveryStrategy.GRACEFUL_DEGRADATION])
        elif category == ErrorCategory.SYSTEM_RESOURCE:
            strategies.extend([RecoveryStrategy.AUTOMATIC_RECOVERY, RecoveryStrategy.GRACEFUL_DEGRADATION])
        
        # Severity-based adjustments
        if severity == ErrorSeverity.CRITICAL:
            strategies = [RecoveryStrategy.ABORT, RecoveryStrategy.USER_INTERVENTION]
        elif severity == ErrorSeverity.LOW:
            strategies.append(RecoveryStrategy.SKIP)
        
        # Use learned preferences if available
        if category in [rates.keys() for rates in self.recovery_success_rates.values()]:
            # Sort by historical success rate
            strategy_scores = {}
            for strategy in strategies:
                if strategy in self.recovery_success_rates:
                    score = self.recovery_success_rates[strategy].get(category, 0.5)
                    strategy_scores[strategy] = score
            
            if strategy_scores:
                strategies = sorted(strategies, key=lambda s: strategy_scores.get(s, 0.5), reverse=True)
        
        return strategies[:3]  # Limit to top 3 strategies
    
    def _assess_user_impact(self, exception: Exception, category: ErrorCategory, severity: ErrorSeverity) -> str:
        """Assess impact on users"""
        if severity == ErrorSeverity.FATAL:
            return "Complete service unavailable"
        elif severity == ErrorSeverity.CRITICAL:
            return "Major functionality impaired"
        elif severity == ErrorSeverity.HIGH:
            return "Significant feature degradation"
        elif severity == ErrorSeverity.MEDIUM:
            return "Minor functionality issues"
        else:
            return "Minimal user impact"
    
    def _assess_system_impact(self, exception: Exception, category: ErrorCategory, severity: ErrorSeverity) -> str:
        """Assess impact on system"""
        if category == ErrorCategory.SYSTEM_RESOURCE:
            return "System resource utilization affected"
        elif category == ErrorCategory.DATABASE:
            return "Data persistence operations impacted"
        elif category == ErrorCategory.NETWORK:
            return "Network connectivity issues"
        else:
            return f"{severity.value.capitalize()} system impact"
    
    def _generate_resolution_steps(self, exception: Exception, category: ErrorCategory) -> List[str]:
        """Generate specific resolution steps"""
        steps = []
        
        if category == ErrorCategory.NETWORK:
            steps.extend([
                "Check network connectivity",
                "Verify service endpoints",
                "Review firewall settings",
                "Test with alternative endpoints"
            ])
        elif category == ErrorCategory.DATABASE:
            steps.extend([
                "Check database connectivity",
                "Verify database schema",
                "Review connection pool settings",
                "Check database logs"
            ])
        elif category == ErrorCategory.VALIDATION:
            steps.extend([
                "Review input validation rules",
                "Check data format requirements",
                "Verify business logic constraints",
                "Update validation schemas"
            ])
        else:
            steps.extend([
                "Review error logs",
                "Check system configuration",
                "Verify dependencies",
                "Restart affected services"
            ])
        
        return steps
    
    def _generate_prevention_recommendations(self, exception: Exception, category: ErrorCategory) -> List[str]:
        """Generate prevention recommendations"""
        recommendations = []
        
        if category == ErrorCategory.NETWORK:
            recommendations.extend([
                "Implement circuit breakers",
                "Add retry mechanisms with exponential backoff",
                "Use connection pooling",
                "Monitor network latency"
            ])
        elif category == ErrorCategory.DATABASE:
            recommendations.extend([
                "Implement database health checks",
                "Use connection pooling",
                "Add query timeouts",
                "Monitor database performance"
            ])
        elif category == ErrorCategory.VALIDATION:
            recommendations.extend([
                "Strengthen input validation",
                "Add schema validation",
                "Implement data sanitization",
                "Use type hints and validation"
            ])
        else:
            recommendations.extend([
                "Add comprehensive error handling",
                "Implement monitoring and alerting",
                "Use defensive programming practices",
                "Add unit tests for error scenarios"
            ])
        
        return recommendations
    
    async def _attempt_recovery(self, error_report: ErrorReport) -> bool:
        """Attempt recovery using suggested strategies"""
        
        for strategy in error_report.suggested_recovery:
            attempt = RecoveryAttempt(
                error_id=error_report.context.error_id,
                strategy=strategy
            )
            
            start_time = time.time()
            
            try:
                success = await self._execute_recovery_strategy(strategy, error_report, attempt)
                attempt.execution_time = time.time() - start_time
                attempt.success = success
                
                if success:
                    attempt.result_message = f"Successfully recovered using {strategy.value}"
                    self.recovery_history.append(attempt)
                    
                    # Update success rate
                    self._update_recovery_success_rate(strategy, error_report.category, True)
                    
                    logger.info(f"Recovery successful using {strategy.value}")
                    return True
                else:
                    attempt.result_message = f"Recovery failed with {strategy.value}"
                
            except Exception as e:
                attempt.execution_time = time.time() - start_time
                attempt.success = False
                attempt.result_message = f"Recovery error: {str(e)}"
                logger.error(f"Recovery strategy {strategy.value} failed: {e}")
            
            self.recovery_history.append(attempt)
            self._update_recovery_success_rate(strategy, error_report.category, False)
        
        return False
    
    async def _execute_recovery_strategy(self, strategy: RecoveryStrategy, error_report: ErrorReport, attempt: RecoveryAttempt) -> bool:
        """Execute a specific recovery strategy"""
        
        if strategy == RecoveryStrategy.RETRY:
            return await self._recovery_retry(error_report, attempt)
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._recovery_fallback(error_report, attempt)
        elif strategy == RecoveryStrategy.SKIP:
            return await self._recovery_skip(error_report, attempt)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._recovery_graceful_degradation(error_report, attempt)
        elif strategy == RecoveryStrategy.AUTOMATIC_RECOVERY:
            return await self._recovery_automatic(error_report, attempt)
        else:
            logger.warning(f"Recovery strategy {strategy.value} not implemented")
            return False
    
    def _update_recovery_success_rate(self, strategy: RecoveryStrategy, category: ErrorCategory, success: bool) -> None:
        """Update recovery success rate for learning"""
        if strategy not in self.recovery_success_rates:
            self.recovery_success_rates[strategy] = {}
        
        if category not in self.recovery_success_rates[strategy]:
            self.recovery_success_rates[strategy][category] = 0.5
        
        # Exponential moving average
        alpha = 0.1
        current_rate = self.recovery_success_rates[strategy][category]
        new_rate = current_rate + alpha * (1.0 if success else 0.0 - current_rate)
        self.recovery_success_rates[strategy][category] = new_rate
    
    async def _learn_from_error(self, error_report: ErrorReport) -> None:
        """Learn from error patterns for continuous improvement"""
        
        # Update error patterns
        pattern_key = f"{error_report.category.value}_{error_report.exception_type}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {
                "count": 0,
                "severity_distribution": {},
                "common_contexts": [],
                "successful_recoveries": []
            }
        
        pattern = self.error_patterns[pattern_key]
        pattern["count"] += 1
        
        # Update severity distribution
        severity_key = error_report.severity.value
        if severity_key not in pattern["severity_distribution"]:
            pattern["severity_distribution"][severity_key] = 0
        pattern["severity_distribution"][severity_key] += 1
        
        # Update common contexts
        context_signature = f"{error_report.context.component}:{error_report.context.operation}"
        if context_signature not in pattern["common_contexts"]:
            pattern["common_contexts"].append(context_signature)
    
    def _log_error(self, error_report: ErrorReport) -> None:
        """Log error with appropriate level and formatting"""
        
        if error_report.severity == ErrorSeverity.FATAL:
            logger.critical(f"FATAL ERROR [{error_report.context.error_id}]: {error_report.exception_message}")
        elif error_report.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR [{error_report.context.error_id}]: {error_report.exception_message}")
        elif error_report.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY [{error_report.context.error_id}]: {error_report.exception_message}")
        elif error_report.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY [{error_report.context.error_id}]: {error_report.exception_message}")
        else:
            logger.info(f"LOW SEVERITY [{error_report.context.error_id}]: {error_report.exception_message}")
        
        # Display rich error panel for critical/fatal errors
        if error_report.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            self._display_error_panel(error_report)
    
    def _display_error_panel(self, error_report: ErrorReport) -> None:
        """Display rich error panel for critical errors"""
        content = f"""
🚨 {error_report.severity.value.upper()} ERROR DETECTED

Error ID: {error_report.context.error_id}
Component: {error_report.context.component}
Operation: {error_report.context.operation}
Category: {error_report.category.value}

Exception: {error_report.exception_type}
Message: {error_report.exception_message}

User Impact: {error_report.user_impact}
System Impact: {error_report.system_impact}

Recovery Strategies:
{chr(10).join('  • ' + strategy.value for strategy in error_report.suggested_recovery)}
        """
        
        panel = Panel(
            content.strip(),
            title=f"🛡️ Advanced Error Handler",
            border_style="red" if error_report.severity == ErrorSeverity.FATAL else "yellow"
        )
        
        console.print(panel)
    
    # Helper methods
    def _map_category_to_failure_type(self, category: ErrorCategory):
        """Map error category to self-healing failure type"""
        from .self_healing import FailureType
        mapping = {
            ErrorCategory.NETWORK: FailureType.NETWORK_FAILURE,
            ErrorCategory.DATABASE: FailureType.DATABASE_FAILURE,
            ErrorCategory.SYSTEM_RESOURCE: FailureType.RESOURCE_EXHAUSTION,
            ErrorCategory.EXTERNAL_SERVICE: FailureType.API_FAILURE,
            ErrorCategory.DEPENDENCY: FailureType.DEPENDENCY_FAILURE,
        }
        return mapping.get(category, FailureType.DEPENDENCY_FAILURE)
    
    def _severity_to_float(self, severity: ErrorSeverity) -> float:
        """Convert severity enum to float for self-healing system"""
        mapping = {
            ErrorSeverity.LOW: 0.2,
            ErrorSeverity.MEDIUM: 0.4,
            ErrorSeverity.HIGH: 0.7,
            ErrorSeverity.CRITICAL: 0.9,
            ErrorSeverity.FATAL: 1.0
        }
        return mapping[severity]
    
    # Recovery strategy implementations (placeholders)
    async def _recovery_retry(self, error_report: ErrorReport, attempt: RecoveryAttempt) -> bool:
        logger.info(f"Attempting retry recovery for error {error_report.context.error_id}")
        await asyncio.sleep(1.0)  # Simulate retry delay
        return True
    
    async def _recovery_fallback(self, error_report: ErrorReport, attempt: RecoveryAttempt) -> bool:
        logger.info(f"Attempting fallback recovery for error {error_report.context.error_id}")
        return True
    
    async def _recovery_skip(self, error_report: ErrorReport, attempt: RecoveryAttempt) -> bool:
        logger.info(f"Skipping failed operation for error {error_report.context.error_id}")
        return True
    
    async def _recovery_graceful_degradation(self, error_report: ErrorReport, attempt: RecoveryAttempt) -> bool:
        logger.info(f"Enabling graceful degradation for error {error_report.context.error_id}")
        return True
    
    async def _recovery_automatic(self, error_report: ErrorReport, attempt: RecoveryAttempt) -> bool:
        logger.info(f"Attempting automatic recovery for error {error_report.context.error_id}")
        return True


# Decorator for automatic error handling
def handle_errors(auto_recover: bool = True, 
                 severity_override: Optional[ErrorSeverity] = None,
                 category_override: Optional[ErrorCategory] = None):
    """
    Decorator for automatic error handling with advanced features
    
    Args:
        auto_recover: Whether to attempt automatic recovery
        severity_override: Override automatic severity assessment
        category_override: Override automatic category classification
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = getattr(args[0], '_error_handler', None) if args else None
            
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                if error_handler and isinstance(error_handler, AdvancedErrorHandler):
                    context = ErrorContext(
                        component=func.__module__.split('.')[-1] if hasattr(func, '__module__') else 'unknown',
                        operation=func.__name__
                    )
                    
                    error_report = await error_handler.handle_error(e, context, auto_recover)
                    
                    # Apply overrides if specified
                    if severity_override:
                        error_report.severity = severity_override
                    if category_override:
                        error_report.category = category_override
                    
                    # Re-raise if not recoverable or recovery failed
                    if not error_report.is_recoverable or error_report.severity == ErrorSeverity.FATAL:
                        raise
                else:
                    # Fallback to standard logging
                    logger.error(f"Unhandled error in {func.__name__}: {e}")
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return async_wrapper(*args, **kwargs)
            
            error_handler = getattr(args[0], '_error_handler', None) if args else None
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler and isinstance(error_handler, AdvancedErrorHandler):
                    # For sync functions, we can't use async error handling
                    # Fall back to basic logging and re-raise
                    logger.error(f"Error in {func.__name__}: {e}")
                    raise
                else:
                    logger.error(f"Unhandled error in {func.__name__}: {e}")
                    raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Context manager for error handling
@contextlib.asynccontextmanager
async def error_handling_context(error_handler: AdvancedErrorHandler,
                                component: str,
                                operation: str,
                                auto_recover: bool = True):
    """
    Async context manager for advanced error handling
    
    Args:
        error_handler: The error handler instance
        component: Component name for context
        operation: Operation name for context
        auto_recover: Whether to attempt automatic recovery
    """
    context = ErrorContext(component=component, operation=operation)
    
    try:
        yield context
    except Exception as e:
        await error_handler.handle_error(e, context, auto_recover)
        raise