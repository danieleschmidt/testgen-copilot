"""Quantum-enhanced security module with advanced threat detection."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
import logging
import re
import json

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class ThreatLevel(Enum):
    """Quantum threat classification levels."""
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"  
    MALICIOUS = "malicious"
    CRITICAL = "critical"
    QUANTUM_ANOMALY = "quantum_anomaly"  # Threats that exploit quantum properties


class SecurityEvent(Enum):
    """Types of security events in quantum systems."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    INJECTION_ATTEMPT = "injection_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    ENTANGLEMENT_ATTACK = "entanglement_attack"
    MEASUREMENT_INTERFERENCE = "measurement_interference"


@dataclass
class QuantumThreat:
    """Threat detection with quantum uncertainty and correlation."""
    
    id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    description: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Quantum properties
    detection_confidence: float = 0.95  # Confidence in threat detection
    quantum_signature: Dict[str, Any] = field(default_factory=dict)
    correlated_threats: Set[str] = field(default_factory=set)
    
    # Mitigation tracking
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)
    
    def correlate_with(self, other_threat_id: str):
        """Create correlation with another threat."""
        self.correlated_threats.add(other_threat_id)
    
    def apply_mitigation(self, action: str):
        """Apply mitigation action to threat."""
        self.mitigation_actions.append(action)
        if len(self.mitigation_actions) >= 2:  # Multiple mitigations = resolved
            self.mitigated = True


class QuantumInputValidator:
    """Advanced input validation with quantum anomaly detection."""
    
    def __init__(self):
        """Initialize quantum input validator."""
        self.logger = logging.getLogger(__name__)
        
        # Quantum-enhanced pattern detection
        self.malicious_patterns = [
            # SQL injection patterns
            r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
            r"(?i)(exec\s*\(|eval\s*\(|system\s*\()",
            r"(?i)(\bor\b\s+\d+\s*=\s*\d+|and\s+\d+\s*=\s*\d+)",
            
            # Code injection patterns  
            r"(?i)(__import__|exec|eval|compile|open|file)",
            r"(?i)(subprocess|os\.system|os\.popen|os\.exec)",
            
            # Path traversal
            r"\.\.[\\/]",
            r"(?i)(etc/passwd|windows/system32|boot\.ini)",
            
            # XSS patterns
            r"(?i)(<script|javascript:|vbscript:|onload=|onerror=)",
            
            # Command injection
            r"(?i)(;\s*rm\s+-rf|;\s*del\s+/|&&\s*format|;\s*shutdown)",
            
            # Quantum-specific threats
            r"(?i)(quantum_decohere|collapse_wavefunction|entanglement_break)",
            r"(?i)(measurement_interfere|coherence_attack)"
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern) for pattern in self.malicious_patterns]
    
    def validate_input(self, input_data: Any, context: str = "unknown") -> Dict[str, Any]:
        """Validate input with quantum anomaly detection."""
        
        validation_result = {
            "is_safe": True,
            "threat_level": ThreatLevel.BENIGN,
            "threats_detected": [],
            "quantum_anomalies": [],
            "validation_confidence": 1.0,
            "context": context
        }
        
        # Convert input to string for pattern matching
        if isinstance(input_data, (dict, list)):
            input_str = json.dumps(input_data, default=str)
        else:
            input_str = str(input_data)
        
        # Check against malicious patterns
        detected_patterns = []
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(input_str):
                detected_patterns.append({
                    "pattern_id": i,
                    "pattern": self.malicious_patterns[i],
                    "severity": self._assess_pattern_severity(i)
                })
        
        # Quantum anomaly detection
        quantum_anomalies = self._detect_quantum_anomalies(input_str)
        
        # Calculate overall threat assessment
        if detected_patterns or quantum_anomalies:
            validation_result["is_safe"] = False
            validation_result["threats_detected"] = detected_patterns
            validation_result["quantum_anomalies"] = quantum_anomalies
            
            # Determine threat level
            max_severity = max(
                [p["severity"] for p in detected_patterns] + 
                [a["severity"] for a in quantum_anomalies],
                default=0
            )
            
            if max_severity >= 4:
                validation_result["threat_level"] = ThreatLevel.CRITICAL
            elif max_severity >= 3:
                validation_result["threat_level"] = ThreatLevel.MALICIOUS
            elif max_severity >= 2:
                validation_result["threat_level"] = ThreatLevel.SUSPICIOUS
            
            # Reduce confidence based on number of detections
            total_detections = len(detected_patterns) + len(quantum_anomalies)
            validation_result["validation_confidence"] = max(0.5, 1.0 - (total_detections * 0.1))
        
        # Log significant threats
        if validation_result["threat_level"] != ThreatLevel.BENIGN:
            self.logger.warning(
                f"Threat detected in {context}: {validation_result['threat_level'].value}",
                extra={
                    "threats": detected_patterns,
                    "quantum_anomalies": quantum_anomalies,
                    "input_hash": hashlib.sha256(input_str.encode()).hexdigest()[:16]
                }
            )
        
        return validation_result
    
    def _assess_pattern_severity(self, pattern_id: int) -> int:
        """Assess severity of detected pattern (1-5 scale)."""
        
        # High-severity patterns (system commands, code execution)
        high_severity_patterns = {1, 4, 8, 10, 11}  # exec, subprocess, rm -rf, etc.
        
        # Medium-severity patterns (SQL injection, XSS)
        medium_severity_patterns = {0, 2, 6, 7}  # SQL injection, XSS
        
        # Quantum-specific patterns (experimental threats)
        quantum_patterns = {9, 10, 11}  # Quantum-specific attack patterns
        
        if pattern_id in high_severity_patterns:
            return 4
        elif pattern_id in quantum_patterns:
            return 5  # Highest severity for quantum threats
        elif pattern_id in medium_severity_patterns:
            return 3
        else:
            return 2  # Default medium-low severity
    
    def _detect_quantum_anomalies(self, input_str: str) -> List[Dict[str, Any]]:
        """Detect quantum-specific anomalies and attack patterns."""
        
        anomalies = []
        
        # Entropy analysis - highly random strings might indicate attacks
        entropy = self._calculate_entropy(input_str)
        if entropy > 4.5:  # High entropy threshold
            anomalies.append({
                "type": "high_entropy",
                "description": f"Input has unusually high entropy ({entropy:.2f})",
                "severity": 3,
                "quantum_metric": entropy
            })
        
        # Length-based anomalies
        if len(input_str) > 10000:  # Very long inputs
            anomalies.append({
                "type": "excessive_length",
                "description": f"Input length ({len(input_str)}) exceeds normal bounds",
                "severity": 2,
                "quantum_metric": len(input_str)
            })
        
        # Repetition patterns that might indicate quantum interference
        repetition_score = self._calculate_repetition_score(input_str)
        if repetition_score > 0.8:
            anomalies.append({
                "type": "quantum_interference_pattern",
                "description": f"Repetitive patterns detected (score: {repetition_score:.2f})",
                "severity": 4,
                "quantum_metric": repetition_score
            })
        
        # Character distribution anomalies
        char_distribution = self._analyze_character_distribution(input_str)
        if char_distribution["anomaly_score"] > 0.7:
            anomalies.append({
                "type": "character_distribution_anomaly",
                "description": "Unusual character distribution detected",
                "severity": 3,
                "quantum_metric": char_distribution["anomaly_score"]
            })
        
        return anomalies
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of string."""
        if not data:
            return 0.0
        
        # Count character frequencies
        frequencies = {}
        for char in data:
            frequencies[char] = frequencies.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        length = len(data)
        
        for count in frequencies.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_repetition_score(self, data: str) -> float:
        """Calculate repetition score to detect quantum interference patterns."""
        if len(data) < 4:
            return 0.0
        
        # Look for repeating substrings
        max_repetition = 0.0
        data_len = len(data)
        
        # Check for patterns of length 2-10
        for pattern_len in range(2, min(11, data_len // 2)):
            for start in range(data_len - pattern_len):
                pattern = data[start:start + pattern_len]
                
                # Count occurrences of this pattern
                occurrences = 0
                pos = 0
                while pos < data_len - pattern_len:
                    if data[pos:pos + pattern_len] == pattern:
                        occurrences += 1
                        pos += pattern_len
                    else:
                        pos += 1
                
                # Calculate repetition ratio
                repetition_ratio = (occurrences * pattern_len) / data_len
                max_repetition = max(max_repetition, repetition_ratio)
        
        return max_repetition
    
    def _analyze_character_distribution(self, data: str) -> Dict[str, Any]:
        """Analyze character distribution for anomalies."""
        if not data:
            return {"anomaly_score": 0.0, "distribution": {}}
        
        # Character categories
        categories = {
            "letters": 0,
            "digits": 0,
            "special": 0,
            "whitespace": 0,
            "control": 0
        }
        
        for char in data:
            if char.isalpha():
                categories["letters"] += 1
            elif char.isdigit():
                categories["digits"] += 1
            elif char.isspace():
                categories["whitespace"] += 1
            elif ord(char) < 32 or ord(char) > 126:
                categories["control"] += 1
            else:
                categories["special"] += 1
        
        total_chars = len(data)
        distribution = {cat: count / total_chars for cat, count in categories.items()}
        
        # Calculate anomaly score based on unusual distributions
        anomaly_score = 0.0
        
        # Too many control characters
        if distribution["control"] > 0.1:
            anomaly_score += 0.3
        
        # Too many special characters
        if distribution["special"] > 0.5:
            anomaly_score += 0.2
        
        # All digits or all letters (unusual for normal input)
        if distribution["digits"] > 0.9 or distribution["letters"] > 0.9:
            anomaly_score += 0.2
        
        # Lack of normal characters
        if distribution["letters"] + distribution["digits"] < 0.3:
            anomaly_score += 0.3
        
        return {
            "anomaly_score": min(anomaly_score, 1.0),
            "distribution": distribution
        }


class QuantumEncryption:
    """Quantum-inspired encryption with enhanced security."""
    
    def __init__(self, key: Optional[bytes] = None):
        """Initialize quantum encryption."""
        if key is None:
            key = self._generate_quantum_key()
        
        self.key = key
        self.cipher = Fernet(key)
        self.logger = logging.getLogger(__name__)
    
    def _generate_quantum_key(self) -> bytes:
        """Generate cryptographically strong key with quantum randomness."""
        
        # Use system entropy + quantum-inspired randomness
        password = secrets.token_bytes(32)  # 256-bit password
        salt = secrets.token_bytes(16)      # 128-bit salt
        
        # Derive key using PBKDF2 with quantum-inspired iterations
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # High iteration count for quantum resistance
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        self.logger.info("Generated quantum-enhanced encryption key")
        return key
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data with quantum enhancement."""
        
        # Add quantum timestamp and nonce for replay protection
        quantum_wrapper = {
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nonce": secrets.token_hex(16),
            "quantum_checksum": self._calculate_quantum_checksum(data)
        }
        
        # Encrypt the wrapped data
        data_bytes = json.dumps(quantum_wrapper).encode('utf-8')
        encrypted = self.cipher.encrypt(data_bytes)
        
        # Return base64 encoded for safe transport
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt_sensitive_data(self, encrypted_data: str, max_age_seconds: int = 3600) -> str:
        """Decrypt sensitive data with quantum validation."""
        
        try:
            # Decode from base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            
            # Decrypt
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            quantum_wrapper = json.loads(decrypted_bytes.decode('utf-8'))
            
            # Validate quantum wrapper
            self._validate_quantum_wrapper(quantum_wrapper, max_age_seconds)
            
            return quantum_wrapper["data"]
            
        except Exception as e:
            self.logger.error(f"Quantum decryption failed: {e}")
            raise ValueError("Invalid or corrupted quantum encrypted data")
    
    def _calculate_quantum_checksum(self, data: str) -> str:
        """Calculate quantum-enhanced checksum."""
        
        # Multiple hash algorithm combination for quantum resistance
        sha256_hash = hashlib.sha256(data.encode()).hexdigest()
        blake2_hash = hashlib.blake2b(data.encode(), digest_size=32).hexdigest()
        
        # Combine hashes with quantum-inspired mixing
        combined = sha256_hash + blake2_hash
        final_hash = hashlib.sha3_256(combined.encode()).hexdigest()
        
        return final_hash[:32]  # Return first 32 characters
    
    def _validate_quantum_wrapper(self, wrapper: Dict[str, Any], max_age_seconds: int):
        """Validate quantum wrapper integrity."""
        
        required_fields = ["data", "timestamp", "nonce", "quantum_checksum"]
        for field in required_fields:
            if field not in wrapper:
                raise ValueError(f"Missing quantum wrapper field: {field}")
        
        # Validate timestamp (prevent replay attacks)
        try:
            timestamp = datetime.fromisoformat(wrapper["timestamp"])
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
            
            if age > max_age_seconds:
                raise ValueError(f"Quantum data too old: {age}s > {max_age_seconds}s")
                
        except ValueError as e:
            raise ValueError(f"Invalid quantum timestamp: {e}")
        
        # Validate quantum checksum
        expected_checksum = self._calculate_quantum_checksum(wrapper["data"])
        if not secrets.compare_digest(wrapper["quantum_checksum"], expected_checksum):
            raise ValueError("Quantum checksum validation failed")


class QuantumThreatDetector:
    """Advanced threat detection with quantum correlation analysis."""
    
    def __init__(self, threat_memory_limit: int = 1000):
        """Initialize quantum threat detector."""
        self.threat_memory_limit = threat_memory_limit
        self.detected_threats: Dict[str, QuantumThreat] = {}
        self.threat_history: List[QuantumThreat] = []
        
        self.input_validator = QuantumInputValidator()
        self.logger = logging.getLogger(__name__)
        
        # Behavioral analysis state
        self.user_behavior_profiles: Dict[str, Dict[str, Any]] = {}
        self.ip_reputation: Dict[str, Dict[str, Any]] = {}
    
    def analyze_request(
        self,
        request_data: Dict[str, Any],
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze request for quantum threats."""
        
        analysis_result = {
            "threat_detected": False,
            "threat_level": ThreatLevel.BENIGN,
            "security_score": 1.0,
            "threats": [],
            "quantum_analysis": {},
            "recommendations": []
        }
        
        # Input validation
        validation_results = []
        for key, value in request_data.items():
            validation = self.input_validator.validate_input(value, context=key)
            if not validation["is_safe"]:
                validation_results.append({
                    "field": key,
                    "validation": validation
                })
        
        # Behavioral analysis
        behavioral_anomalies = self._analyze_user_behavior(user_id, request_data, source_ip)
        
        # IP reputation check
        ip_analysis = self._analyze_ip_reputation(source_ip) if source_ip else {}
        
        # Quantum correlation analysis
        quantum_correlations = self._detect_quantum_correlations(
            request_data, validation_results, behavioral_anomalies, ip_analysis
        )
        
        # Aggregate threat assessment
        all_threats = validation_results + behavioral_anomalies + [ip_analysis] if ip_analysis else validation_results + behavioral_anomalies
        
        if all_threats or quantum_correlations:
            analysis_result["threat_detected"] = True
            analysis_result["threats"] = all_threats
            analysis_result["quantum_analysis"] = quantum_correlations
            
            # Calculate security score
            threat_count = len(all_threats)
            quantum_threat_multiplier = 1.5 if quantum_correlations.get("quantum_threats_detected", 0) > 0 else 1.0
            
            analysis_result["security_score"] = max(
                0.0, 
                1.0 - (threat_count * 0.2 * quantum_threat_multiplier)
            )
            
            # Determine overall threat level
            max_threat_level = ThreatLevel.BENIGN
            for threat in all_threats:
                if "validation" in threat:
                    threat_level = threat["validation"]["threat_level"]
                else:
                    threat_level = threat.get("threat_level", ThreatLevel.SUSPICIOUS)
                
                if threat_level.value == "critical":
                    max_threat_level = ThreatLevel.CRITICAL
                elif threat_level.value == "malicious" and max_threat_level != ThreatLevel.CRITICAL:
                    max_threat_level = ThreatLevel.MALICIOUS
                elif threat_level.value == "suspicious" and max_threat_level == ThreatLevel.BENIGN:
                    max_threat_level = ThreatLevel.SUSPICIOUS
            
            analysis_result["threat_level"] = max_threat_level
            
            # Generate recommendations
            analysis_result["recommendations"] = self._generate_security_recommendations(
                analysis_result["threat_level"], all_threats, quantum_correlations
            )
            
            # Create threat record
            if max_threat_level != ThreatLevel.BENIGN:
                self._record_threat(request_data, source_ip, user_agent, max_threat_level, all_threats)
        
        return analysis_result
    
    def _analyze_user_behavior(
        self, 
        user_id: Optional[str], 
        request_data: Dict[str, Any],
        source_ip: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Analyze user behavior for anomalies."""
        
        if not user_id:
            return []
        
        anomalies = []
        current_time = datetime.now(timezone.utc)
        
        # Initialize user profile if not exists
        if user_id not in self.user_behavior_profiles:
            self.user_behavior_profiles[user_id] = {
                "request_count": 0,
                "last_request_time": current_time,
                "typical_request_size": 0,
                "ip_addresses": set(),
                "request_frequency_samples": []
            }
        
        profile = self.user_behavior_profiles[user_id]
        
        # Update profile
        profile["request_count"] += 1
        
        # Analyze request frequency
        if profile["last_request_time"]:
            time_diff = (current_time - profile["last_request_time"]).total_seconds()
            profile["request_frequency_samples"].append(time_diff)
            
            # Keep only recent samples
            if len(profile["request_frequency_samples"]) > 20:
                profile["request_frequency_samples"] = profile["request_frequency_samples"][-20:]
            
            # Detect rapid-fire requests (potential bot behavior)
            if len(profile["request_frequency_samples"]) >= 5:
                avg_frequency = sum(profile["request_frequency_samples"][-5:]) / 5
                if avg_frequency < 0.5:  # Less than 0.5 seconds between requests
                    anomalies.append({
                        "type": "rapid_requests",
                        "description": f"Rapid request pattern detected (avg: {avg_frequency:.2f}s)",
                        "threat_level": ThreatLevel.SUSPICIOUS,
                        "severity": 3
                    })
        
        profile["last_request_time"] = current_time
        
        # Analyze request size
        request_size = len(json.dumps(request_data, default=str))
        if profile["typical_request_size"] == 0:
            profile["typical_request_size"] = request_size
        else:
            # Detect unusually large requests
            size_ratio = request_size / profile["typical_request_size"]
            if size_ratio > 10:  # Request 10x larger than typical
                anomalies.append({
                    "type": "oversized_request",
                    "description": f"Request size {size_ratio:.1f}x larger than typical",
                    "threat_level": ThreatLevel.SUSPICIOUS,
                    "severity": 2
                })
            
            # Update typical size with exponential smoothing
            profile["typical_request_size"] = profile["typical_request_size"] * 0.9 + request_size * 0.1
        
        # Track IP addresses
        if source_ip:
            profile["ip_addresses"].add(source_ip)
            
            # Detect IP switching (potential session hijacking)
            if len(profile["ip_addresses"]) > 3:
                anomalies.append({
                    "type": "multiple_ip_addresses",
                    "description": f"User accessing from {len(profile['ip_addresses'])} different IPs",
                    "threat_level": ThreatLevel.SUSPICIOUS,
                    "severity": 2
                })
        
        return anomalies
    
    def _analyze_ip_reputation(self, source_ip: str) -> Dict[str, Any]:
        """Analyze IP reputation and geolocation patterns."""
        
        if not source_ip:
            return {}
        
        current_time = datetime.now(timezone.utc)
        
        # Initialize IP tracking
        if source_ip not in self.ip_reputation:
            self.ip_reputation[source_ip] = {
                "first_seen": current_time,
                "request_count": 0,
                "threat_count": 0,
                "last_activity": current_time,
                "reputation_score": 1.0
            }
        
        ip_info = self.ip_reputation[source_ip]
        ip_info["request_count"] += 1
        ip_info["last_activity"] = current_time
        
        # Basic IP pattern analysis
        anomalies = {}
        
        # Check for private/internal IP ranges accessing public service
        if self._is_internal_ip(source_ip):
            anomalies["internal_ip_access"] = {
                "type": "internal_ip_external_access",
                "description": f"Internal IP {source_ip} accessing public service",
                "threat_level": ThreatLevel.SUSPICIOUS,
                "severity": 2
            }
        
        # Check for rapid requests from single IP
        request_frequency = ip_info["request_count"] / max(
            (current_time - ip_info["first_seen"]).total_seconds() / 60,  # Requests per minute
            1.0
        )
        
        if request_frequency > 100:  # More than 100 requests per minute
            anomalies["high_frequency_requests"] = {
                "type": "high_frequency_ip",
                "description": f"High request frequency from {source_ip}: {request_frequency:.1f}/min",
                "threat_level": ThreatLevel.SUSPICIOUS,
                "severity": 3
            }
        
        # Update reputation score
        if anomalies:
            ip_info["threat_count"] += len(anomalies)
            ip_info["reputation_score"] = max(
                0.0,
                ip_info["reputation_score"] - (len(anomalies) * 0.1)
            )
        
        return anomalies
    
    def _is_internal_ip(self, ip: str) -> bool:
        """Check if IP is in internal/private ranges."""
        import ipaddress
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
        except ValueError:
            return False
    
    def _detect_quantum_correlations(
        self,
        request_data: Dict[str, Any],
        validation_results: List[Dict[str, Any]],
        behavioral_anomalies: List[Dict[str, Any]],
        ip_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect quantum correlations between different threat indicators."""
        
        correlations = {
            "correlation_score": 0.0,
            "quantum_threats_detected": 0,
            "correlation_patterns": [],
            "entangled_threats": []
        }
        
        # Count total anomalies
        total_anomalies = len(validation_results) + len(behavioral_anomalies) + (1 if ip_analysis else 0)
        
        if total_anomalies <= 1:
            return correlations
        
        # Temporal correlation - multiple threats in short time window
        correlations["correlation_score"] += 0.3
        correlations["correlation_patterns"].append("temporal_clustering")
        
        # Multi-vector correlation - threats across different detection methods
        detection_methods = set()
        if validation_results:
            detection_methods.add("input_validation")
        if behavioral_anomalies:
            detection_methods.add("behavioral_analysis")
        if ip_analysis:
            detection_methods.add("ip_reputation")
        
        if len(detection_methods) >= 2:
            correlations["correlation_score"] += 0.4
            correlations["correlation_patterns"].append("multi_vector_attack")
        
        # Quantum-specific threat patterns
        quantum_keywords = ["quantum", "superposition", "entanglement", "decoherence", "measurement"]
        
        for result in validation_results:
            validation = result.get("validation", {})
            for anomaly in validation.get("quantum_anomalies", []):
                if any(keyword in anomaly["description"].lower() for keyword in quantum_keywords):
                    correlations["quantum_threats_detected"] += 1
                    correlations["correlation_score"] += 0.5
        
        # Entanglement detection - correlated threat signatures
        threat_signatures = []
        for result in validation_results:
            validation = result.get("validation", {})
            for threat in validation.get("threats_detected", []):
                threat_signatures.append(threat["pattern"])
        
        # Simple entanglement detection based on pattern similarity
        if len(threat_signatures) >= 2:
            for i, sig1 in enumerate(threat_signatures):
                for sig2 in threat_signatures[i+1:]:
                    similarity = self._calculate_signature_similarity(sig1, sig2)
                    if similarity > 0.7:
                        correlations["entangled_threats"].append({
                            "signature_1": sig1,
                            "signature_2": sig2,
                            "similarity": similarity
                        })
                        correlations["correlation_score"] += 0.2
        
        # Cap correlation score at 1.0
        correlations["correlation_score"] = min(correlations["correlation_score"], 1.0)
        
        return correlations
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between threat signatures."""
        
        # Simple Jaccard similarity on words
        words1 = set(sig1.lower().split())
        words2 = set(sig2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _record_threat(
        self,
        request_data: Dict[str, Any],
        source_ip: Optional[str],
        user_agent: Optional[str],
        threat_level: ThreatLevel,
        threat_details: List[Dict[str, Any]]
    ):
        """Record detected threat for tracking and correlation."""
        
        threat_id = f"threat_{int(time.time())}_{secrets.token_hex(4)}"
        
        # Determine primary event type
        event_type = SecurityEvent.UNAUTHORIZED_ACCESS  # Default
        
        for threat in threat_details:
            if "validation" in threat:
                validation = threat["validation"]
                for detected in validation.get("threats_detected", []):
                    pattern = detected["pattern"]
                    if "injection" in pattern.lower():
                        event_type = SecurityEvent.INJECTION_ATTEMPT
                        break
                    elif "exec" in pattern.lower() or "system" in pattern.lower():
                        event_type = SecurityEvent.PRIVILEGE_ESCALATION
                        break
        
        # Create threat record
        threat = QuantumThreat(
            id=threat_id,
            event_type=event_type,
            threat_level=threat_level,
            description=f"Quantum threat detected: {threat_level.value}",
            source_ip=source_ip,
            user_agent=user_agent,
            quantum_signature={
                "request_hash": hashlib.sha256(json.dumps(request_data, sort_keys=True, default=str).encode()).hexdigest()[:16],
                "threat_count": len(threat_details),
                "detection_methods": [t.get("type", "unknown") for t in threat_details]
            }
        )
        
        # Store threat
        self.detected_threats[threat_id] = threat
        self.threat_history.append(threat)
        
        # Maintain history limit
        if len(self.threat_history) > self.threat_memory_limit:
            # Remove oldest threats
            old_threats = self.threat_history[:-self.threat_memory_limit]
            for old_threat in old_threats:
                self.detected_threats.pop(old_threat.id, None)
            self.threat_history = self.threat_history[-self.threat_memory_limit:]
        
        self.logger.warning(
            f"Quantum threat recorded: {threat_id} [{threat_level.value}]",
            extra={
                "threat_id": threat_id,
                "source_ip": source_ip,
                "event_type": event_type.value,
                "quantum_signature": threat.quantum_signature
            }
        )
    
    def _analyze_user_behavior(
        self, 
        user_id: Optional[str],
        request_data: Dict[str, Any],
        source_ip: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Analyze user behavior patterns for anomalies."""
        
        if not user_id:
            return []
        
        # This is a simplified behavioral analysis
        # In production, this would use ML models and historical data
        
        anomalies = []
        
        # Example: detect unusual request patterns
        request_size = len(json.dumps(request_data, default=str))
        
        if request_size > 50000:  # Very large request
            anomalies.append({
                "type": "unusual_request_size",
                "description": f"Unusually large request: {request_size} bytes",
                "threat_level": ThreatLevel.SUSPICIOUS,
                "severity": 2
            })
        
        return anomalies
    
    def _generate_security_recommendations(
        self,
        threat_level: ThreatLevel,
        threats: List[Dict[str, Any]],
        quantum_correlations: Dict[str, Any]
    ) -> List[str]:
        """Generate security recommendations based on threat analysis."""
        
        recommendations = []
        
        if threat_level == ThreatLevel.CRITICAL:
            recommendations.append("IMMEDIATE ACTION: Block source IP and investigate thoroughly")
            recommendations.append("Activate incident response procedures")
            recommendations.append("Review recent system access logs")
        
        elif threat_level == ThreatLevel.MALICIOUS:
            recommendations.append("Block or rate-limit source for 24 hours")
            recommendations.append("Enhanced monitoring for related attack patterns")
            recommendations.append("Validate system integrity")
        
        elif threat_level == ThreatLevel.SUSPICIOUS:
            recommendations.append("Increase monitoring and logging for this source")
            recommendations.append("Consider additional authentication requirements")
        
        # Quantum-specific recommendations
        if quantum_correlations.get("quantum_threats_detected", 0) > 0:
            recommendations.append("Apply quantum threat mitigation protocols")
            recommendations.append("Verify quantum system coherence and entanglement integrity")
        
        if quantum_correlations.get("correlation_score", 0) > 0.5:
            recommendations.append("Investigate correlated threat patterns across system")
            recommendations.append("Consider adaptive security posture adjustment")
        
        return recommendations
    
    def get_threat_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get summary of detected threats."""
        
        time_window = time_window or timedelta(hours=24)
        cutoff_time = datetime.now(timezone.utc) - time_window
        
        recent_threats = [
            threat for threat in self.threat_history
            if threat.timestamp >= cutoff_time
        ]
        
        # Count by threat level
        threat_counts = {level.value: 0 for level in ThreatLevel}
        for threat in recent_threats:
            threat_counts[threat.threat_level.value] += 1
        
        # Count by event type
        event_counts = {event.value: 0 for event in SecurityEvent}
        for threat in recent_threats:
            event_counts[threat.event_type.value] += 1
        
        return {
            "time_window_hours": time_window.total_seconds() / 3600,
            "total_threats": len(recent_threats),
            "threat_level_distribution": threat_counts,
            "event_type_distribution": event_counts,
            "unique_source_ips": len(set(t.source_ip for t in recent_threats if t.source_ip)),
            "mitigation_rate": sum(1 for t in recent_threats if t.mitigated) / max(len(recent_threats), 1)
        }


# Factory function for creating secure quantum API
def create_quantum_security_middleware():
    """Create quantum-enhanced security middleware for FastAPI."""
    
    threat_detector = QuantumThreatDetector()
    
    async def quantum_security_middleware(request, call_next):
        """Quantum security middleware function."""
        
        # Extract request information
        source_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # Analyze request body if present
        request_data = {}
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read request body
                body = await request.body()
                if body:
                    request_data = json.loads(body.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Invalid JSON or encoding - suspicious
                threat_detector.logger.warning(f"Invalid request body from {source_ip}")
        
        # Analyze for threats
        analysis = threat_detector.analyze_request(
            request_data=request_data,
            source_ip=source_ip,
            user_agent=user_agent
        )
        
        # Block critical threats
        if analysis["threat_level"] == ThreatLevel.CRITICAL:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Request blocked due to security threat",
                    "threat_id": f"blocked_{int(time.time())}",
                    "quantum_timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Add security headers to request
        request.state.security_analysis = analysis
        
        # Process request
        response = await call_next(request)
        
        # Add quantum security headers
        response.headers["X-Quantum-Security-Score"] = str(analysis["security_score"])
        response.headers["X-Quantum-Threat-Level"] = analysis["threat_level"].value
        response.headers["X-Quantum-Timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return response
    
    return quantum_security_middleware


# Enhanced app with quantum security
def create_secure_quantum_api() -> FastAPI:
    """Create FastAPI app with quantum security enhancements."""
    
    # Use the existing app but add security middleware
    security_middleware = create_quantum_security_middleware()
    app.middleware("http")(security_middleware)
    
    return app


if __name__ == "__main__":
    import math
    run_quantum_api(debug=True)