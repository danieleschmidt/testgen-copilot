#!/usr/bin/env python3
"""
Basic functionality test for TestGen Copilot Assistant
Tests core functionality to ensure GENERATION 1 success
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that core modules can be imported"""
    try:
        import testgen_copilot
        from testgen_copilot.cli import main
        from testgen_copilot.core import identity
        from testgen_copilot.quantum_planner import create_quantum_planner
        from testgen_copilot.generator import TestGenerator
        from testgen_copilot.security import SecurityScanner
        print("✅ All core modules imported successfully")
        assert True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        assert False, f"Import error: {e}"

def test_quantum_planner_creation():
    """Test quantum planner instantiation"""
    try:
        from testgen_copilot.quantum_planner import create_quantum_planner
        planner = create_quantum_planner()
        print(f"✅ Quantum planner created with {len(planner.resources)} resources")
        assert True
    except Exception as e:
        print(f"❌ Quantum planner error: {e}")
        assert False, f"Quantum planner error: {e}"

def test_test_generator_creation():
    """Test test generator instantiation"""
    try:
        from testgen_copilot.generator import TestGenerator, GenerationConfig
        config = GenerationConfig(language="python")
        generator = TestGenerator(config)
        print("✅ Test generator created successfully")
        assert True
    except Exception as e:
        print(f"❌ Test generator error: {e}")
        assert False, f"Test generator error: {e}"

def test_security_scanner_creation():
    """Test security scanner instantiation"""
    try:
        from testgen_copilot.security import SecurityScanner
        scanner = SecurityScanner()
        print("✅ Security scanner created successfully")
        assert True
    except Exception as e:
        print(f"❌ Security scanner error: {e}")
        assert False, f"Security scanner error: {e}"

def test_cli_help():
    """Test CLI help functionality"""
    try:
        from testgen_copilot.cli import _build_parser
        parser = _build_parser()
        help_text = parser.format_help()
        assert "TestGen Copilot CLI" in help_text
        assert "quantum" in help_text
        assert "generate" in help_text
        print("✅ CLI help functionality works")
        assert True
    except Exception as e:
        print(f"❌ CLI help error: {e}")
        assert False, f"CLI help error: {e}"

def main():
    """Run all basic functionality tests"""
    print("🚀 GENERATION 1: Testing Basic Functionality")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_quantum_planner_creation,
        test_test_generator_creation,
        test_security_scanner_creation,
        test_cli_help
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"✅ {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 GENERATION 1: MAKE IT WORK - SUCCESS!")
        print("   All basic functionality is working correctly.")
    else:
        print("❌ GENERATION 1: FAILED - Some basic functionality is broken")
        sys.exit(1)

if __name__ == "__main__":
    main()