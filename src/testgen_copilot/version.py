"""Version management for TestGen Copilot package."""

import sys
from typing import Optional

# Fallback version for development and edge cases
FALLBACK_VERSION = "0.1.0"


def get_package_version(package_name: str = "testgen-copilot") -> str:
    """Get the package version from metadata with fallback handling.
    
    Args:
        package_name: Name of the package to get version for
        
    Returns:
        str: Package version string, or fallback version if detection fails
    """

    # Try modern importlib.metadata (Python 3.8+)
    try:
        if sys.version_info >= (3, 8):
            import importlib.metadata as metadata
        else:
            # Fallback for Python 3.7
            import importlib_metadata as metadata

        version = metadata.version(package_name)
        return version

    except ImportError:
        # importlib_metadata not available, try pkg_resources
        pass

    except metadata.PackageNotFoundError:
        # Package not found in metadata
        pass

    except Exception:
        pass

    # Try pkg_resources as fallback
    try:
        import pkg_resources
        version = pkg_resources.get_distribution(package_name).version
        return version

    except ImportError:
        pass

    except pkg_resources.DistributionNotFound:
        pass

    except Exception:
        pass

    # Try reading from pyproject.toml as final fallback
    version = _read_version_from_pyproject()
    if version:
        return version

    # Use fallback version
    return FALLBACK_VERSION


def _read_version_from_pyproject() -> Optional[str]:
    """Try to read version from pyproject.toml file."""
    try:
        from pathlib import Path

        # Look for pyproject.toml in common locations
        search_paths = [
            Path(__file__).parent.parent.parent / "pyproject.toml",  # ../../../pyproject.toml
            Path(__file__).parent.parent / "pyproject.toml",        # ../../pyproject.toml
            Path.cwd() / "pyproject.toml",                          # ./pyproject.toml
        ]

        for pyproject_path in search_paths:
            if pyproject_path.exists():
                content = pyproject_path.read_text(encoding='utf-8')

                # Simple regex to extract version
                import re
                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if version_match:
                    return version_match.group(1)

    except Exception:
        # Silently fail - this is a fallback method
        pass

    return None


def get_version_info() -> dict:
    """Get comprehensive version information for debugging.
    
    Returns:
        dict: Version information including method used and Python version
    """
    version = get_package_version()

    # Determine which method was successful
    methods_tried = []
    final_method = "unknown"

    # Check which method would work
    try:
        if sys.version_info >= (3, 8):
            import importlib.metadata as metadata
            try:
                metadata.version("testgen-copilot")
                final_method = "importlib.metadata"
            except metadata.PackageNotFoundError:
                pass
        else:
            import importlib_metadata as metadata
            try:
                metadata.version("testgen-copilot")
                final_method = "importlib_metadata"
            except metadata.PackageNotFoundError:
                pass
    except ImportError:
        pass

    if final_method == "unknown":
        try:
            import pkg_resources
            pkg_resources.get_distribution("testgen-copilot")
            final_method = "pkg_resources"
        except (ImportError, pkg_resources.DistributionNotFound):
            pass

    if final_method == "unknown":
        if _read_version_from_pyproject():
            final_method = "pyproject_toml"
        else:
            final_method = "fallback"

    return {
        "version": version,
        "method": final_method,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "fallback_version": FALLBACK_VERSION,
        "package_name": "testgen-copilot"
    }


# Export the version for easy access
__version__ = FALLBACK_VERSION
