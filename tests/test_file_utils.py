"""Tests for safe file I/O utilities."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import tempfile
import os

from testgen_copilot.file_utils import safe_read_file, FileSizeError


class TestSafeReadFile:
    """Test safe_read_file utility function."""

    def test_safe_read_file_success(self, tmp_path):
        """Test successful file reading."""
        test_file = tmp_path / "test.py"
        test_content = "def hello():\n    return 'world'"
        test_file.write_text(test_content)
        
        result = safe_read_file(test_file)
        assert result == test_content

    def test_safe_read_file_nonexistent(self, tmp_path):
        """Test reading non-existent file raises FileNotFoundError."""
        nonexistent_file = tmp_path / "does_not_exist.py"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            safe_read_file(nonexistent_file)
        
        assert "not found" in str(exc_info.value).lower()
        assert str(nonexistent_file) in str(exc_info.value)

    def test_safe_read_file_permission_error(self, tmp_path):
        """Test reading file with permission error."""
        test_file = tmp_path / "no_permission.py"
        test_file.write_text("content")
        
        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError) as exc_info:
                safe_read_file(test_file)
            
            assert "permission denied" in str(exc_info.value).lower()
            assert str(test_file) in str(exc_info.value)

    def test_safe_read_file_unicode_error(self, tmp_path):
        """Test reading file with unicode decode error."""
        test_file = tmp_path / "bad_encoding.py"
        # Write binary data that will cause UnicodeDecodeError
        test_file.write_bytes(b'\xff\xfe\x80\x81')
        
        with pytest.raises(ValueError) as exc_info:
            safe_read_file(test_file)
        
        assert "encoding" in str(exc_info.value).lower()
        assert str(test_file) in str(exc_info.value)

    def test_safe_read_file_size_limit_default(self, tmp_path):
        """Test file size limit with default max size."""
        test_file = tmp_path / "large.py"
        
        with patch("pathlib.Path.stat") as mock_stat:
            # Mock file size larger than default 10MB limit
            from stat import S_IFREG
            mock_stat_result = type('MockStat', (), {
                'st_size': 11 * 1024 * 1024,  # 11MB
                'st_mode': S_IFREG  # Regular file mode
            })()
            mock_stat.return_value = mock_stat_result
            
            with pytest.raises(FileSizeError) as exc_info:
                safe_read_file(test_file)
            
            assert "too large" in str(exc_info.value).lower()
            assert str(test_file) in str(exc_info.value)
            assert "10MB" in str(exc_info.value)

    def test_safe_read_file_size_limit_custom(self, tmp_path):
        """Test file size limit with custom max size."""
        test_file = tmp_path / "medium.py"
        
        with patch("pathlib.Path.stat") as mock_stat:
            # Mock file size larger than custom 1MB limit
            from stat import S_IFREG
            mock_stat_result = type('MockStat', (), {
                'st_size': 2 * 1024 * 1024,  # 2MB
                'st_mode': S_IFREG  # Regular file mode
            })()
            mock_stat.return_value = mock_stat_result
            
            with pytest.raises(FileSizeError) as exc_info:
                safe_read_file(test_file, max_size_mb=1)
            
            assert "too large" in str(exc_info.value).lower()
            assert "1MB" in str(exc_info.value)

    def test_safe_read_file_size_within_limit(self, tmp_path):
        """Test file reading succeeds when size is within limit."""
        test_file = tmp_path / "small.py"
        test_content = "small content"
        test_file.write_text(test_content)
        
        # Should succeed with default limit
        result = safe_read_file(test_file)
        assert result == test_content
        
        # Should succeed with custom limit larger than file
        result = safe_read_file(test_file, max_size_mb=1)
        assert result == test_content

    def test_safe_read_file_pathlib_object(self, tmp_path):
        """Test that function works with Path objects."""
        test_file = tmp_path / "pathlib_test.py"
        test_content = "pathlib content"
        test_file.write_text(test_content)
        
        result = safe_read_file(test_file)  # Path object
        assert result == test_content

    def test_safe_read_file_string_path(self, tmp_path):
        """Test that function works with string paths."""
        test_file = tmp_path / "string_test.py"
        test_content = "string path content"
        test_file.write_text(test_content)
        
        result = safe_read_file(str(test_file))  # String path
        assert result == test_content

    def test_safe_read_file_os_error(self, tmp_path):
        """Test handling of general OS errors."""
        test_file = tmp_path / "os_error.py"
        test_file.write_text("content")
        
        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = OSError("Disk full")
            
            with pytest.raises(OSError) as exc_info:
                safe_read_file(test_file)
            
            assert "file i/o error" in str(exc_info.value).lower()
            assert str(test_file) in str(exc_info.value)