"""Common AST parsing utilities with consistent error handling."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from .logging_config import get_generator_logger


class ASTParsingError(Exception):
    """Exception raised when AST parsing fails with structured context."""

    def __init__(self, message: str, file_path: Optional[Path] = None, line_number: Optional[int] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.file_path = file_path
        self.line_number = line_number
        self.original_error = original_error

    def __str__(self) -> str:
        result = super().__str__()
        if self.file_path:
            result += f" (file: {self.file_path})"
        if self.line_number:
            result += f" (line: {self.line_number})"
        return result


def safe_parse_ast(content: str, file_path: Optional[Path] = None) -> ast.AST:
    """Parse Python source code into AST with consistent error handling and logging.
    
    Args:
        content: Python source code to parse
        file_path: Optional path to the source file for better error reporting
        
    Returns:
        Parsed AST tree
        
    Raises:
        ASTParsingError: If parsing fails with structured context
    """
    logger = get_generator_logger()

    if not content.strip():
        # Empty content is valid Python (empty module)
        logger.debug("Parsing empty content as empty module", {
            "file_path": str(file_path) if file_path else "unknown"
        })
        return ast.Module(body=[], type_ignores=[])

    try:
        logger.debug("Starting AST parsing", {
            "file_path": str(file_path) if file_path else "unknown",
            "content_length": len(content),
            "line_count": content.count('\n') + 1
        })

        # Parse the content
        tree = ast.parse(content)

        # Log successful parsing with statistics
        node_count = len(list(ast.walk(tree)))
        logger.debug("AST parsing successful", {
            "file_path": str(file_path) if file_path else "unknown",
            "content_length": len(content),
            "ast_nodes": node_count,
            "top_level_nodes": len(tree.body)
        })

        return tree

    except SyntaxError as e:
        error_message = f"Syntax error in Python code: {e.msg}"

        logger.error("AST parsing failed due to syntax error", {
            "file_path": str(file_path) if file_path else "unknown",
            "line_number": e.lineno,
            "column_number": e.offset,
            "error_message": e.msg,
            "error_text": e.text.strip() if e.text else None
        })

        raise ASTParsingError(
            error_message,
            file_path=file_path,
            line_number=e.lineno,
            original_error=e
        )

    except ValueError as e:
        # Can occur with certain malformed input
        error_message = f"Invalid Python content: {e}"

        logger.error("AST parsing failed due to invalid content", {
            "file_path": str(file_path) if file_path else "unknown",
            "error_message": str(e),
            "content_preview": content[:200] + "..." if len(content) > 200 else content
        })

        raise ASTParsingError(
            error_message,
            file_path=file_path,
            original_error=e
        )

    except RecursionError as e:
        # Can occur with very deeply nested code
        error_message = f"Python code too deeply nested for parsing: {e}"

        logger.error("AST parsing failed due to recursion limit", {
            "file_path": str(file_path) if file_path else "unknown",
            "error_message": str(e),
            "content_length": len(content)
        })

        raise ASTParsingError(
            error_message,
            file_path=file_path,
            original_error=e
        )

    except Exception as e:
        # Catch any other unexpected errors
        error_message = f"Unexpected error during AST parsing: {e}"

        logger.error("AST parsing failed unexpectedly", {
            "file_path": str(file_path) if file_path else "unknown",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "content_length": len(content)
        })

        raise ASTParsingError(
            error_message,
            file_path=file_path,
            original_error=e
        )


def extract_functions(tree: ast.AST) -> list[ast.FunctionDef]:
    """Extract all function definitions from an AST tree.
    
    Args:
        tree: AST tree to search
        
    Returns:
        List of function definition nodes
    """
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node)
    return functions


def extract_classes(tree: ast.AST) -> list[ast.ClassDef]:
    """Extract all class definitions from an AST tree.
    
    Args:
        tree: AST tree to search
        
    Returns:
        List of class definition nodes
    """
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node)
    return classes


def extract_function_names(tree: ast.AST) -> list[str]:
    """Extract names of all functions defined in an AST tree.
    
    Args:
        tree: AST tree to search
        
    Returns:
        List of function names
    """
    return [func.name for func in extract_functions(tree)]


def extract_class_names(tree: ast.AST) -> list[str]:
    """Extract names of all classes defined in an AST tree.
    
    Args:
        tree: AST tree to search
        
    Returns:
        List of class names
    """
    return [cls.name for cls in extract_classes(tree)]
