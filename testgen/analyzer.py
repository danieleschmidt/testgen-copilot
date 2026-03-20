import ast
from pathlib import Path
from typing import List, Dict, Any


def extract_functions(source: str, filename: str = "<string>") -> List[Dict]:
    """Extract all function definitions from source code."""
    tree = ast.parse(source, filename=filename)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private/dunder methods
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            args = [a.arg for a in node.args.args if a.arg != "self"]
            annotations = {}
            for a in node.args.args:
                if a.arg != "self" and a.annotation:
                    annotations[a.arg] = ast.unparse(a.annotation)
            docstring = ast.get_docstring(node) or ""
            functions.append({
                "name": node.name,
                "args": args,
                "annotations": annotations,
                "docstring": docstring,
                "lineno": node.lineno,
                "is_private": node.name.startswith("_"),
            })
    return functions


def extract_tested_functions(test_source: str) -> set:
    """Extract function names that appear to be tested."""
    tested = set()
    tree = ast.parse(test_source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                # Naive: collect all Name/Attribute calls within test functions
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            tested.add(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            tested.add(child.func.attr)
    return tested


def find_untested(source_file: str, test_file: str = None) -> List[Dict]:
    """Return list of functions in source_file that lack tests."""
    source = Path(source_file).read_text(encoding="utf-8")
    funcs = extract_functions(source, source_file)

    tested = set()
    if test_file:
        try:
            test_source = Path(test_file).read_text(encoding="utf-8")
            tested = extract_tested_functions(test_source)
        except FileNotFoundError:
            pass

    return [f for f in funcs if f["name"] not in tested]
