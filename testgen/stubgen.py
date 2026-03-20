from typing import List, Dict
from .analyzer import extract_functions


def _infer_edge_cases(arg: str, annotation: str) -> List[str]:
    """Return edge case values for an argument based on annotation."""
    t = annotation.lower()
    if "int" in t or "float" in t:
        return ["0", "-1", "None"]
    if "str" in t:
        return ['""', '"a" * 1000', "None"]
    if "list" in t:
        return ["[]", "[None]", "None"]
    if "dict" in t:
        return ["{}", "None"]
    if "bool" in t:
        return ["True", "False", "None"]
    return ["None", '""', "[]"]


def generate_stubs(source_file: str, module_name: str = None) -> str:
    """Generate pytest stub file for all functions in source_file."""
    from pathlib import Path
    source = Path(source_file).read_text(encoding="utf-8")
    functions = extract_functions(source, source_file)

    if module_name is None:
        module_name = Path(source_file).stem

    lines = [
        f"# Auto-generated test stubs for {source_file}",
        "import pytest",
        f"# from {module_name} import *  # adjust import as needed",
        "",
    ]

    for func in functions:
        if func["is_private"]:
            continue
        name = func["name"]
        args = func["args"]
        annotations = func["annotations"]

        # Basic happy path stub
        arg_str = ", ".join("None" for _ in args)
        lines.append(f"def test_{name}_basic():")
        lines.append(f'    """Test basic behavior of {name}."""')
        lines.append(f"    # TODO: replace None with real values")
        lines.append(f"    # result = {name}({arg_str})")
        lines.append(f"    # assert result is not None")
        lines.append(f"    pass")
        lines.append("")

        # Edge case stubs
        if args:
            for i, arg in enumerate(args[:2]):  # limit to first 2 args
                annotation = annotations.get(arg, "")
                edge_cases = _infer_edge_cases(arg, annotation)
                lines.append(f"@pytest.mark.parametrize('{arg}', [{', '.join(edge_cases)}])")
                lines.append(f"def test_{name}_edge_{arg}({arg}):")
                lines.append(f'    """Edge cases for {name} with varying {arg}."""')
                lines.append(f"    # TODO: call {name}({arg}=...{', ...' if len(args) > 1 else ''})")
                lines.append(f"    pass")
                lines.append("")

        # Type error stub
        lines.append(f"def test_{name}_type_error():")
        lines.append(f'    """Test {name} raises TypeError or handles wrong types."""')
        lines.append(f"    # with pytest.raises((TypeError, ValueError)):")
        lines.append(f"    #     {name}()")
        lines.append(f"    pass")
        lines.append("")

    return "\n".join(lines)
