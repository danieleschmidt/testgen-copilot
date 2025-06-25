import json
import pathlib
import re
from typing import List


PLAN_PATH = pathlib.Path("DEVELOPMENT_PLAN.md")
BOARD_PATH = pathlib.Path("SPRINT_BOARD.md")
CRITERIA_PATH = pathlib.Path("tests/sprint_acceptance_criteria.json")


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def read_plan() -> List[str]:
    text = PLAN_PATH.read_text()
    # find first unchecked task line starting with '- [ ]'
    for line in text.splitlines():
        if line.startswith("- [ ]"):
            task = line[5:].strip()
            return [task]
    return []


def main() -> None:
    tasks = read_plan()
    if not tasks:
        print("No upcoming tasks found")
        return
    epic = tasks[0]
    epic_clean = re.sub(r"^\*\*Feature:\*\*\s*", "", epic)
    epic_name = epic_clean.split(" - ")[0].strip()

    sub_tasks = [
        f"Scaffold VS Code extension for {epic_name}",
        "Add command to generate tests from active file",
        "Provide real-time suggestions using LSP diagnostics",
        "Document extension usage and configuration",
    ]

    # prepare board
    lines = [
        "# Sprint Board",
        "",
        "## Backlog",
        "| Task | Owner | Priority | Status |",
        "| --- | --- | --- | --- |",
    ]
    for st in sub_tasks:
        lines.append(f"| {st} | @agent | P1 | Todo |")

    if BOARD_PATH.exists():
        board_text = BOARD_PATH.read_text()
        done_rows = re.findall(r"\| ([^|]+) \| [^|]+ \| [^|]+ \| Done \|", board_text)
        for row in done_rows:
            lines.append(f"| {row} | @agent | P0 | Done |")

    BOARD_PATH.write_text("\n".join(lines) + "\n")

    criteria = {}
    for st in sub_tasks:
        slug = slugify(st)
        criteria[slug] = {
            "test_file": f"tests/{slug}.py",
            "description": st,
            "cases": {
                "success": f"{st} functions correctly",
                "edge_case": "Handles error conditions gracefully",
            },
        }
    if CRITERIA_PATH.exists():
        existing = json.loads(CRITERIA_PATH.read_text())
        existing.update(criteria)
        criteria = existing

    CRITERIA_PATH.write_text(json.dumps(criteria, indent=2) + "\n")
    print("Sprint board and criteria updated")


if __name__ == "__main__":
    main()
