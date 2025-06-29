from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any, List


def scaffold_extension(path: str | Path) -> Path:
    """Create minimal VS Code extension scaffold under *path*.

    Returns the path to the created ``package.json`` file.
    Raises ``FileExistsError`` if the target already contains a package.
    """

    dest = Path(path)
    package_file = dest / "package.json"
    if package_file.exists():
        raise FileExistsError(f"{package_file} already exists")

    (dest / "src").mkdir(parents=True, exist_ok=True)

    package_data = {
        "name": "testgen-copilot",
        "displayName": "TestGen Copilot",
        "publisher": "testgen",
        "version": "0.0.1",
        "engines": {"vscode": "^1.60.0"},
        "activationEvents": [
            "onCommand:testgen.generateTests",
            "onCommand:testgen.generateTestsFromActiveFile",
            "onCommand:testgen.runSecurityScan",
            "onCommand:testgen.showCoverage",
        ],
        "main": "./src/extension.js",
        "contributes": {
            "commands": [
                {
                    "command": "testgen.generateTests",
                    "title": "Generate Tests with TestGen",
                },
                {
                    "command": "testgen.generateTestsFromActiveFile",
                    "title": "Generate Tests for Active File",
                },
                {
                    "command": "testgen.runSecurityScan",
                    "title": "Run Security Scan",
                },
                {
                    "command": "testgen.showCoverage",
                    "title": "Show Coverage",
                },
            ]
        },
    }
    package_file.write_text(json.dumps(package_data, indent=2) + "\n")

    extension_js = dest / "src" / "extension.js"
    extension_js.write_text(
        """const vscode = require('vscode');
const cp = require('child_process');

function activate(context) {
  let disposable = vscode.commands.registerCommand(
    'testgen.generateTests',
    () => {
      vscode.window.showInformationMessage('TestGen Copilot activated!');
    },
  );

  let genActive = vscode.commands.registerCommand(
    'testgen.generateTestsFromActiveFile',
    () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('No active editor');
      return;
    }
    const filePath = editor.document.fileName;
    const workspace = vscode.workspace.workspaceFolders?.[0];
    const outDir = workspace ? `${workspace.uri.fsPath}/tests` : '';
    cp.execFile(
      'python',
      ['-m', 'testgen_copilot', 'generate', '--file', filePath, '--output', outDir],
      err => {
        if (err) {
          vscode.window.showErrorMessage('Failed to generate tests');
        } else {
          vscode.window.showInformationMessage('Tests generated');
        }
      },
    );
  });

  let runScan = vscode.commands.registerCommand(
    'testgen.runSecurityScan',
    () => {
      const workspace = vscode.workspace.workspaceFolders?.[0];
      if (!workspace) {
        vscode.window.showErrorMessage('No workspace folder found');
        return;
      }
      cp.execFile(
        'python',
        ['-m', 'testgen_copilot', 'analyze', '--project', workspace.uri.fsPath, '--security-scan'],
        err => {
          if (err) {
            vscode.window.showErrorMessage('Security scan failed');
          } else {
            vscode.window.showInformationMessage('Security scan complete');
          }
        },
      );
    },
  );

  let showCov = vscode.commands.registerCommand(
    'testgen.showCoverage',
    () => {
      const workspace = vscode.workspace.workspaceFolders?.[0];
      if (!workspace) {
        vscode.window.showErrorMessage('No workspace folder found');
        return;
      }
      cp.execFile(
        'python',
        ['-m', 'testgen_copilot', 'analyze', '--project', workspace.uri.fsPath, '--coverage-target', '0'],
        (err, stdout) => {
          if (err) {
            vscode.window.showErrorMessage('Coverage run failed');
          } else {
            vscode.window.showInformationMessage(stdout);
          }
        },
      );
    },
  );

  context.subscriptions.push(disposable, genActive, runScan, showCov);
}

function deactivate() {}

module.exports = { activate, deactivate };
"""
    )

    return package_file


def suggest_from_diagnostics(
    diagnostics: Iterable[Mapping[str, Any]] | None,
) -> List[str]:
    """Convert LSP diagnostics to simple suggestion strings."""

    if not diagnostics:
        return []

    severity_map = {1: "Hint", 2: "Info", 3: "Warning", 4: "Error"}
    suggestions: List[str] = []
    for diag in diagnostics:
        msg = str(diag.get("message", ""))
        if not msg:
            continue
        sev = severity_map.get(int(diag.get("severity", 1)), "Info")
        suggestions.append(f"{sev}: {msg}")

    return suggestions


def write_usage_docs(path: str | Path) -> Path:
    """Write extension usage documentation to the given directory.

    Returns the path to the created ``USAGE.md`` file. Raises
    ``NotADirectoryError`` if *path* is not a directory or cannot be created.
    """

    dest = Path(path)
    if dest.exists() and not dest.is_dir():
        raise NotADirectoryError(f"{dest} is not a directory")

    dest.mkdir(parents=True, exist_ok=True)
    usage_file = dest / "USAGE.md"
    usage_file.write_text(
        """# TestGen Copilot VS Code Extension Usage

## Commands

- **Generate Tests with TestGen** – generate tests for the current project.
- **Generate Tests for Active File** – generate tests for the currently open file.
- **Run Security Scan** – check the project for insecure code patterns.
- **Show Coverage** – display coverage statistics in VS Code.

Place generated tests under the project's ``tests`` directory and review them before committing.
"""
    )

    return usage_file
