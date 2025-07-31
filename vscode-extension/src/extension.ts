import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import { TestGenProvider } from './testGenProvider';
import { SecurityProvider } from './securityProvider';
import { CoverageProvider } from './coverageProvider';

export function activate(context: vscode.ExtensionContext) {
    console.log('TestGen Copilot is now active!');

    // Initialize providers
    const testGenProvider = new TestGenProvider();
    const securityProvider = new SecurityProvider();
    const coverageProvider = new CoverageProvider();

    // Register providers
    vscode.window.registerTreeDataProvider('testgenExplorer', testGenProvider);

    // Command: Generate Tests
    const generateTests = vscode.commands.registerCommand('testgen.generateTests', async () => {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('No workspace folder found');
            return;
        }

        const config = vscode.workspace.getConfiguration('testgen');
        const pythonPath = config.get<string>('pythonPath', 'python');
        const outputDir = config.get<string>('outputDirectory', 'tests');

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'Generating tests...',
            cancellable: true
        }, async (progress, token) => {
            return new Promise<void>((resolve, reject) => {
                const child = cp.spawn(pythonPath, [
                    '-m', 'testgen_copilot', 
                    'generate',
                    '--project', workspaceFolder.uri.fsPath,
                    '--output', path.join(workspaceFolder.uri.fsPath, outputDir)
                ]);

                child.on('close', (code) => {
                    if (code === 0) {
                        vscode.window.showInformationMessage('Tests generated successfully!');
                        resolve();
                    } else {
                        vscode.window.showErrorMessage('Failed to generate tests');
                        reject(new Error(`Process exited with code ${code}`));
                    }
                });

                token.onCancellationRequested(() => {
                    child.kill();
                    reject(new Error('Operation cancelled'));
                });
            });
        });
    });

    // Command: Generate Tests from Active File
    const generateTestsFromActiveFile = vscode.commands.registerCommand('testgen.generateTestsFromActiveFile', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const document = editor.document;
        if (document.languageId !== 'python') {
            vscode.window.showErrorMessage('Active file must be a Python file');
            return;
        }

        const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('File must be in a workspace');
            return;
        }

        const config = vscode.workspace.getConfiguration('testgen');
        const pythonPath = config.get<string>('pythonPath', 'python');
        const outputDir = config.get<string>('outputDirectory', 'tests');

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Generating tests for ${path.basename(document.fileName)}...`,
            cancellable: true
        }, async (progress, token) => {
            return new Promise<void>((resolve, reject) => {
                const child = cp.spawn(pythonPath, [
                    '-m', 'testgen_copilot',
                    'generate',
                    '--file', document.fileName,
                    '--output', path.join(workspaceFolder.uri.fsPath, outputDir)
                ]);

                child.on('close', (code) => {
                    if (code === 0) {
                        vscode.window.showInformationMessage(`Tests generated for ${path.basename(document.fileName)}!`);
                        resolve();
                    } else {
                        vscode.window.showErrorMessage('Failed to generate tests');
                        reject(new Error(`Process exited with code ${code}`));
                    }
                });

                token.onCancellationRequested(() => {
                    child.kill();
                    reject(new Error('Operation cancelled'));
                });
            });
        });
    });

    // Command: Run Security Scan
    const runSecurityScan = vscode.commands.registerCommand('testgen.runSecurityScan', async () => {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('No workspace folder found');
            return;
        }

        const config = vscode.workspace.getConfiguration('testgen');
        const pythonPath = config.get<string>('pythonPath', 'python');
        const scanLevel = config.get<string>('securityScanLevel', 'standard');

        await securityProvider.runScan(workspaceFolder.uri.fsPath, pythonPath, scanLevel);
    });

    // Command: Show Coverage
    const showCoverage = vscode.commands.registerCommand('testgen.showCoverage', async () => {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('No workspace folder found');
            return;
        }

        const config = vscode.workspace.getConfiguration('testgen');
        const pythonPath = config.get<string>('pythonPath', 'python');

        await coverageProvider.showCoverage(workspaceFolder.uri.fsPath, pythonPath);
    });

    // Command: Open Settings
    const openSettings = vscode.commands.registerCommand('testgen.openSettings', () => {
        vscode.commands.executeCommand('workbench.action.openSettings', 'testgen');
    });

    // Code Lens Provider
    if (vscode.workspace.getConfiguration('testgen').get('enableCodeLens')) {
        const codeLensProvider = vscode.languages.registerCodeLensProvider(
            { language: 'python' },
            {
                provideCodeLenses: (document, token) => {
                    const lenses: vscode.CodeLens[] = [];
                    const text = document.getText();
                    const lines = text.split('\n');

                    lines.forEach((line, index) => {
                        if (line.match(/^(def |class )/)) {
                            const range = new vscode.Range(index, 0, index, line.length);
                            lenses.push(new vscode.CodeLens(range, {
                                title: "⚡ Generate Tests",
                                command: "testgen.generateTestsFromActiveFile"
                            }));
                        }
                    });

                    return lenses;
                }
            }
        );
        context.subscriptions.push(codeLensProvider);
    }

    // Register all commands
    context.subscriptions.push(
        generateTests,
        generateTestsFromActiveFile,
        runSecurityScan,
        showCoverage,
        openSettings
    );

    // Language server integration for real-time suggestions
    if (vscode.workspace.getConfiguration('testgen').get('enableRealTimeSuggestions')) {
        const diagnosticCollection = vscode.languages.createDiagnosticCollection('testgen');
        context.subscriptions.push(diagnosticCollection);

        // Monitor Python file changes
        const fileWatcher = vscode.workspace.createFileSystemWatcher('**/*.py');
        fileWatcher.onDidChange((uri) => {
            // Provide real-time suggestions based on code changes
            provideDiagnostics(uri, diagnosticCollection);
        });
        context.subscriptions.push(fileWatcher);
    }
}

async function provideDiagnostics(uri: vscode.Uri, collection: vscode.DiagnosticCollection) {
    const document = await vscode.workspace.openTextDocument(uri);
    const diagnostics: vscode.Diagnostic[] = [];
    
    const text = document.getText();
    const lines = text.split('\n');
    
    lines.forEach((line, index) => {
        // Suggest tests for functions without corresponding test files
        if (line.match(/^def [a-zA-Z_][a-zA-Z0-9_]*\(/)) {
            const range = new vscode.Range(index, 0, index, line.length);
            const diagnostic = new vscode.Diagnostic(
                range,
                'Consider generating tests for this function',
                vscode.DiagnosticSeverity.Information
            );
            diagnostic.code = 'testgen-suggestion';
            diagnostics.push(diagnostic);
        }
        
        // Security suggestions
        if (line.includes('eval(') || line.includes('exec(')) {
            const range = new vscode.Range(index, 0, index, line.length);
            const diagnostic = new vscode.Diagnostic(
                range,
                'Potential security risk: eval/exec usage',
                vscode.DiagnosticSeverity.Warning
            );
            diagnostic.code = 'testgen-security';
            diagnostics.push(diagnostic);
        }
    });
    
    collection.set(uri, diagnostics);
}

export function deactivate() {
    console.log('TestGen Copilot deactivated');
}