import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

export class CoverageProvider {
    async showCoverage(projectPath: string, pythonPath: string): Promise<void> {
        return vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'Generating coverage report...',
            cancellable: true
        }, async (progress, token) => {
            return new Promise<void>((resolve, reject) => {
                const args = [
                    '-m', 'testgen_copilot',
                    'analyze',
                    '--project', projectPath,
                    '--coverage-target', '0'
                ];

                const child = cp.spawn(pythonPath, args);
                let output = '';
                let errorOutput = '';

                child.stdout?.on('data', (data) => {
                    output += data.toString();
                });

                child.stderr?.on('data', (data) => {
                    errorOutput += data.toString();
                });

                child.on('close', (code) => {
                    if (code === 0) {
                        this.displayCoverageResults(projectPath, output);
                        resolve();
                    } else {
                        vscode.window.showErrorMessage(`Coverage analysis failed: ${errorOutput}`);
                        reject(new Error(`Process exited with code ${code}`));
                    }
                });

                token.onCancellationRequested(() => {
                    child.kill();
                    reject(new Error('Operation cancelled'));
                });
            });
        });
    }

    private async displayCoverageResults(projectPath: string, output: string): Promise<void> {
        // First, try to find and display HTML coverage report
        const htmlCoverageIndex = path.join(projectPath, 'htmlcov', 'index.html');
        
        if (fs.existsSync(htmlCoverageIndex)) {
            // Show HTML coverage report in webview
            await this.showHtmlCoverageReport(htmlCoverageIndex);
        } else {
            // Fallback to text-based coverage display
            await this.showTextCoverageReport(output);
        }

        // Also show coverage decorations in editor
        this.showCoverageDecorations(projectPath);
    }

    private async showHtmlCoverageReport(htmlPath: string): Promise<void> {
        const panel = vscode.window.createWebviewPanel(
            'coverageReport',
            'Coverage Report',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                localResourceRoots: [vscode.Uri.file(path.dirname(htmlPath))]
            }
        );

        try {
            const htmlContent = fs.readFileSync(htmlPath, 'utf8');
            
            // Update resource paths to work in webview
            const updatedContent = htmlContent.replace(
                /href="([^"]+)"/g,
                (match, relativePath) => {
                    if (relativePath.startsWith('http')) return match;
                    const resourceUri = panel.webview.asWebviewUri(
                        vscode.Uri.file(path.join(path.dirname(htmlPath), relativePath))
                    );
                    return `href="${resourceUri}"`;
                }
            ).replace(
                /src="([^"]+)"/g,
                (match, relativePath) => {
                    if (relativePath.startsWith('http')) return match;
                    const resourceUri = panel.webview.asWebviewUri(
                        vscode.Uri.file(path.join(path.dirname(htmlPath), relativePath))
                    );
                    return `src="${resourceUri}"`;
                }
            );

            panel.webview.html = updatedContent;
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to load coverage report: ${error}`);
        }
    }

    private async showTextCoverageReport(output: string): Promise<void> {
        const document = await vscode.workspace.openTextDocument({
            content: this.formatCoverageOutput(output),
            language: 'plaintext'
        });

        await vscode.window.showTextDocument(document, {
            viewColumn: vscode.ViewColumn.Beside,
            preview: false
        });
    }

    private formatCoverageOutput(output: string): string {
        const lines = output.split('\n');
        let formatted = '# Test Coverage Report\n\n';
        
        let inTable = false;
        for (const line of lines) {
            if (line.includes('Name') && line.includes('Stmts') && line.includes('Miss') && line.includes('Cover')) {
                formatted += '## Coverage Summary\n\n';
                formatted += '```\n';
                formatted += line + '\n';
                inTable = true;
            } else if (inTable && line.includes('------')) {
                formatted += line + '\n';
            } else if (inTable && line.trim() === '') {
                formatted += '```\n\n';
                inTable = false;
            } else if (inTable) {
                formatted += line + '\n';
            } else if (line.includes('TOTAL')) {
                formatted += '### Overall Coverage\n\n';
                formatted += `**${line}**\n\n`;
            } else if (line.startsWith('Required coverage')) {
                formatted += `📊 ${line}\n\n`;
            } else if (line.includes('FAILED') || line.includes('ERROR')) {
                formatted += `❌ ${line}\n\n`;
            } else if (line.includes('PASSED') || line.includes('OK')) {
                formatted += `✅ ${line}\n\n`;
            } else if (line.trim()) {
                formatted += `${line}\n\n`;
            }
        }
        
        formatted += '\n---\n\n';
        formatted += `Report generated at: ${new Date().toLocaleString()}\n`;
        formatted += 'Generated by TestGen Copilot Coverage Analyzer\n';
        
        return formatted;
    }

    private showCoverageDecorations(projectPath: string): void {
        // Look for .coverage file or coverage.json
        const coverageJsonPath = path.join(projectPath, 'coverage.json');
        const coverageXmlPath = path.join(projectPath, 'coverage.xml');
        
        if (fs.existsSync(coverageJsonPath)) {
            this.applyCoverageDecorations(coverageJsonPath, 'json');
        } else if (fs.existsSync(coverageXmlPath)) {
            this.applyCoverageDecorations(coverageXmlPath, 'xml');
        }
    }

    private applyCoverageDecorations(coveragePath: string, format: 'json' | 'xml'): void {
        try {
            const decorationType = vscode.window.createTextEditorDecorationType({
                backgroundColor: 'rgba(255, 0, 0, 0.2)',
                border: '1px solid rgba(255, 0, 0, 0.5)',
                overviewRulerColor: 'red',
                overviewRulerLane: vscode.OverviewRulerLane.Right
            });

            // Parse coverage data based on format
            let coverageData: any = {};
            
            if (format === 'json') {
                const content = fs.readFileSync(coveragePath, 'utf8');
                coverageData = JSON.parse(content);
            }
            // XML parsing would go here for format === 'xml'

            // Apply decorations to currently open Python files
            vscode.window.visibleTextEditors.forEach(editor => {
                if (editor.document.languageId === 'python') {
                    const filePath = editor.document.fileName;
                    const relativePath = path.relative(path.dirname(coveragePath), filePath);
                    
                    if (coverageData.files && coverageData.files[relativePath]) {
                        const missedLines = coverageData.files[relativePath].missing_lines || [];
                        const decorations: vscode.DecorationOptions[] = [];
                        
                        for (const lineNum of missedLines) {
                            const range = new vscode.Range(lineNum - 1, 0, lineNum - 1, 1000);
                            decorations.push({
                                range,
                                hoverMessage: 'This line is not covered by tests'
                            });
                        }
                        
                        editor.setDecorations(decorationType, decorations);
                    }
                }
            });
        } catch (error) {
            console.error('Failed to apply coverage decorations:', error);
        }
    }
}