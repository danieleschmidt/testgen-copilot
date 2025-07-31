import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export class TestGenProvider implements vscode.TreeDataProvider<TestGenItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<TestGenItem | undefined | null | void> = new vscode.EventEmitter<TestGenItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<TestGenItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor() {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: TestGenItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: TestGenItem): Thenable<TestGenItem[]> {
        if (!vscode.workspace.workspaceFolders) {
            return Promise.resolve([]);
        }

        const workspaceFolder = vscode.workspace.workspaceFolders[0];
        
        if (element) {
            return Promise.resolve(this.getTestFiles(element.resourceUri!.fsPath));
        } else {
            return Promise.resolve(this.getProjectStructure(workspaceFolder.uri.fsPath));
        }
    }

    private getProjectStructure(projectPath: string): TestGenItem[] {
        const items: TestGenItem[] = [];
        
        // Test Coverage Summary
        items.push(new TestGenItem(
            'Test Coverage',
            vscode.TreeItemCollapsibleState.None,
            'coverage-summary',
            undefined,
            {
                command: 'testgen.showCoverage',
                title: 'Show Coverage'
            }
        ));

        // Security Scan Status
        items.push(new TestGenItem(
            'Security Scan',
            vscode.TreeItemCollapsibleState.None,
            'security-scan',
            undefined,
            {
                command: 'testgen.runSecurityScan',
                title: 'Run Security Scan'
            }
        ));

        // Test Files
        const testsDir = path.join(projectPath, 'tests');
        if (fs.existsSync(testsDir)) {
            items.push(new TestGenItem(
                'Test Files',
                vscode.TreeItemCollapsibleState.Expanded,
                'test-files',
                vscode.Uri.file(testsDir)
            ));
        }

        // Python Files Without Tests
        const pythonFiles = this.findPythonFilesWithoutTests(projectPath);
        if (pythonFiles.length > 0) {
            items.push(new TestGenItem(
                `Files Needing Tests (${pythonFiles.length})`,
                vscode.TreeItemCollapsibleState.Collapsed,
                'files-needing-tests'
            ));
        }

        return items;
    }

    private getTestFiles(dirPath: string): TestGenItem[] {
        const items: TestGenItem[] = [];
        
        try {
            const files = fs.readdirSync(dirPath);
            
            for (const file of files) {
                const filePath = path.join(dirPath, file);
                const stat = fs.statSync(filePath);
                
                if (stat.isFile() && file.endsWith('.py') && file.startsWith('test_')) {
                    items.push(new TestGenItem(
                        file,
                        vscode.TreeItemCollapsibleState.None,
                        'test-file',
                        vscode.Uri.file(filePath),
                        {
                            command: 'vscode.open',
                            title: 'Open',
                            arguments: [vscode.Uri.file(filePath)]
                        }
                    ));
                }
            }
        } catch (error) {
            console.error('Error reading test files:', error);
        }
        
        return items;
    }

    private findPythonFilesWithoutTests(projectPath: string): string[] {
        const pythonFiles: string[] = [];
        const testsDir = path.join(projectPath, 'tests');
        
        const findPythonFiles = (dir: string) => {
            try {
                const files = fs.readdirSync(dir);
                
                for (const file of files) {
                    const filePath = path.join(dir, file);
                    const stat = fs.statSync(filePath);
                    
                    if (stat.isDirectory() && !file.startsWith('.') && file !== 'tests') {
                        findPythonFiles(filePath);
                    } else if (stat.isFile() && file.endsWith('.py') && !file.startsWith('test_')) {
                        const relativePath = path.relative(projectPath, filePath);
                        const testFileName = `test_${path.basename(file)}`;
                        const expectedTestPath = path.join(testsDir, testFileName);
                        
                        if (!fs.existsSync(expectedTestPath)) {
                            pythonFiles.push(relativePath);
                        }
                    }
                }
            } catch (error) {
                console.error('Error scanning directory:', error);
            }
        };
        
        findPythonFiles(projectPath);
        return pythonFiles;
    }
}

class TestGenItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly contextValue: string,
        public readonly resourceUri?: vscode.Uri,
        public readonly command?: vscode.Command
    ) {
        super(label, collapsibleState);
        
        this.tooltip = this.getTooltip();
        this.iconPath = this.getIconPath();
    }

    private getTooltip(): string {
        switch (this.contextValue) {
            case 'coverage-summary':
                return 'View test coverage report';
            case 'security-scan':
                return 'Run security vulnerability scan';
            case 'test-files':
                return 'Browse test files';
            case 'files-needing-tests':
                return 'Python files that need test coverage';
            case 'test-file':
                return `Open ${this.label}`;
            default:
                return this.label;
        }
    }

    private getIconPath(): vscode.ThemeIcon {
        switch (this.contextValue) {
            case 'coverage-summary':
                return new vscode.ThemeIcon('graph');
            case 'security-scan':
                return new vscode.ThemeIcon('shield');
            case 'test-files':
                return new vscode.ThemeIcon('folder');
            case 'files-needing-tests':
                return new vscode.ThemeIcon('warning');
            case 'test-file':
                return new vscode.ThemeIcon('beaker');
            default:
                return new vscode.ThemeIcon('file');
        }
    }
}