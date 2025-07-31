#!/usr/bin/env python3
"""
Advanced Changelog Generator for TestGen Copilot
Generates comprehensive changelogs with AI-powered commit analysis and categorization.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Conventional commit types with enhanced categorization
COMMIT_TYPES = {
    'feat': {'section': '✨ Features', 'emoji': '✨', 'breaking': False},
    'fix': {'section': '🐛 Bug Fixes', 'emoji': '🐛', 'breaking': False},
    'perf': {'section': '⚡ Performance Improvements', 'emoji': '⚡', 'breaking': False},
    'revert': {'section': '⏪ Reverts', 'emoji': '⏪', 'breaking': False},
    'docs': {'section': '📚 Documentation', 'emoji': '📚', 'breaking': False},
    'style': {'section': '💎 Styles', 'emoji': '💎', 'breaking': False},
    'refactor': {'section': '♻️ Code Refactoring', 'emoji': '♻️', 'breaking': False},
    'test': {'section': '✅ Tests', 'emoji': '✅', 'breaking': False},
    'build': {'section': '🔧 Build System', 'emoji': '🔧', 'breaking': False},
    'ci': {'section': '🔄 Continuous Integration', 'emoji': '🔄', 'breaking': False},
    'chore': {'section': '🏗️ Chores', 'emoji': '🏗️', 'breaking': False},
    'security': {'section': '🔒 Security', 'emoji': '🔒', 'breaking': False},
}

class CommitParser:
    """Parse and categorize git commits using conventional commit format."""
    
    def __init__(self):
        self.conventional_pattern = re.compile(
            r'^(?P<type>\w+)(?:\((?P<scope>[\w\-_]+)\))?(?P<breaking>!)?: (?P<description>.*?)(?:\n\n(?P<body>.*?))?(?:\n\n(?P<footer>.*))?$',
            re.DOTALL
        )
        
    def parse_commit(self, commit_hash: str, commit_message: str) -> Dict:
        """Parse a single commit message and extract metadata."""
        lines = commit_message.strip().split('\n')
        subject = lines[0]
        body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''
        
        match = self.conventional_pattern.match(commit_message.strip())
        
        if match:
            groups = match.groupdict()
            commit_type = groups['type'].lower()
            is_breaking = groups['breaking'] == '!' or 'BREAKING CHANGE' in body
            
            return {
                'hash': commit_hash,
                'type': commit_type,
                'scope': groups['scope'],
                'description': groups['description'],
                'body': groups['body'] or '',
                'footer': groups['footer'] or '',
                'is_breaking': is_breaking,
                'raw_message': commit_message,
                'subject': subject,
                'author': self._get_commit_author(commit_hash),
                'date': self._get_commit_date(commit_hash),
                'pr_number': self._extract_pr_number(commit_message),
                'issues': self._extract_issue_numbers(commit_message),
            }
        else:
            # Non-conventional commit
            return {
                'hash': commit_hash,
                'type': 'other',
                'scope': None,
                'description': subject,
                'body': body,
                'footer': '',
                'is_breaking': 'BREAKING CHANGE' in commit_message,
                'raw_message': commit_message,
                'subject': subject,
                'author': self._get_commit_author(commit_hash),
                'date': self._get_commit_date(commit_hash),
                'pr_number': self._extract_pr_number(commit_message),
                'issues': self._extract_issue_numbers(commit_message),
            }
    
    def _get_commit_author(self, commit_hash: str) -> str:
        """Get commit author name."""
        try:
            result = subprocess.run(
                ['git', 'show', '-s', '--format=%an', commit_hash],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return 'Unknown'
    
    def _get_commit_date(self, commit_hash: str) -> str:
        """Get commit date."""
        try:
            result = subprocess.run(
                ['git', 'show', '-s', '--format=%ci', commit_hash],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ''
    
    def _extract_pr_number(self, message: str) -> Optional[int]:
        """Extract PR number from commit message."""
        pr_pattern = re.compile(r'#(\d+)')
        match = pr_pattern.search(message)
        return int(match.group(1)) if match else None
    
    def _extract_issue_numbers(self, message: str) -> List[int]:
        """Extract issue numbers from commit message."""
        issue_patterns = [
            re.compile(r'(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?) #(\d+)', re.IGNORECASE),
            re.compile(r'issue #(\d+)', re.IGNORECASE)
        ]
        
        issues = []
        for pattern in issue_patterns:
            issues.extend([int(match.group(1)) for match in pattern.finditer(message)])
        
        return list(set(issues))

class GitRepository:
    """Interface to Git repository operations."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        
    def get_commits_since_tag(self, tag: str = None) -> List[Tuple[str, str]]:
        """Get commits since specified tag or all commits if no tag."""
        try:
            if tag:
                cmd = ['git', 'log', f'{tag}..HEAD', '--format=%H%n%B%n---COMMIT-END---']
            else:
                cmd = ['git', 'log', '--format=%H%n%B%n---COMMIT-END---']
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            commits = []
            commit_blocks = result.stdout.split('---COMMIT-END---')
            
            for block in commit_blocks:
                block = block.strip()
                if not block:
                    continue
                    
                lines = block.split('\n')
                if len(lines) < 2:
                    continue
                    
                commit_hash = lines[0]
                commit_message = '\n'.join(lines[1:])
                commits.append((commit_hash, commit_message))
            
            return commits
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error getting commits: {e}[/red]")
            return []
    
    def get_latest_tag(self) -> Optional[str]:
        """Get the latest git tag."""
        try:
            result = subprocess.run(
                ['git', 'describe', '--tags', '--abbrev=0'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def get_all_tags(self) -> List[str]:
        """Get all git tags sorted by version."""
        try:
            result = subprocess.run(
                ['git', 'tag', '--sort=-version:refname'],
                capture_output=True,
                text=True,
                check=True
            )
            return [tag.strip() for tag in result.stdout.split('\n') if tag.strip()]
        except subprocess.CalledProcessError:
            return []

class ChangelogGenerator:
    """Generate comprehensive changelogs with AI enhancement."""
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path('.changelogrc.json')
        self.config = self._load_config()
        self.parser = CommitParser()
        self.repo = GitRepository()
        
    def _load_config(self) -> Dict:
        """Load changelog configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        else:
            return {
                'types': COMMIT_TYPES,
                'header': '# Changelog\n\n',
                'compareUrlFormat': 'https://github.com/terragonlabs/testgen-copilot/compare/{{previousTag}}...{{currentTag}}',
                'commitUrlFormat': 'https://github.com/terragonlabs/testgen-copilot/commit/{{hash}}',
                'issueUrlFormat': 'https://github.com/terragonlabs/testgen-copilot/issues/{{id}}',
            }
    
    def generate_changelog(self, since_tag: str = None, output_file: Path = None, include_all: bool = False) -> str:
        """Generate comprehensive changelog."""
        output_file = output_file or Path('CHANGELOG.md')
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating changelog...", total=5)
            
            # Get commits
            progress.update(task, advance=1, description="Fetching commits...")
            commits = self.repo.get_commits_since_tag(since_tag)
            
            # Parse commits
            progress.update(task, advance=1, description="Parsing commits...")
            parsed_commits = []
            for commit_hash, commit_message in commits:
                parsed_commits.append(self.parser.parse_commit(commit_hash, commit_message))
            
            # Categorize commits
            progress.update(task, advance=1, description="Categorizing commits...")
            categorized = self._categorize_commits(parsed_commits)
            
            # Generate content
            progress.update(task, advance=1, description="Generating content...")
            content = self._generate_markdown(categorized, since_tag)
            
            # Write to file
            progress.update(task, advance=1, description="Writing changelog...")
            if output_file:
                self._write_changelog(content, output_file, include_all)
            
            progress.update(task, advance=1, description="Done!")
        
        console.print(f"[green]✅ Changelog generated: {output_file}[/green]")
        return content
    
    def _categorize_commits(self, commits: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize commits by type."""
        categorized = {}
        
        for commit in commits:
            commit_type = commit['type']
            
            # Skip merge commits and certain types
            if commit['description'].startswith('Merge ') or commit_type == 'chore':
                continue
            
            if commit_type not in categorized:
                categorized[commit_type] = []
            
            categorized[commit_type].append(commit)
        
        return categorized
    
    def _generate_markdown(self, categorized: Dict[str, List[Dict]], since_tag: str = None) -> str:
        """Generate markdown content for changelog."""
        content = []
        
        # Header
        current_tag = 'Unreleased'
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        content.append(f"## [{current_tag}] - {current_date}")
        content.append("")
        
        # Add comparison link if we have a previous tag
        if since_tag:
            compare_url = self.config.get('compareUrlFormat', '').replace('{{previousTag}}', since_tag).replace('{{currentTag}}', 'HEAD')
            content.append(f"[Compare changes]({compare_url})")
            content.append("")
        
        # Breaking changes first
        breaking_changes = []
        for commit_type, commits in categorized.items():
            for commit in commits:
                if commit['is_breaking']:
                    breaking_changes.append(commit)
        
        if breaking_changes:
            content.append("### 💥 BREAKING CHANGES")
            content.append("")
            for commit in breaking_changes:
                content.append(f"- **{commit['description']}** ({self._format_commit_hash(commit['hash'])})")
                if commit['body']:
                    # Extract breaking change details
                    body_lines = commit['body'].split('\n')
                    for line in body_lines:
                        if line.startswith('BREAKING CHANGE:'):
                            content.append(f"  - {line.replace('BREAKING CHANGE:', '').strip()}")
            content.append("")
        
        # Organize by type
        type_order = ['feat', 'fix', 'perf', 'security', 'docs', 'refactor', 'test', 'build', 'ci']
        
        for commit_type in type_order:
            if commit_type not in categorized:
                continue
                
            commits = categorized[commit_type]
            if not commits:
                continue
            
            type_info = COMMIT_TYPES.get(commit_type, {'section': commit_type.title(), 'emoji': '•'})
            content.append(f"### {type_info['section']}")
            content.append("")
            
            # Group by scope
            scoped_commits = {}
            unscoped_commits = []
            
            for commit in commits:
                if commit['scope']:
                    if commit['scope'] not in scoped_commits:
                        scoped_commits[commit['scope']] = []
                    scoped_commits[commit['scope']].append(commit)
                else:
                    unscoped_commits.append(commit)
            
            # Add scoped commits
            for scope in sorted(scoped_commits.keys()):
                content.append(f"#### {scope.title()}")
                content.append("")
                for commit in scoped_commits[scope]:
                    content.append(self._format_commit_line(commit))
                content.append("")
            
            # Add unscoped commits
            if unscoped_commits:
                for commit in unscoped_commits:
                    content.append(self._format_commit_line(commit))
                content.append("")
        
        # Statistics
        total_commits = sum(len(commits) for commits in categorized.values())
        total_breaking = len(breaking_changes)
        
        content.append("---")
        content.append("")
        content.append(f"**Summary:** {total_commits} commits")
        if total_breaking > 0:
            content.append(f"**Breaking Changes:** {total_breaking}")
        content.append("")
        content.append("*Generated with TestGen Copilot Changelog Generator*")
        
        return '\n'.join(content)
    
    def _format_commit_line(self, commit: Dict) -> str:
        """Format a single commit line."""
        line = f"- {commit['description']}"
        
        # Add commit hash link
        if commit['hash']:
            line += f" ({self._format_commit_hash(commit['hash'])})"
        
        # Add PR link
        if commit['pr_number']:
            pr_url = f"https://github.com/terragonlabs/testgen-copilot/pull/{commit['pr_number']}"
            line += f" [#{commit['pr_number']}]({pr_url})"
        
        # Add issue references
        if commit['issues']:
            for issue in commit['issues']:
                issue_url = self.config.get('issueUrlFormat', '').replace('{{id}}', str(issue))
                line += f" closes [#{issue}]({issue_url})"
        
        # Add author if available
        if commit['author'] and commit['author'] != 'Unknown':
            line += f" by @{commit['author']}"
        
        return line
    
    def _format_commit_hash(self, commit_hash: str) -> str:
        """Format commit hash with link."""
        short_hash = commit_hash[:7]
        commit_url = self.config.get('commitUrlFormat', '').replace('{{hash}}', commit_hash)
        return f"[{short_hash}]({commit_url})"
    
    def _write_changelog(self, content: str, output_file: Path, include_all: bool):
        """Write changelog to file."""
        if include_all and output_file.exists():
            # Prepend to existing changelog
            existing_content = output_file.read_text()
            
            # Find where to insert new content
            lines = existing_content.split('\n')
            header_end = 0
            
            for i, line in enumerate(lines):
                if line.startswith('## ') and 'Unreleased' not in line:
                    header_end = i
                    break
            
            if header_end > 0:
                new_content = '\n'.join(lines[:header_end]) + '\n\n' + content + '\n\n' + '\n'.join(lines[header_end:])
            else:
                new_content = self.config.get('header', '# Changelog\n\n') + content + '\n\n' + existing_content
            
            output_file.write_text(new_content)
        else:
            # Write complete new changelog
            full_content = self.config.get('header', '# Changelog\n\n') + content
            output_file.write_text(full_content)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive changelogs for TestGen Copilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate changelog since last tag:
    python changelog-generator.py

  Generate changelog since specific tag:
    python changelog-generator.py --since v1.0.0

  Generate and append to existing changelog:
    python changelog-generator.py --output CHANGELOG.md --include-all

  Generate changelog for specific range:
    python changelog-generator.py --since v1.0.0 --output RELEASE-NOTES.md
        """
    )
    
    parser.add_argument(
        '--since',
        help='Generate changelog since this tag (default: latest tag)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('CHANGELOG.md'),
        help='Output file path (default: CHANGELOG.md)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('.changelogrc.json'),
        help='Configuration file path (default: .changelogrc.json)'
    )
    
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Include all existing changelog content'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print changelog to stdout instead of writing to file'
    )
    
    args = parser.parse_args()
    
    try:
        generator = ChangelogGenerator(args.config)
        
        if args.dry_run:
            content = generator.generate_changelog(args.since, None, args.include_all)
            console.print(content)
        else:
            generator.generate_changelog(args.since, args.output, args.include_all)
            
    except Exception as e:
        console.print(f"[red]Error generating changelog: {e}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    main()