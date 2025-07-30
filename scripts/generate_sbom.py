#!/usr/bin/env python3
"""
SBOM (Software Bill of Materials) Generation Script
Generates comprehensive SBOM for security and compliance tracking
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import click


def run_command(cmd: List[str]) -> str:
    """Run command and return output."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        click.echo(f"Error running {' '.join(cmd)}: {e}", err=True)
        return ""


def generate_cyclonedx_sbom(output_path: Path) -> bool:
    """Generate SBOM using CycloneDx."""
    try:
        cmd = [
            "cyclonedx-py",
            "-o", str(output_path),
            "--format", "json",
            "--requirements", "requirements.txt",
            "--dev-requirements", "requirements-dev.txt"
        ]
        run_command(cmd)
        return True
    except Exception as e:
        click.echo(f"CycloneDx SBOM generation failed: {e}", err=True)
        return False


def generate_pip_audit_sbom(output_path: Path) -> bool:
    """Generate SBOM using pip-audit."""
    try:
        cmd = ["pip-audit", "--format=json", "--output", str(output_path)]
        run_command(cmd)
        return True
    except Exception as e:
        click.echo(f"pip-audit SBOM generation failed: {e}", err=True)
        return False


def generate_custom_sbom() -> Dict[str, Any]:
    """Generate custom SBOM with additional metadata."""
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:{datetime.now().isoformat()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "tools": [
                {
                    "vendor": "Terragon Labs",
                    "name": "TestGen Copilot SBOM Generator",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "application",
                "name": "testgen-copilot",
                "version": "0.0.1",
                "description": "CLI tool for automated test generation",
                "licenses": [{"license": {"id": "MIT"}}]
            }
        },
        "components": [],
        "vulnerabilities": [],
        "dependencies": []
    }


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./sbom",
    help="Output directory for SBOM files"
)
@click.option(
    "--format",
    type=click.Choice(["json", "xml", "all"]),
    default="all",
    help="Output format for SBOM"
)
def main(output_dir: str, format: str):
    """Generate Software Bill of Materials (SBOM) for the project."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    click.echo("🔍 Generating Software Bill of Materials (SBOM)...")
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    success_count = 0
    total_attempts = 0
    
    # CycloneDx SBOM
    if format in ["json", "all"]:
        total_attempts += 1
        cyclone_path = output_path / f"sbom_cyclonedx_{timestamp}.json"
        if generate_cyclonedx_sbom(cyclone_path):
            click.echo(f"✅ CycloneDx SBOM generated: {cyclone_path}")
            success_count += 1
        else:
            click.echo("❌ CycloneDx SBOM generation failed")
    
    # pip-audit SBOM
    total_attempts += 1
    audit_path = output_path / f"sbom_pip_audit_{timestamp}.json"
    if generate_pip_audit_sbom(audit_path):
        click.echo(f"✅ pip-audit SBOM generated: {audit_path}")
        success_count += 1
    else:
        click.echo("❌ pip-audit SBOM generation failed")
    
    # Custom SBOM
    total_attempts += 1
    custom_sbom = generate_custom_sbom()
    custom_path = output_path / f"sbom_custom_{timestamp}.json"
    try:
        with open(custom_path, 'w') as f:
            json.dump(custom_sbom, f, indent=2)
        click.echo(f"✅ Custom SBOM generated: {custom_path}")
        success_count += 1
    except Exception as e:
        click.echo(f"❌ Custom SBOM generation failed: {e}")
    
    # Summary
    click.echo(f"\n📊 SBOM Generation Summary:")
    click.echo(f"   Successful: {success_count}/{total_attempts}")
    
    if success_count > 0:
        click.echo(f"   Output directory: {output_path}")
        click.echo("   Files generated:")
        for file in output_path.glob(f"*{timestamp}*"):
            click.echo(f"     - {file.name}")
    
    sys.exit(0 if success_count > 0 else 1)


if __name__ == "__main__":
    main()