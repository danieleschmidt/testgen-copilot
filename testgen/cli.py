import sys
import argparse
from .stubgen import generate_stubs
from .vulnscan import scan_file, format_report
from .analyzer import find_untested


def main():
    parser = argparse.ArgumentParser(description="Test generator and vulnerability scanner")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("gen", help="Generate test stubs for a Python file")
    gen.add_argument("file", help="Python source file")
    gen.add_argument("--module", help="Module import name")

    scan = sub.add_parser("scan", help="Scan for security vulnerabilities")
    scan.add_argument("file", nargs="+", help="Python file(s) to scan")

    untested = sub.add_parser("untested", help="List untested functions")
    untested.add_argument("source", help="Source file")
    untested.add_argument("--tests", help="Test file to compare against")

    args = parser.parse_args()

    if args.command == "gen":
        print(generate_stubs(args.file, args.module))
    elif args.command == "scan":
        all_findings = []
        for f in args.file:
            all_findings.extend(scan_file(f))
        print(format_report(all_findings))
    elif args.command == "untested":
        funcs = find_untested(args.source, args.tests)
        if not funcs:
            print("All functions appear to be tested.")
        else:
            print(f"Untested functions ({len(funcs)}):")
            for f in funcs:
                print(f"  Line {f['lineno']}: {f['name']}({', '.join(f['args'])})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
