#!/usr/bin/env python3
"""
Test runner script for the RAG Document Q&A system.
Provides different test execution modes and reporting options.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional
import json
import time

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print colored header."""
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")


def print_section(text: str):
    """Print section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-'*40}{Colors.ENDC}")


def run_command(cmd: List[str], description: str = "") -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    if description:
        print(f"{Colors.CYAN}Running: {description}{Colors.ENDC}")
    
    print(f"{Colors.YELLOW}Command: {' '.join(cmd)}{Colors.ENDC}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out after 10 minutes"
    except Exception as e:
        return 1, "", str(e)


def run_unit_tests(verbose: bool = False, coverage: bool = True) -> bool:
    """Run unit tests."""
    print_section("Running Unit Tests")
    
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html:htmlcov"])
    
    # Add markers to skip slow tests by default
    cmd.extend(["-m", "not slow"])
    
    exit_code, stdout, stderr = run_command(cmd, "Unit tests with coverage")
    
    if stdout:
        print(stdout)
    if stderr and exit_code != 0:
        print(f"{Colors.RED}STDERR: {stderr}{Colors.ENDC}")
    
    if exit_code == 0:
        print(f"{Colors.GREEN}✅ Unit tests passed!{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.RED}❌ Unit tests failed!{Colors.ENDC}")
        return False


def run_integration_tests(verbose: bool = False) -> bool:
    """Run integration tests."""
    print_section("Running Integration Tests")
    
    cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    
    exit_code, stdout, stderr = run_command(cmd, "Integration tests")
    
    if stdout:
        print(stdout)
    if stderr and exit_code != 0:
        print(f"{Colors.RED}STDERR: {stderr}{Colors.ENDC}")
    
    if exit_code == 0:
        print(f"{Colors.GREEN}✅ Integration tests passed!{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.RED}❌ Integration tests failed!{Colors.ENDC}")
        return False


def run_specific_test(test_path: str, verbose: bool = False) -> bool:
    """Run a specific test file or test."""
    print_section(f"Running Specific Test: {test_path}")
    
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    exit_code, stdout, stderr = run_command(cmd, f"Test: {test_path}")
    
    if stdout:
        print(stdout)
    if stderr and exit_code != 0:
        print(f"{Colors.RED}STDERR: {stderr}{Colors.ENDC}")
    
    return exit_code == 0


def run_coverage_report() -> bool:
    """Generate and display coverage report."""
    print_section("Generating Coverage Report")
    
    # Run tests with coverage
    cmd = [
        "python", "-m", "pytest", 
        "--cov=src", 
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml",
        "tests/unit/"
    ]
    
    exit_code, stdout, stderr = run_command(cmd, "Coverage analysis")
    
    if stdout:
        print(stdout)
    
    if exit_code == 0:
        print(f"\n{Colors.GREEN}✅ Coverage report generated!{Colors.ENDC}")
        print(f"{Colors.CYAN}HTML report available at: htmlcov/index.html{Colors.ENDC}")
        print(f"{Colors.CYAN}XML report available at: coverage.xml{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.RED}❌ Coverage report generation failed!{Colors.ENDC}")
        return False


def run_lint_checks() -> bool:
    """Run code linting checks."""
    print_section("Running Code Quality Checks")
    
    # Check if flake8 is available
    try:
        subprocess.run(["flake8", "--version"], capture_output=True, check=True)
        flake8_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        flake8_available = False
    
    success = True
    
    if flake8_available:
        cmd = ["flake8", "src/", "tests/", "--max-line-length=100", "--ignore=E203,W503"]
        exit_code, stdout, stderr = run_command(cmd, "Flake8 linting")
        
        if exit_code == 0:
            print(f"{Colors.GREEN}✅ Flake8 checks passed!{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ Flake8 checks failed!{Colors.ENDC}")
            if stdout:
                print(stdout)
            success = False
    else:
        print(f"{Colors.YELLOW}⚠️ Flake8 not available, skipping lint checks{Colors.ENDC}")
    
    return success


def check_test_environment() -> bool:
    """Check if test environment is properly set up."""
    print_section("Checking Test Environment")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print(f"{Colors.RED}❌ Python 3.8+ required{Colors.ENDC}")
        return False
    
    # Check required packages
    required_packages = ["pytest", "pytest-cov", "langchain"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"{Colors.GREEN}✅ {package} available{Colors.ENDC}")
        except ImportError:
            print(f"{Colors.RED}❌ {package} missing{Colors.ENDC}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n{Colors.YELLOW}Install missing packages with:{Colors.ENDC}")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Check test directories
    test_dirs = ["tests", "tests/unit"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"{Colors.GREEN}✅ {test_dir}/ directory exists{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ {test_dir}/ directory missing{Colors.ENDC}")
            return False
    
    return True


def generate_test_report(results: dict) -> None:
    """Generate a JSON test report."""
    print_section("Generating Test Report")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "summary": {
            "total_suites": len(results),
            "passed_suites": sum(1 for r in results.values() if r),
            "failed_suites": sum(1 for r in results.values() if not r)
        }
    }
    
    report_file = Path("test_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Test report saved to: {report_file}")
    
    # Print summary
    summary = report["summary"]
    print(f"\n{Colors.BOLD}Test Summary:{Colors.ENDC}")
    print(f"  Total test suites: {summary['total_suites']}")
    print(f"  Passed: {Colors.GREEN}{summary['passed_suites']}{Colors.ENDC}")
    print(f"  Failed: {Colors.RED}{summary['failed_suites']}{Colors.ENDC}")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="RAG Document Q&A Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--lint", action="store_true", help="Run lint checks")
    parser.add_argument("--check-env", action="store_true", help="Check test environment")
    parser.add_argument("--test", type=str, help="Run specific test file/path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage analysis")
    
    args = parser.parse_args()
    
    print_header("RAG Document Q&A Test Runner")
    
    if args.check_env or not any([args.unit, args.integration, args.all, args.coverage, args.lint, args.test]):
        if not check_test_environment():
            sys.exit(1)
        if not any([args.unit, args.integration, args.all, args.coverage, args.lint, args.test]):
            return
    
    results = {}
    
    if args.lint:
        results["lint"] = run_lint_checks()
    
    if args.test:
        results["specific"] = run_specific_test(args.test, args.verbose)
    elif args.unit:
        results["unit"] = run_unit_tests(args.verbose, not args.no_coverage)
    elif args.integration:
        results["integration"] = run_integration_tests(args.verbose)
    elif args.all:
        results["unit"] = run_unit_tests(args.verbose, not args.no_coverage)
        # Only run integration if unit tests pass
        if results["unit"]:
            results["integration"] = run_integration_tests(args.verbose)
    elif args.coverage:
        results["coverage"] = run_coverage_report()
    
    if results:
        generate_test_report(results)
        
        # Exit with error code if any tests failed
        if not all(results.values()):
            print(f"\n{Colors.RED}Some tests failed!{Colors.ENDC}")
            sys.exit(1)
        else:
            print(f"\n{Colors.GREEN}All tests passed!{Colors.ENDC}")


if __name__ == "__main__":
    main()