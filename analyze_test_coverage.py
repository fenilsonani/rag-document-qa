"""
Test Coverage Analysis for Enterprise RAG Platform
Analyzes current test coverage and provides detailed reporting.
"""

import os
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class ModuleInfo:
    """Information about a Python module."""
    path: str
    name: str
    functions: List[str]
    classes: List[str]
    lines_of_code: int
    complexity_score: int


@dataclass
class TestInfo:
    """Information about test files."""
    path: str
    name: str
    test_functions: List[str]
    test_classes: List[str]
    covers_modules: Set[str]
    lines_of_code: int


@dataclass
class CoverageReport:
    """Complete test coverage report."""
    total_modules: int
    total_test_files: int
    covered_modules: int
    uncovered_modules: int
    coverage_percentage: float
    function_coverage: Dict[str, float]
    class_coverage: Dict[str, float]
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]


class TestCoverageAnalyzer:
    """Analyzes test coverage for the RAG platform codebase."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.source_modules: Dict[str, ModuleInfo] = {}
        self.test_files: Dict[str, TestInfo] = {}
        self.coverage_data: Dict[str, Any] = {}
        
        # Patterns to identify source code vs test files
        self.test_patterns = ['test_', '_test', 'tests/', 'conftest.py']
        self.ignore_patterns = [
            'venv/', '__pycache__/', '.git/', '.pytest_cache/',
            'build/', 'dist/', '*.egg-info/', 'node_modules/',
            'vendor/', 'third_party/'
        ]
    
    def analyze_coverage(self) -> CoverageReport:
        """Analyze test coverage comprehensively."""
        
        print("🔍 Analyzing Test Coverage for Enterprise RAG Platform")
        print("=" * 60)
        
        # Step 1: Discover and analyze source modules
        print("📁 Discovering source modules...")
        self._discover_source_modules()
        
        # Step 2: Discover and analyze test files  
        print("🧪 Discovering test files...")
        self._discover_test_files()
        
        # Step 3: Analyze coverage relationships
        print("🔗 Analyzing coverage relationships...")
        self._analyze_coverage_relationships()
        
        # Step 4: Calculate coverage metrics
        print("📊 Calculating coverage metrics...")
        report = self._calculate_coverage_metrics()
        
        # Step 5: Generate recommendations
        print("💡 Generating recommendations...")
        report.recommendations = self._generate_recommendations()
        
        return report
    
    def _discover_source_modules(self):
        """Discover all source code modules."""
        
        for py_file in self.project_root.rglob("*.py"):
            # Skip if in ignored patterns
            if self._should_ignore_path(py_file):
                continue
            
            # Skip test files
            if self._is_test_file(py_file):
                continue
            
            try:
                module_info = self._analyze_module(py_file)
                if module_info:
                    self.source_modules[module_info.name] = module_info
                    
            except Exception as e:
                print(f"   Warning: Could not analyze {py_file}: {e}")
        
        print(f"   Found {len(self.source_modules)} source modules")
    
    def _discover_test_files(self):
        """Discover all test files."""
        
        for py_file in self.project_root.rglob("*.py"):
            # Skip if in ignored patterns
            if self._should_ignore_path(py_file):
                continue
            
            # Only include test files
            if not self._is_test_file(py_file):
                continue
            
            try:
                test_info = self._analyze_test_file(py_file)
                if test_info:
                    self.test_files[test_info.name] = test_info
                    
            except Exception as e:
                print(f"   Warning: Could not analyze {py_file}: {e}")
        
        print(f"   Found {len(self.test_files)} test files")
    
    def _analyze_module(self, file_path: Path) -> ModuleInfo:
        """Analyze a source code module."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            functions = []
            classes = []
            complexity = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                    complexity += self._calculate_complexity(node)
                elif isinstance(node, ast.AsyncFunctionDef):
                    functions.append(node.name)
                    complexity += self._calculate_complexity(node)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    # Add methods to functions list
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            functions.append(f"{node.name}.{item.name}")
            
            lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
            
            module_name = str(file_path.relative_to(self.project_root)).replace('/', '.').replace('.py', '')
            
            return ModuleInfo(
                path=str(file_path),
                name=module_name,
                functions=functions,
                classes=classes,
                lines_of_code=lines_of_code,
                complexity_score=complexity
            )
            
        except Exception as e:
            print(f"   Error analyzing {file_path}: {e}")
            return None
    
    def _analyze_test_file(self, file_path: Path) -> TestInfo:
        """Analyze a test file."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            test_functions = []
            test_classes = []
            covers_modules = set()
            
            # Find test functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_') or '_test' in node.name:
                        test_functions.append(node.name)
                elif isinstance(node, ast.AsyncFunctionDef):
                    if node.name.startswith('test_') or '_test' in node.name:
                        test_functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    if 'test' in node.name.lower():
                        test_classes.append(node.name)
                        # Add test methods
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if item.name.startswith('test_') or '_test' in item.name:
                                    test_functions.append(f"{node.name}.{item.name}")
            
            # Try to determine what modules this test file covers
            covers_modules = self._infer_covered_modules(file_path, content)
            
            lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
            
            test_name = str(file_path.relative_to(self.project_root)).replace('/', '.').replace('.py', '')
            
            return TestInfo(
                path=str(file_path),
                name=test_name,
                test_functions=test_functions,
                test_classes=test_classes,
                covers_modules=covers_modules,
                lines_of_code=lines_of_code
            )
            
        except Exception as e:
            print(f"   Error analyzing test file {file_path}: {e}")
            return None
    
    def _infer_covered_modules(self, test_file: Path, content: str) -> Set[str]:
        """Infer which modules a test file covers based on imports and naming."""
        
        covered = set()
        
        # Parse imports
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not alias.name.startswith('_'):  # Skip private imports
                            covered.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and not node.module.startswith('_'):
                        covered.add(node.module)
        except:
            pass
        
        # Infer from file name
        test_name = test_file.stem
        if test_name.startswith('test_'):
            module_name = test_name[5:]  # Remove 'test_' prefix
            covered.add(module_name)
        
        # Filter to only include modules that exist in our source code
        existing_modules = set(self.source_modules.keys())
        covered = {mod for mod in covered if any(mod in existing_mod for existing_mod in existing_modules)}
        
        return covered
    
    def _analyze_coverage_relationships(self):
        """Analyze relationships between tests and source modules."""
        
        self.coverage_data = {
            'covered_modules': set(),
            'uncovered_modules': set(),
            'partially_covered_modules': set(),
            'test_to_module_mapping': {},
            'module_to_test_mapping': {}
        }
        
        # Map tests to modules
        for test_name, test_info in self.test_files.items():
            self.coverage_data['test_to_module_mapping'][test_name] = list(test_info.covers_modules)
            
            for module in test_info.covers_modules:
                if module not in self.coverage_data['module_to_test_mapping']:
                    self.coverage_data['module_to_test_mapping'][module] = []
                self.coverage_data['module_to_test_mapping'][module].append(test_name)
        
        # Categorize modules by coverage
        for module_name in self.source_modules:
            if module_name in self.coverage_data['module_to_test_mapping']:
                self.coverage_data['covered_modules'].add(module_name)
            else:
                # Check if partially covered (similar names)
                partial_coverage = False
                for test_module in self.coverage_data['module_to_test_mapping']:
                    if (module_name in test_module or test_module in module_name or
                        self._names_similar(module_name, test_module)):
                        self.coverage_data['partially_covered_modules'].add(module_name)
                        partial_coverage = True
                        break
                
                if not partial_coverage:
                    self.coverage_data['uncovered_modules'].add(module_name)
    
    def _calculate_coverage_metrics(self) -> CoverageReport:
        """Calculate comprehensive coverage metrics."""
        
        total_modules = len(self.source_modules)
        covered_modules = len(self.coverage_data['covered_modules'])
        partially_covered = len(self.coverage_data['partially_covered_modules'])
        uncovered_modules = len(self.coverage_data['uncovered_modules'])
        
        # Adjust coverage calculation to include partial coverage
        effective_coverage = covered_modules + (partially_covered * 0.3)  # 30% credit for partial coverage
        coverage_percentage = (effective_coverage / total_modules) * 100 if total_modules > 0 else 0
        
        # Calculate function and class coverage
        function_coverage = self._calculate_function_coverage()
        class_coverage = self._calculate_class_coverage()
        
        # Detailed analysis
        detailed_analysis = {
            "module_breakdown": {
                "total_source_files": total_modules,
                "total_test_files": len(self.test_files),
                "fully_covered": covered_modules,
                "partially_covered": partially_covered,
                "uncovered": uncovered_modules
            },
            "lines_of_code": {
                "source_lines": sum(module.lines_of_code for module in self.source_modules.values()),
                "test_lines": sum(test.lines_of_code for test in self.test_files.values()),
                "test_to_source_ratio": 0
            },
            "complexity_analysis": {
                "high_complexity_modules": [],
                "untested_high_complexity": [],
                "avg_complexity": sum(module.complexity_score for module in self.source_modules.values()) / len(self.source_modules) if self.source_modules else 0
            },
            "test_quality": {
                "total_test_functions": sum(len(test.test_functions) for test in self.test_files.values()),
                "total_test_classes": sum(len(test.test_classes) for test in self.test_files.values()),
                "avg_tests_per_module": 0
            }
        }
        
        # Calculate ratios
        if detailed_analysis["lines_of_code"]["source_lines"] > 0:
            detailed_analysis["lines_of_code"]["test_to_source_ratio"] = (
                detailed_analysis["lines_of_code"]["test_lines"] / 
                detailed_analysis["lines_of_code"]["source_lines"]
            )
        
        if total_modules > 0:
            detailed_analysis["test_quality"]["avg_tests_per_module"] = (
                detailed_analysis["test_quality"]["total_test_functions"] / total_modules
            )
        
        # Identify high complexity modules
        for module_name, module_info in self.source_modules.items():
            if module_info.complexity_score > 10:  # Arbitrary threshold
                detailed_analysis["complexity_analysis"]["high_complexity_modules"].append({
                    "name": module_name,
                    "complexity": module_info.complexity_score,
                    "functions": len(module_info.functions)
                })
                
                if module_name in self.coverage_data['uncovered_modules']:
                    detailed_analysis["complexity_analysis"]["untested_high_complexity"].append(module_name)
        
        return CoverageReport(
            total_modules=total_modules,
            total_test_files=len(self.test_files),
            covered_modules=covered_modules,
            uncovered_modules=uncovered_modules,
            coverage_percentage=coverage_percentage,
            function_coverage=function_coverage,
            class_coverage=class_coverage,
            detailed_analysis=detailed_analysis,
            recommendations=[]  # Will be filled by generate_recommendations
        )
    
    def _calculate_function_coverage(self) -> Dict[str, float]:
        """Calculate function-level coverage."""
        
        function_coverage = {}
        
        for module_name, module_info in self.source_modules.items():
            total_functions = len(module_info.functions)
            if total_functions == 0:
                function_coverage[module_name] = 100.0  # No functions to test
                continue
            
            # Count tested functions (simplified heuristic)
            tested_functions = 0
            if module_name in self.coverage_data['module_to_test_mapping']:
                # Each test file that covers this module gets credit
                test_files = self.coverage_data['module_to_test_mapping'][module_name]
                for test_file in test_files:
                    test_info = self.test_files.get(test_file)
                    if test_info:
                        # Heuristic: assume test file covers proportional functions
                        tested_functions += min(len(test_info.test_functions), total_functions)
            
            coverage = min(100.0, (tested_functions / total_functions) * 100)
            function_coverage[module_name] = coverage
        
        return function_coverage
    
    def _calculate_class_coverage(self) -> Dict[str, float]:
        """Calculate class-level coverage."""
        
        class_coverage = {}
        
        for module_name, module_info in self.source_modules.items():
            total_classes = len(module_info.classes)
            if total_classes == 0:
                class_coverage[module_name] = 100.0  # No classes to test
                continue
            
            # Count tested classes (simplified heuristic)
            tested_classes = 0
            if module_name in self.coverage_data['module_to_test_mapping']:
                test_files = self.coverage_data['module_to_test_mapping'][module_name]
                for test_file in test_files:
                    test_info = self.test_files.get(test_file)
                    if test_info:
                        tested_classes += min(len(test_info.test_classes), total_classes)
            
            coverage = min(100.0, (tested_classes / total_classes) * 100)
            class_coverage[module_name] = coverage
        
        return class_coverage
    
    def _generate_recommendations(self) -> List[str]:
        """Generate testing recommendations."""
        
        recommendations = []
        
        # Coverage-based recommendations
        if self.coverage_data['uncovered_modules']:
            recommendations.append(f"🔴 PRIORITY: Create unit tests for {len(self.coverage_data['uncovered_modules'])} completely untested modules")
            
            # List top priority untested modules
            high_priority = []
            for module in list(self.coverage_data['uncovered_modules'])[:5]:
                module_info = self.source_modules.get(module)
                if module_info and (len(module_info.functions) > 5 or module_info.complexity_score > 8):
                    high_priority.append(module)
            
            if high_priority:
                recommendations.append(f"⚡ Urgent: Focus on {', '.join(high_priority)} (high complexity/function count)")
        
        # Test quality recommendations
        total_source_lines = sum(module.lines_of_code for module in self.source_modules.values())
        total_test_lines = sum(test.lines_of_code for test in self.test_files.values())
        test_ratio = total_test_lines / total_source_lines if total_source_lines > 0 else 0
        
        if test_ratio < 0.3:
            recommendations.append(f"📏 Increase test coverage: Current test-to-source ratio is {test_ratio:.1%}, aim for 30%+")
        
        # Integration testing recommendations
        integration_tests = [test for test in self.test_files.values() if 'integration' in test.name.lower()]
        if len(integration_tests) < 3:
            recommendations.append("🔗 Add more integration tests to validate component interactions")
        
        # Enterprise testing recommendations
        enterprise_tests = [test for test in self.test_files.values() if any(keyword in test.name.lower() for keyword in ['enterprise', 'e2e', 'end-to-end'])]
        if len(enterprise_tests) < 2:
            recommendations.append("🏢 Implement comprehensive end-to-end testing for enterprise workflows")
        
        # Specific module recommendations
        critical_modules = ['rag_chain', 'document_loader', 'vector_store', 'query_intelligence']
        untested_critical = [mod for mod in critical_modules if mod in self.coverage_data['uncovered_modules']]
        if untested_critical:
            recommendations.append(f"⚠️ Critical modules need testing: {', '.join(untested_critical)}")
        
        # Performance testing recommendations
        perf_tests = [test for test in self.test_files.values() if 'performance' in test.name.lower() or 'benchmark' in test.name.lower()]
        if len(perf_tests) == 0:
            recommendations.append("⚡ Add performance benchmarks to validate response time and throughput claims")
        
        # Security testing recommendations  
        security_tests = [test for test in self.test_files.values() if 'security' in test.name.lower()]
        if len(security_tests) == 0:
            recommendations.append("🔐 Implement security testing for authentication, authorization, and input validation")
        
        if not recommendations:
            recommendations.append("✅ Test coverage is comprehensive! Continue maintaining high standards.")
        
        return recommendations
    
    def _should_ignore_path(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        path_str = str(path)
        return any(pattern in path_str for pattern in self.ignore_patterns)
    
    def _is_test_file(self, path: Path) -> bool:
        """Check if a file is a test file."""
        path_str = str(path).lower()
        return any(pattern in path_str for pattern in self.test_patterns)
    
    def _names_similar(self, name1: str, name2: str) -> bool:
        """Check if two module names are similar."""
        name1_parts = name1.lower().split('.')
        name2_parts = name2.lower().split('.')
        
        # Check if any part of one name is in the other
        for part1 in name1_parts:
            for part2 in name2_parts:
                if part1 in part2 or part2 in part1:
                    if len(part1) > 3 and len(part2) > 3:  # Avoid false positives with short names
                        return True
        return False
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.BoolOp, ast.Compare)):
                complexity += 1
        
        return complexity
    
    def print_coverage_report(self, report: CoverageReport):
        """Print a formatted coverage report."""
        
        print("\n" + "="*80)
        print("📊 ENTERPRISE RAG PLATFORM - TEST COVERAGE REPORT")
        print("="*80)
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Overall coverage
        print(f"\n📈 OVERALL TEST COVERAGE:")
        print(f"   • Coverage Percentage: {report.coverage_percentage:.1f}%")
        print(f"   • Total Source Modules: {report.total_modules}")
        print(f"   • Total Test Files: {report.total_test_files}")
        print(f"   • Fully Covered Modules: {report.covered_modules}")
        print(f"   • Uncovered Modules: {report.uncovered_modules}")
        
        # Coverage status
        if report.coverage_percentage >= 80:
            status = "✅ EXCELLENT"
        elif report.coverage_percentage >= 60:
            status = "⚠️ GOOD"  
        elif report.coverage_percentage >= 40:
            status = "🟡 FAIR"
        else:
            status = "🔴 POOR"
        
        print(f"   • Coverage Status: {status}")
        
        # Detailed breakdown
        analysis = report.detailed_analysis
        print(f"\n📋 DETAILED ANALYSIS:")
        print(f"   • Source Lines of Code: {analysis['lines_of_code']['source_lines']:,}")
        print(f"   • Test Lines of Code: {analysis['lines_of_code']['test_lines']:,}")
        print(f"   • Test-to-Source Ratio: {analysis['lines_of_code']['test_to_source_ratio']:.1%}")
        print(f"   • Total Test Functions: {analysis['test_quality']['total_test_functions']}")
        print(f"   • Average Tests per Module: {analysis['test_quality']['avg_tests_per_module']:.1f}")
        
        # Module breakdown
        breakdown = analysis['module_breakdown']
        print(f"\n🗂️ MODULE BREAKDOWN:")
        print(f"   • Fully Covered: {breakdown['fully_covered']} modules")
        print(f"   • Partially Covered: {breakdown['partially_covered']} modules")
        print(f"   • Uncovered: {breakdown['uncovered']} modules")
        
        # High complexity modules
        high_complexity = analysis['complexity_analysis']['high_complexity_modules']
        if high_complexity:
            print(f"\n⚠️ HIGH COMPLEXITY MODULES (>10 complexity score):")
            for module in high_complexity[:5]:  # Show top 5
                print(f"   • {module['name']}: {module['complexity']} complexity, {module['functions']} functions")
        
        # Uncovered high complexity modules
        untested_complex = analysis['complexity_analysis']['untested_high_complexity']
        if untested_complex:
            print(f"\n🚨 UNTESTED HIGH COMPLEXITY MODULES:")
            for module in untested_complex[:5]:
                print(f"   • {module}")
        
        # Function and class coverage samples
        if report.function_coverage:
            print(f"\n🔧 FUNCTION COVERAGE (sample):")
            sample_modules = list(report.function_coverage.items())[:5]
            for module, coverage in sample_modules:
                print(f"   • {module}: {coverage:.1f}%")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"   • {rec}")
        
        # Test categories present
        test_categories = self._analyze_test_categories()
        print(f"\n🧪 TEST CATEGORIES PRESENT:")
        for category, count in test_categories.items():
            print(f"   • {category}: {count} files")
        
        # Overall assessment
        print(f"\n" + "="*80)
        if report.coverage_percentage >= 70:
            print("🎉 TEST COVERAGE ASSESSMENT: GOOD")
            print("   The codebase has reasonable test coverage for an enterprise platform.")
        elif report.coverage_percentage >= 40:
            print("⚠️ TEST COVERAGE ASSESSMENT: NEEDS IMPROVEMENT")
            print("   Additional testing is needed for enterprise-grade reliability.")
        else:
            print("🚨 TEST COVERAGE ASSESSMENT: CRITICAL")
            print("   Comprehensive testing is required before production deployment.")
        
        print("="*80)
    
    def _analyze_test_categories(self) -> Dict[str, int]:
        """Analyze what categories of tests are present."""
        
        categories = {
            "Unit Tests": 0,
            "Integration Tests": 0,
            "End-to-End Tests": 0,
            "Performance Tests": 0,
            "Security Tests": 0,
            "Enterprise Validation": 0,
            "Format Tests": 0
        }
        
        for test_name, test_info in self.test_files.items():
            name_lower = test_name.lower()
            
            if 'integration' in name_lower:
                categories["Integration Tests"] += 1
            elif any(keyword in name_lower for keyword in ['e2e', 'end_to_end', 'enterprise']):
                categories["End-to-End Tests"] += 1
            elif 'performance' in name_lower or 'benchmark' in name_lower:
                categories["Performance Tests"] += 1
            elif 'security' in name_lower:
                categories["Security Tests"] += 1
            elif 'enterprise' in name_lower or 'validation' in name_lower:
                categories["Enterprise Validation"] += 1
            elif 'format' in name_lower or 'pdf' in name_lower or 'multimodal' in name_lower:
                categories["Format Tests"] += 1
            else:
                categories["Unit Tests"] += 1
        
        return categories
    
    def save_coverage_report(self, report: CoverageReport, filename: str = "test_coverage_report.json"):
        """Save coverage report to JSON file."""
        
        # Convert sets to lists for JSON serialization
        report_dict = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_modules": report.total_modules,
                "total_test_files": report.total_test_files,
                "covered_modules": report.covered_modules,
                "uncovered_modules": report.uncovered_modules,
                "coverage_percentage": report.coverage_percentage
            },
            "function_coverage": report.function_coverage,
            "class_coverage": report.class_coverage,
            "detailed_analysis": report.detailed_analysis,
            "recommendations": report.recommendations,
            "uncovered_module_list": list(self.coverage_data['uncovered_modules']),
            "covered_module_list": list(self.coverage_data['covered_modules']),
            "test_categories": self._analyze_test_categories(),
            "module_details": {
                name: {
                    "path": info.path,
                    "functions": info.functions,
                    "classes": info.classes,
                    "lines_of_code": info.lines_of_code,
                    "complexity_score": info.complexity_score
                }
                for name, info in self.source_modules.items()
            },
            "test_details": {
                name: {
                    "path": info.path,
                    "test_functions": info.test_functions,
                    "test_classes": info.test_classes,
                    "covers_modules": list(info.covers_modules),
                    "lines_of_code": info.lines_of_code
                }
                for name, info in self.test_files.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"\n📄 Detailed coverage report saved to: {filename}")


def main():
    """Run test coverage analysis."""
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    analyzer = TestCoverageAnalyzer(project_root)
    
    try:
        # Run coverage analysis
        report = analyzer.analyze_coverage()
        
        # Print the report
        analyzer.print_coverage_report(report)
        
        # Save detailed report
        analyzer.save_coverage_report(report)
        
        # Return success based on coverage
        return report.coverage_percentage >= 40  # Minimum acceptable coverage
        
    except Exception as e:
        print(f"❌ Coverage analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)