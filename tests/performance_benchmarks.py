"""
Enterprise RAG Platform - Performance Benchmarking Suite
Validates all performance claims with comprehensive load testing.
"""

import asyncio
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import numpy as np
import psutil
import threading
from collections import defaultdict


@dataclass
class LoadTestConfig:
    """Load test configuration."""
    concurrent_users: int = 100
    total_requests: int = 1000
    ramp_up_duration: int = 30  # seconds
    test_duration: int = 300   # 5 minutes
    target_rps: int = 100
    endpoints: List[Dict[str, Any]] = None


@dataclass 
class PerformanceBenchmark:
    """Performance benchmark result."""
    test_name: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    requests_per_second: float
    error_rate: float
    throughput_mb_per_sec: float
    cpu_usage_percent: float
    memory_usage_mb: float
    
    def meets_sla(self) -> bool:
        """Check if benchmark meets SLA targets."""
        return (
            self.p95_response_time_ms < 200 and
            self.error_rate < 0.001 and  # 0.1%
            self.requests_per_second >= 100
        )


class PerformanceTester:
    """Comprehensive performance testing suite."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[PerformanceBenchmark] = []
        self.system_metrics: List[Dict[str, float]] = []
        
    async def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        print("🚀 Starting Comprehensive Performance Benchmarks")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Response Time Benchmark
        print("\n⚡ Running Response Time Benchmark...")
        response_benchmark = await self._benchmark_response_times()
        self.results.append(response_benchmark)
        
        # 2. Throughput Benchmark
        print("\n🔥 Running Throughput Benchmark...")
        throughput_benchmark = await self._benchmark_throughput()
        self.results.append(throughput_benchmark)
        
        # 3. Concurrency Benchmark
        print("\n👥 Running Concurrency Benchmark...")
        concurrency_benchmark = await self._benchmark_concurrency()
        self.results.append(concurrency_benchmark)
        
        # 4. Sustained Load Benchmark
        print("\n🏋️ Running Sustained Load Benchmark...")
        sustained_benchmark = await self._benchmark_sustained_load()
        self.results.append(sustained_benchmark)
        
        # 5. Stress Test Benchmark
        print("\n💪 Running Stress Test Benchmark...")
        stress_benchmark = await self._benchmark_stress_test()
        self.results.append(stress_benchmark)
        
        # 6. Memory Usage Benchmark
        print("\n🧠 Running Memory Usage Benchmark...")
        memory_benchmark = await self._benchmark_memory_usage()
        self.results.append(memory_benchmark)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        return self._generate_performance_report(total_time)
    
    async def _benchmark_response_times(self) -> PerformanceBenchmark:
        """Benchmark response times for single requests."""
        
        endpoint = f"{self.base_url}/health"
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._monitor_system_metrics(30))
        
        start_time = time.time()
        
        # Send 100 individual requests to measure response time
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(100):
                try:
                    request_start = time.time()
                    response = await client.get(endpoint)
                    request_duration = (time.time() - request_start) * 1000
                    
                    response_times.append(request_duration)
                    
                    if response.status_code == 200:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                        
                    # Small delay between requests
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    failed_requests += 1
                    print(f"   Request {i+1} failed: {e}")
        
        duration = time.time() - start_time
        monitor_task.cancel()
        
        return self._create_benchmark_result(
            "Response Time Test",
            duration,
            response_times,
            successful_requests,
            failed_requests
        )
    
    async def _benchmark_throughput(self) -> PerformanceBenchmark:
        """Benchmark maximum throughput."""
        
        endpoint = f"{self.base_url}/health"
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._monitor_system_metrics(60))
        
        start_time = time.time()
        
        # Send as many requests as possible in 60 seconds
        async with httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_connections=200)) as client:
            
            async def make_request():
                try:
                    request_start = time.time()
                    response = await client.get(endpoint)
                    request_duration = (time.time() - request_start) * 1000
                    return request_duration, response.status_code == 200
                except:
                    return 0, False
            
            # Create a pool of concurrent requests
            end_time = start_time + 60  # Run for 60 seconds
            tasks = []
            
            while time.time() < end_time:
                # Maintain 50 concurrent requests
                while len(tasks) < 50 and time.time() < end_time:
                    tasks.append(asyncio.create_task(make_request()))
                
                # Process completed tasks
                done_tasks = [task for task in tasks if task.done()]
                for task in done_tasks:
                    try:
                        duration, success = await task
                        response_times.append(duration)
                        if success:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                    except:
                        failed_requests += 1
                    tasks.remove(task)
                
                await asyncio.sleep(0.01)  # Small delay
            
            # Wait for remaining tasks
            if tasks:
                remaining_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in remaining_results:
                    if isinstance(result, tuple):
                        duration, success = result
                        response_times.append(duration)
                        if success:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                    else:
                        failed_requests += 1
        
        duration = time.time() - start_time
        monitor_task.cancel()
        
        return self._create_benchmark_result(
            "Throughput Test",
            duration,
            response_times,
            successful_requests,
            failed_requests
        )
    
    async def _benchmark_concurrency(self) -> PerformanceBenchmark:
        """Benchmark concurrent request handling."""
        
        endpoint = f"{self.base_url}/health"
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._monitor_system_metrics(30))
        
        start_time = time.time()
        
        async def concurrent_request_batch(batch_size: int):
            """Send a batch of concurrent requests."""
            batch_times = []
            batch_success = 0
            batch_failed = 0
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                
                async def single_request():
                    try:
                        request_start = time.time()
                        response = await client.get(endpoint)
                        duration = (time.time() - request_start) * 1000
                        return duration, response.status_code == 200
                    except:
                        return 0, False
                
                # Send concurrent requests
                tasks = [single_request() for _ in range(batch_size)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, tuple):
                        duration, success = result
                        batch_times.append(duration)
                        if success:
                            batch_success += 1
                        else:
                            batch_failed += 1
                    else:
                        batch_failed += 1
            
            return batch_times, batch_success, batch_failed
        
        # Test different concurrency levels
        for concurrency_level in [10, 25, 50, 100]:
            print(f"   Testing {concurrency_level} concurrent requests...")
            
            batch_times, batch_success, batch_failed = await concurrent_request_batch(concurrency_level)
            
            response_times.extend(batch_times)
            successful_requests += batch_success
            failed_requests += batch_failed
            
            # Small delay between batches
            await asyncio.sleep(2)
        
        duration = time.time() - start_time
        monitor_task.cancel()
        
        return self._create_benchmark_result(
            "Concurrency Test",
            duration,
            response_times,
            successful_requests,
            failed_requests
        )
    
    async def _benchmark_sustained_load(self) -> PerformanceBenchmark:
        """Benchmark sustained load handling."""
        
        endpoint = f"{self.base_url}/health"
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._monitor_system_metrics(180))  # 3 minutes
        
        start_time = time.time()
        test_duration = 180  # 3 minutes
        
        async with httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_connections=50)) as client:
            
            async def sustained_requester():
                """Continuously send requests for the test duration."""
                requests_made = 0
                requests_success = 0
                requests_failed = 0
                times = []
                
                end_time = start_time + test_duration
                
                while time.time() < end_time:
                    try:
                        request_start = time.time()
                        response = await client.get(endpoint)
                        duration = (time.time() - request_start) * 1000
                        
                        times.append(duration)
                        requests_made += 1
                        
                        if response.status_code == 200:
                            requests_success += 1
                        else:
                            requests_failed += 1
                            
                        # Maintain approximately 10 RPS per worker
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        requests_failed += 1
                        requests_made += 1
                        await asyncio.sleep(0.1)
                
                return times, requests_success, requests_failed
            
            # Run 10 concurrent sustained requesters
            tasks = [sustained_requester() for _ in range(10)]
            results = await asyncio.gather(*tasks)
            
            # Aggregate results
            for times, success, failed in results:
                response_times.extend(times)
                successful_requests += success
                failed_requests += failed
        
        duration = time.time() - start_time
        monitor_task.cancel()
        
        return self._create_benchmark_result(
            "Sustained Load Test",
            duration,
            response_times,
            successful_requests,
            failed_requests
        )
    
    async def _benchmark_stress_test(self) -> PerformanceBenchmark:
        """Benchmark stress test - push system to limits."""
        
        endpoint = f"{self.base_url}/health"
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._monitor_system_metrics(120))
        
        start_time = time.time()
        
        # Gradually increase load to stress test the system
        async with httpx.AsyncClient(timeout=5.0, limits=httpx.Limits(max_connections=500)) as client:
            
            for load_level in [50, 100, 200, 300]:
                print(f"   Stress testing with {load_level} concurrent requests...")
                
                async def stress_request():
                    try:
                        request_start = time.time()
                        response = await client.get(endpoint)
                        duration = (time.time() - request_start) * 1000
                        return duration, response.status_code == 200
                    except:
                        return 0, False
                
                # Send high concurrent load
                tasks = [stress_request() for _ in range(load_level)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, tuple):
                        duration, success = result
                        response_times.append(duration)
                        if success:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                    else:
                        failed_requests += 1
                
                # Cool down between stress levels
                await asyncio.sleep(5)
        
        duration = time.time() - start_time
        monitor_task.cancel()
        
        return self._create_benchmark_result(
            "Stress Test",
            duration,
            response_times,
            successful_requests,
            failed_requests
        )
    
    async def _benchmark_memory_usage(self) -> PerformanceBenchmark:
        """Benchmark memory usage under load."""
        
        endpoint = f"{self.base_url}/health"
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._monitor_system_metrics(60))
        
        start_time = time.time()
        
        # Send continuous requests for 60 seconds while monitoring memory
        async with httpx.AsyncClient(timeout=10.0) as client:
            
            end_time = start_time + 60
            
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    response = await client.get(endpoint)
                    duration = (time.time() - request_start) * 1000
                    
                    response_times.append(duration)
                    
                    if response.status_code == 200:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                        
                    await asyncio.sleep(0.05)  # 20 RPS
                    
                except:
                    failed_requests += 1
                    await asyncio.sleep(0.05)
        
        duration = time.time() - start_time
        monitor_task.cancel()
        
        return self._create_benchmark_result(
            "Memory Usage Test",
            duration,
            response_times,
            successful_requests,
            failed_requests
        )
    
    async def _monitor_system_metrics(self, duration: int):
        """Monitor system metrics during tests."""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_mb": memory.used / (1024 * 1024),
                    "memory_available_mb": memory.available / (1024 * 1024)
                }
                
                self.system_metrics.append(metrics)
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error monitoring system metrics: {e}")
    
    def _create_benchmark_result(
        self,
        test_name: str,
        duration: float,
        response_times: List[float],
        successful_requests: int,
        failed_requests: int
    ) -> PerformanceBenchmark:
        """Create benchmark result from raw data."""
        
        if not response_times:
            response_times = [0]
        
        total_requests = successful_requests + failed_requests
        
        # Calculate statistics
        avg_response_time = statistics.mean(response_times)
        p50_response_time = np.percentile(response_times, 50)
        p95_response_time = np.percentile(response_times, 95)
        p99_response_time = np.percentile(response_times, 99)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        requests_per_second = total_requests / duration if duration > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        # Calculate average system usage during test
        if self.system_metrics:
            avg_cpu = statistics.mean([m["cpu_percent"] for m in self.system_metrics])
            avg_memory = statistics.mean([m["memory_used_mb"] for m in self.system_metrics])
        else:
            avg_cpu = 0
            avg_memory = 0
        
        return PerformanceBenchmark(
            test_name=test_name,
            duration_seconds=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            max_response_time_ms=max_response_time,
            min_response_time_ms=min_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            throughput_mb_per_sec=0,  # Would need actual response size calculation
            cpu_usage_percent=avg_cpu,
            memory_usage_mb=avg_memory
        )
    
    def _generate_performance_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        # Calculate overall statistics
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.total_requests - r.successful_requests for r in self.results)
        
        overall_success_rate = (total_successful / total_requests) * 100 if total_requests > 0 else 0
        overall_error_rate = (total_failed / total_requests) if total_requests > 0 else 0
        
        # Find best and worst performing tests
        sla_compliant_tests = [r for r in self.results if r.meets_sla()]
        
        # Performance targets assessment
        performance_assessment = {
            "sub_200ms_response": any(r.p95_response_time_ms < 200 for r in self.results),
            "low_error_rate": overall_error_rate < 0.001,
            "high_throughput": any(r.requests_per_second > 100 for r in self.results),
            "stable_under_load": len(sla_compliant_tests) >= len(self.results) * 0.7,
            "sla_compliant_tests": len(sla_compliant_tests)
        }
        
        report = {
            "performance_summary": {
                "timestamp": datetime.utcnow().isoformat(),
                "total_execution_time_seconds": round(total_time, 2),
                "total_requests_sent": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_failed,
                "overall_success_rate_percent": round(overall_success_rate, 2),
                "overall_error_rate": round(overall_error_rate, 4),
                "tests_meeting_sla": len(sla_compliant_tests),
                "total_tests": len(self.results)
            },
            "performance_targets": {
                "response_time_sla": "< 200ms (95th percentile)",
                "error_rate_sla": "< 0.1%",
                "throughput_sla": "> 100 RPS",
                "targets_met": performance_assessment
            },
            "benchmark_results": [asdict(r) for r in self.results],
            "system_metrics_summary": self._analyze_system_metrics(),
            "recommendations": self._generate_recommendations()
        }
        
        self._print_performance_report(report)
        
        return report
    
    def _analyze_system_metrics(self) -> Dict[str, Any]:
        """Analyze collected system metrics."""
        if not self.system_metrics:
            return {"status": "no_metrics_collected"}
        
        cpu_values = [m["cpu_percent"] for m in self.system_metrics]
        memory_values = [m["memory_percent"] for m in self.system_metrics]
        
        return {
            "cpu_usage": {
                "average": round(statistics.mean(cpu_values), 2),
                "max": round(max(cpu_values), 2),
                "min": round(min(cpu_values), 2)
            },
            "memory_usage": {
                "average": round(statistics.mean(memory_values), 2),
                "max": round(max(memory_values), 2),
                "min": round(min(memory_values), 2)
            },
            "samples_collected": len(self.system_metrics)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze results and provide recommendations
        high_response_times = [r for r in self.results if r.p95_response_time_ms > 200]
        high_error_rates = [r for r in self.results if r.error_rate > 0.01]
        low_throughput = [r for r in self.results if r.requests_per_second < 100]
        
        if high_response_times:
            recommendations.append("Consider optimizing response times - some tests exceeded 200ms SLA")
        
        if high_error_rates:
            recommendations.append("Investigate high error rates in some test scenarios")
        
        if low_throughput:
            recommendations.append("Consider scaling improvements for higher throughput")
        
        if not recommendations:
            recommendations.append("Performance is meeting all SLA targets - excellent work!")
        
        return recommendations
    
    def _print_performance_report(self, report: Dict[str, Any]):
        """Print formatted performance report."""
        summary = report["performance_summary"]
        targets = report["performance_targets"]["targets_met"]
        
        print("\n" + "="*80)
        print("⚡ ENTERPRISE RAG PLATFORM PERFORMANCE REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   • Total Requests: {summary['total_requests_sent']:,}")
        print(f"   • Success Rate: {summary['overall_success_rate_percent']:.2f}%")
        print(f"   • Error Rate: {summary['overall_error_rate']:.4f}")
        print(f"   • SLA Compliance: {summary['tests_meeting_sla']}/{summary['total_tests']} tests")
        print(f"   • Execution Time: {summary['total_execution_time_seconds']}s")
        
        print(f"\n🎯 SLA TARGETS ASSESSMENT:")
        for target, met in targets.items():
            if isinstance(met, bool):
                status = "✅" if met else "❌"
                print(f"   • {target.replace('_', ' ').title()}: {status}")
            else:
                print(f"   • {target.replace('_', ' ').title()}: {met}")
        
        print(f"\n📈 BENCHMARK RESULTS:")
        for result in self.results:
            sla_status = "✅" if result.meets_sla() else "⚠️"
            print(f"   {sla_status} {result.test_name}:")
            print(f"      - P95 Response Time: {result.p95_response_time_ms:.1f}ms")
            print(f"      - Throughput: {result.requests_per_second:.1f} RPS")
            print(f"      - Error Rate: {result.error_rate:.4f}")
            print(f"      - Success Rate: {(result.successful_requests/result.total_requests*100):.1f}%")
        
        system_metrics = report.get("system_metrics_summary", {})
        if "cpu_usage" in system_metrics:
            print(f"\n🖥️ SYSTEM RESOURCE USAGE:")
            print(f"   • Average CPU: {system_metrics['cpu_usage']['average']:.1f}%")
            print(f"   • Peak CPU: {system_metrics['cpu_usage']['max']:.1f}%")
            print(f"   • Average Memory: {system_metrics['memory_usage']['average']:.1f}%")
            print(f"   • Peak Memory: {system_metrics['memory_usage']['max']:.1f}%")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   • {rec}")
        
        # Overall assessment
        sla_compliance_rate = (summary['tests_meeting_sla'] / summary['total_tests']) * 100
        if sla_compliance_rate >= 80:
            print(f"\n🎉 PERFORMANCE VALIDATION SUCCESSFUL!")
            print(f"   Platform meets enterprise performance standards.")
        else:
            print(f"\n⚠️  PERFORMANCE NEEDS OPTIMIZATION")
            print(f"   Some areas need improvement for full SLA compliance.")
        
        print("="*80)


# Main execution
async def main():
    """Run performance benchmarking suite."""
    
    print("🔥 Enterprise RAG Platform - Performance Benchmarking Suite")
    print("Testing all performance claims with comprehensive load testing...")
    
    tester = PerformanceTester()
    
    try:
        report = await tester.run_comprehensive_benchmarks()
        
        # Save detailed report
        with open("performance_benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: performance_benchmark_report.json")
        
        # Check if performance targets are met
        targets_met = report["performance_targets"]["targets_met"]
        success = all([
            targets_met["sub_200ms_response"],
            targets_met["low_error_rate"], 
            targets_met["high_throughput"],
            targets_met["stable_under_load"]
        ])
        
        return success
        
    except Exception as e:
        print(f"\n❌ Performance benchmarking failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n🏁 Performance benchmarking {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)