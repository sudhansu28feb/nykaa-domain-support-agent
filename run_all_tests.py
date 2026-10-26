"""
Comprehensive test runner for Nykaa Support Agent environment verification and mock tests.
"""

import os
import subprocess
import sys
import time

TEST_FILES = [
    "tests/test_milestone1_dataset.py",
    "tests/test_milestone2_task3_chunking.py",
    "tests/test_milestone2_rag.py",
    "tests/test_milestone3_task6_lookup.py",
    "tests/test_milestone3_task7_crew.py",
    "tests/test_milestone3_task8_memory.py",
    "tests/test_milestone3_task9_schema.py",
    "tests/test_milestone3_task10_guardrails.py",
    "tests/test_milestone4_task14_autogen.py",
    "tests/test_milestone4_task15_governance.py",
    "tests/test_milestone5_task11_12_api.py",
    "tests/test_milestone5_task13_evaluation.py",
    "tests/test_standalone_grounded.py",
    "tests/test_crewai_mock.py",
    "tests/test_judge_mock.py",
    "tests/test_autogen_mock.py",
    "tests/test_offline_verification.py",
]

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    python_executable = sys.executable
    print("=" * 70, flush=True)
    print("NYKAA SUPPORT AGENT - TEST SUITE RUNNER", flush=True)
    print(f"Python: {python_executable}", flush=True)
    print(f"Version: {sys.version}", flush=True)
    print("=" * 70, flush=True)

    results = {}
    all_passed = True

    project_root = os.path.dirname(os.path.abspath(__file__))
    for test_file in TEST_FILES:
        print(f"\n>>> Running: {test_file} ...", flush=True)
        start_time = time.time()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        env["OTEL_SDK_DISABLED"] = "true"
        env["CREWAI_TESTING"] = "true"

        res = subprocess.run(
            [python_executable, test_file],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        duration = time.time() - start_time

        if res.returncode == 0:
            print(f"PASS ({duration:.2f}s)", flush=True)
            results[test_file] = ("PASS", duration, res.stdout)
        else:
            print(f"FAIL ({duration:.2f}s)", flush=True)
            print("STDERR:\n", res.stderr, flush=True)
            print("STDOUT:\n", res.stdout, flush=True)
            results[test_file] = ("FAIL", duration, res.stderr + "\n" + res.stdout)
            all_passed = False

    print("\n" + "=" * 70, flush=True)
    print("TEST SUITE EXECUTION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for test_file, (status, duration, _) in results.items():
        print(f"[{status}] {test_file:<45} ({duration:.2f}s)", flush=True)

    print("=" * 70, flush=True)
    if all_passed:
        print(f"ALL TESTS PASSED SUCCESSFULLY ({len(results)}/{len(TEST_FILES)})!", flush=True)
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
