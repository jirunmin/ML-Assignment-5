import subprocess
import sys
import re
import argparse


def build_run_command(args):
    """
    Build the run.py command with all arguments
    """
    cmd = [sys.executable, "run.py"]
    cmd.extend(["--agent", args.agent])
    cmd.extend(["--test_case_json", args.test_case_json])
    
    if args.api_key:
        cmd.extend(["--api_key", args.api_key])
    if args.base_url:
        cmd.extend(["--base_url", args.base_url])
    if args.test_mode:
        cmd.extend(["--test_mode", args.test_mode])
    if args.evaluator_type:
        cmd.extend(["--evaluator_type", args.evaluator_type])
    if args.haystack_dir:
        cmd.extend(["--haystack_dir", args.haystack_dir])
    if args.results_version:
        cmd.extend(["--results_version", str(args.results_version)])
    if args.num_tests:
        cmd.extend(["--num_tests", str(args.num_tests)])
    if args.max_test_cases:
        cmd.extend(["--max_test_cases", str(args.max_test_cases)])
    cmd.extend(["--save_results", str(args.save_results).lower()])
    cmd.extend(["--save_contexts", str(args.save_contexts).lower()])
    cmd.extend(["--print_ongoing_status", str(args.print_ongoing_status).lower()])
    
    return cmd


def filter_output(output_lines, show_questions=False, show_needles=False, show_test_runs=False):
    """
    Filter the run.py output to show only key information
    """
    filtered_lines = []
    
    # Track different states
    in_summary = False
    in_overall_summary = False
    in_test_case = False
    
    # Join all lines into a single string for easier multiline matching
    full_output = ''.join(output_lines)
    
    # Extract test case summaries
    test_summary_pattern = re.compile(r'(\-{70,}\nTest Case \d+ Summary:\n.*?\-{70,})', re.DOTALL)
    test_summaries = test_summary_pattern.findall(full_output)
    for summary in test_summaries:
        filtered_lines.append(summary)
    
    # Extract overall summary
    overall_summary_pattern = re.compile(r'(\={70,}\nOVERALL TEST SUMMARY\n.*?)(?=\n\={70,}\nFILTERED TEST SUMMARY|$)', re.DOTALL)
    overall_summaries = overall_summary_pattern.findall(full_output)
    for summary in overall_summaries:
        # Ensure we capture the closing delimiter
        if summary.strip():
            filtered_lines.append(summary.strip() + '\n' + '='*80)
    
    # Extract errors if any
    error_pattern = re.compile(r'(❌ Error.*?)(?=\-{70,}|\={70,}|$)', re.DOTALL)
    errors = error_pattern.findall(full_output)
    for error in errors:
        filtered_lines.append(error)
    
    return filtered_lines


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run filtered tests using run.py")
    
    # Required arguments
    parser.add_argument("--agent", required=True, help="Agent specification (module.path:ClassName)")
    parser.add_argument("--test_case_json", required=True, help="Path to test case JSON file")
    
    # Optional arguments (matching run.py)
    parser.add_argument("--api_key", help="API key")
    parser.add_argument("--base_url", help="API base URL")
    parser.add_argument("--test_mode", choices=["single", "multi"], default="multi", help="Test mode")
    parser.add_argument("--evaluator_type", choices=["string", "llm"], default="string", help="Evaluator type")
    parser.add_argument("--haystack_dir", default="PaulGrahamEssays", help="Haystack directory")
    parser.add_argument("--results_version", type=int, default=1, help="Results version")
    parser.add_argument("--num_tests", type=int, default=5, help="Number of tests per case")
    parser.add_argument("--max_test_cases", type=int, help="Maximum number of test cases to run")
    parser.add_argument("--save_results", action="store_true", default=True, help="Save results")
    parser.add_argument("--save_contexts", action="store_true", default=False, help="Save contexts")
    parser.add_argument("--print_ongoing_status", action="store_true", default=True, help="Print ongoing status")
    
    # Filter options
    parser.add_argument("--show_questions", action="store_true", help="Show questions in output")
    parser.add_argument("--show_needles", action="store_true", help="Show needles in output")
    parser.add_argument("--show_test_runs", action="store_true", help="Show individual test runs")
    
    args = parser.parse_args()
    
    # Build and execute the run.py command
    cmd = build_run_command(args)
    
    print(f"Executing: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Collect all output
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line)
            # Print unfiltered output in real-time if requested
            sys.stdout.write(line)
            sys.stdout.flush()
        
        process.wait()
        
        # Filter and print the summarized output
        print("\n" + "=" * 80)
        print("FILTERED TEST SUMMARY")
        print("=" * 80)
        
        filtered_lines = filter_output(
            output_lines,
            args.show_questions,
            args.show_needles,
            args.show_test_runs
        )
        
        for line in filtered_lines:
            print(line)
            
    except Exception as e:
        print(f"Error executing run.py: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()