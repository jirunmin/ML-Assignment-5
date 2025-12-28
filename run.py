import importlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
from dotenv import load_dotenv
from jsonargparse import CLI

from llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester
from llm_single_needle_haystack_tester import LLMSingleNeedleHaystackTester
from test_case_loader import load_test_cases, get_needles
from evaluators.llm_evaluator import LLMEvaluator
from evaluators.string_match_evaluator import StringMatchEvaluator
from model import ModelProvider

load_dotenv()


@dataclass
class CommandArgs:
    """Command line arguments"""
    agent: str
    test_case_json: str

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    test_mode: Optional[str] = "multi"
    evaluator_type: Optional[str] = "llm"  # 'string' or 'llm'
    haystack_dir: Optional[str] = "PaulGrahamEssays"
    results_version: Optional[int] = 1

    num_tests: Optional[int] = 3
    num_test_cases: Optional[int] = 3  # Number of test cases to run
    context_lengths_min: Optional[int] = 1000
    context_lengths_max: Optional[int] = 100000
    context_lengths_num_intervals: Optional[int] = 10
    document_depth_percent_min: Optional[int] = 0
    document_depth_percent_max: Optional[int] = 100
    document_depth_percent_intervals: Optional[int] = 10

    save_results: Optional[bool] = True
    save_contexts: Optional[bool] = False
    print_ongoing_status: Optional[bool] = True


def parse_agent_spec(agent_spec: str) -> Tuple[str, str]:
    """
    Parse agent specification string.

    Args:
        agent_spec: Format "module.path:ClassName"

    Returns:
        Tuple of (module_name, class_name)
    """
    if ':' not in agent_spec:
        raise ValueError(
            f"Invalid agent specification: {agent_spec}. "
            f"Expected format: 'module.path:ClassName'"
        )

    module_name, class_name = agent_spec.split(':', 1)
    return module_name, class_name


def load_agent(agent_spec: str, api_key: str, base_url: str) -> ModelProvider:
    """
    Dynamically load agent implementation.

    Args:
        agent_spec: Agent specification "module.path:ClassName"
        api_key: API key
        base_url: API base URL

    Returns:
        Agent instance
    """
    module_name, class_name = parse_agent_spec(agent_spec)
    module = importlib.import_module(module_name)
    agent_class = getattr(module, class_name)
    return agent_class(api_key=api_key, base_url=base_url)


def run_single_test_case(test_case: dict, agent_spec: str, api_key: str,
                         base_url: str, args: CommandArgs) -> List[dict]:
    """
    Run a single test case.

    Args:
        test_case: Test case dictionary
        agent_spec: Agent specification
        api_key: API key
        base_url: API base URL
        args: Command arguments

    Returns:
        List of test results
    """
    needles = get_needles(test_case)

    # Load agent
    agent = load_agent(agent_spec, api_key, base_url)

    # Initialize evaluator based on type
    if args.evaluator_type == 'llm':
        evaluator = LLMEvaluator(
            api_key=api_key,
            base_url=base_url,
            ground_truth=test_case['ground_truth'],
            question=test_case['question']
        )
    elif args.evaluator_type == 'string':
        evaluator = StringMatchEvaluator(
            ground_truth=test_case['ground_truth'],
            case_sensitive=False,
            strip_whitespace=True
        )
    else:
        raise ValueError(f"Invalid evaluator_type: {args.evaluator_type}. Must be 'string' or 'llm'")

    # Create and run tester
    if args.test_mode == 'multi':
        tester = LLMMultiNeedleHaystackTester(
            model_to_test=agent,
            evaluator=evaluator,
            needles=needles,
            haystack_dir=args.haystack_dir,
            question=test_case['question'],
            results_version=args.results_version,
            num_tests=args.num_tests,
            save_results=args.save_results,
            save_contexts=args.save_contexts,
            print_ongoing_status=args.print_ongoing_status
        )
    else:
        if len(needles) != 1:
            raise ValueError(f"Single document mode requires exactly 1 needle, got {len(needles)}")

        tester = LLMSingleNeedleHaystackTester(
            model_to_test=agent,
            evaluator=evaluator,
            needle=needles[0],
            haystack_dir=args.haystack_dir,
            question=test_case['question'],
            results_version=args.results_version,
            context_lengths_min=args.context_lengths_min,
            context_lengths_max=args.context_lengths_max,
            context_lengths_num_intervals=args.context_lengths_num_intervals,
            document_depth_percent_min=args.document_depth_percent_min,
            document_depth_percent_max=args.document_depth_percent_max,
            document_depth_percent_intervals=args.document_depth_percent_intervals,
            save_results=args.save_results,
            save_contexts=args.save_contexts,
            print_ongoing_status=args.print_ongoing_status
        )

    tester.start_test()
    return tester.get_results()


def main():
    """Main function"""
    args = CLI(CommandArgs, as_positional=False)

    api_key = args.api_key or os.getenv('API_KEY')
    base_url = args.base_url or os.getenv('BASE_URL')

    if not api_key or not base_url:
        raise ValueError("API_KEY and BASE_URL must be provided via arguments or environment variables")

    if args.test_mode not in ['single', 'multi']:
        raise ValueError(f"test_mode must be 'single' or 'multi', got: {args.test_mode}")

    if args.evaluator_type not in ['string', 'llm']:
        raise ValueError(f"evaluator_type must be 'string' or 'llm', got: {args.evaluator_type}")

    # Load all test cases
    test_cases = load_test_cases(args.test_case_json)

    # Only run the first N test cases specified by num_test_cases argument
    num_test_cases_to_run = args.num_test_cases
    test_cases = test_cases[:num_test_cases_to_run]

    print("\n" + "=" * 80)
    print(f"Loaded {len(test_cases)} test case(s) from {args.test_case_json}")
    print(f"Only running first {num_test_cases_to_run} test cases")
    print(f"Evaluator Type: {args.evaluator_type}")
    print("=" * 80)

    # Run all test cases
    all_results = []
    test_case_summaries = []

    for idx, test_case in enumerate(test_cases, 1):
        test_id = test_case.get('id', idx)

        print("\n" + "=" * 80)
        print(f"Running Test Case {test_id} ({idx}/{len(test_cases)})")
        print("=" * 80)
        print(f"Question: {test_case['question']}")
        print(f"Ground Truth: {test_case['ground_truth']}")

        needles = get_needles(test_case)
        print(f"Needles: {len(needles)}")
        for i, needle in enumerate(needles, 1):
            print(f"  {i}. {needle[:100]}..." if len(needle) > 100 else f"  {i}. {needle}")
        print("=" * 80)

        try:
            results = run_single_test_case(
                test_case=test_case,
                agent_spec=args.agent,
                api_key=api_key,
                base_url=base_url,
                args=args
            )

            # Add test case ID to each result
            for result in results:
                result['test_case_id'] = test_id

            all_results.extend(results)

            # Calculate summary for this test case
            scores = [r['score'] for r in results]
            summary = {
                'test_case_id': test_id,
                'question': test_case['question'][:100],
                'num_runs': len(results),
                'avg_score': sum(scores) / len(scores) if scores else 0,
                'best_score': max(scores) if scores else 0,
                'worst_score': min(scores) if scores else 0,
                'perfect_count': sum(1 for s in scores if s == 10)
            }
            test_case_summaries.append(summary)

            print("\n" + "-" * 80)
            print(f"Test Case {test_id} Summary:")
            print(f"  Runs: {summary['num_runs']}")
            print(f"  Average Score: {summary['avg_score']:.2f}/10")
            print(f"  Best Score: {summary['best_score']}/10")
            print(f"  Worst Score: {summary['worst_score']}/10")
            print(f"  Perfect Scores: {summary['perfect_count']}")
            print("-" * 80)

        except Exception as e:
            print(f"\n❌ Error running test case {test_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL TEST SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Total test runs: {len(all_results)}")

    if all_results:
        all_scores = [r['score'] for r in all_results]
        print(f"\nOverall Statistics:")
        print(f"  Average score: {sum(all_scores) / len(all_scores):.2f}/10")
        print(f"  Best score: {max(all_scores)}/10")
        print(f"  Worst score: {min(all_scores)}/10")
        print(f"  Perfect scores: {sum(1 for s in all_scores if s == 10)}")

        print(f"\nPer Test Case Breakdown:")
        for summary in test_case_summaries:
            print(f"\n  Test Case {summary['test_case_id']}:")
            print(f"    Question: {summary['question']}")
            print(
                f"    Avg: {summary['avg_score']:.2f}/10 | Best: {summary['best_score']}/10 | Perfect: {summary['perfect_count']}/{summary['num_runs']}")

    print("=" * 80)


if __name__ == "__main__":
    main()