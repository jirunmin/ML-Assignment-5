import json
from typing import Dict, List, Union


def load_test_cases(json_path: str) -> List[Dict]:
    """
    Load test cases from JSON file.
    Supports both single test case and multiple test cases format.

    Args:
        json_path: Path to JSON file containing test case(s)

    Returns:
        List of test case dictionaries, each containing 'needle', 'question', and 'ground_truth'
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if it's a single test case or multiple test cases
    if isinstance(data, list):
        # Multiple test cases format
        test_cases = data
    elif isinstance(data, dict):
        # Check if it's a wrapper with 'test_cases' key
        if 'test_cases' in data:
            test_cases = data['test_cases']
        else:
            # Single test case format
            test_cases = [data]
    else:
        raise ValueError("Invalid JSON format: expected dict or list")

    # Validate all test cases
    required_fields = ['needle', 'question', 'ground_truth']
    for idx, test_case in enumerate(test_cases):
        for field in required_fields:
            if field not in test_case:
                raise ValueError(f"Test case {idx}: Missing required field '{field}'")

        # Add test case ID if not present
        if 'id' not in test_case:
            test_case['id'] = idx + 1

    return test_cases


def load_test_case(json_path: str) -> Dict:
    """
    Load single test case from JSON file (backward compatibility).

    Args:
        json_path: Path to JSON file containing test case

    Returns:
        Dict containing 'needle', 'question', and 'ground_truth'
    """
    test_cases = load_test_cases(json_path)
    if len(test_cases) != 1:
        raise ValueError(f"Expected single test case, but found {len(test_cases)}")
    return test_cases[0]


def is_multi_needle(test_case: Dict) -> bool:
    """
    Determine if test case is multi-needle based on needle field.

    Args:
        test_case: Test case dictionary

    Returns:
        True if multi-needle, False otherwise
    """
    needle = test_case['needle']
    return isinstance(needle, list) and len(needle) > 1


def get_needles(test_case: Dict) -> List[str]:
    """
    Extract needles from test case.

    Args:
        test_case: Test case dictionary

    Returns:
        List of needle strings
    """
    needle = test_case['needle']
    if isinstance(needle, list):
        return needle
    else:
        return [needle]