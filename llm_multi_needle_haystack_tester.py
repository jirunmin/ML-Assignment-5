import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import List, Dict

from evaluators.evaluator import Evaluator
from model import ModelProvider


class LLMMultiNeedleHaystackTester:
    """
    Multi-document Needle test framework.
    
    - Needles are randomly inserted into different text files at different depths
    - The agent needs to retrieve information from all relevant files
    """

    def __init__(self,
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None,
                 needles: List[str] = None,
                 haystack_dir: str = "PaulGrahamEssays",
                 question: str = None,
                 results_version: int = 1,
                 save_results: bool = True,
                 save_contexts: bool = False,
                 print_ongoing_status: bool = True,
                 num_tests: int = 1):
        """
        Initialize multi-document test framework.

        Args:
            model_to_test: The agent model to test
            evaluator: Evaluator for scoring responses
            needles: List of information to be inserted
            haystack_dir: Folder containing text files
            question: Question to ask about the needles
            results_version: Version number for results
            save_results: Whether to save test results
            save_contexts: Whether to save generated contexts
            print_ongoing_status: Whether to print progress
            num_tests: Number of test runs (each with random insertion positions)
        """
        if not model_to_test or not needles or not question:
            raise ValueError("model_to_test, needles, and question must be provided.")

        self.model_to_test = model_to_test
        self.evaluator = evaluator
        self.needles = needles
        self.haystack_dir = haystack_dir
        self.question = question
        self.results_version = results_version
        self.save_results = save_results
        self.save_contexts = save_contexts
        self.print_ongoing_status = print_ongoing_status
        self.num_tests = num_tests
        self.testing_results = []

        self.txt_files = self._load_all_txt_files()

        if len(self.txt_files) < len(needles):
            raise ValueError(f"Not enough text files ({len(self.txt_files)}) for needles ({len(needles)})")

    def _load_all_txt_files(self) -> List[Dict[str, str]]:
        """
        Load all text files from the haystack directory.

        Returns:
            List of dict: [{"path": "xxx.txt", "content": "...", "tokens": [...]}, ...]
        """
        base_dir = os.path.abspath(os.path.dirname(__file__))
        txt_dir = os.path.join(base_dir, self.haystack_dir)

        files = []
        for filename in os.listdir(txt_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(txt_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                tokens = self.model_to_test.encode_text_to_tokens(content)

                files.append({
                    'filename': filename,
                    'path': filepath,
                    'content': content,
                    'tokens': tokens,
                    'token_count': len(tokens)
                })

        if self.print_ongoing_status:
            print(f"Loaded {len(files)} text files from {self.haystack_dir}")
            for f in files:
                print(f"  - {f['filename']}: {f['token_count']} tokens")

        return files

    def _insert_needle_into_file(self, file_data: Dict, needle: str) -> Dict:
        """
        Insert needle randomly into a file at some depth.

        Args:
            file_data: File data dictionary
            needle: Needle to insert

        Returns:
            dict: {"filename": ..., "depth_percent": ..., "modified_content": ...}
        """
        tokens = file_data['tokens'].copy()
        needle_tokens = self.model_to_test.encode_text_to_tokens(needle)

        # Random depth percentage
        depth_percent = random.uniform(0, 100)
        insertion_point = int(len(tokens) * (depth_percent / 100))

        # Find nearest sentence boundary
        period_tokens = self.model_to_test.encode_text_to_tokens('.')
        while insertion_point > 0 and tokens[insertion_point - 1] not in period_tokens:
            insertion_point -= 1

        # Insert needle
        new_tokens = tokens[:insertion_point] + needle_tokens + tokens[insertion_point:]
        modified_content = self.model_to_test.decode_tokens(new_tokens)

        # Calculate actual depth
        actual_depth = (insertion_point / len(tokens)) * 100

        return {
            'filename': file_data['filename'],
            'depth_percent': actual_depth,
            'modified_content': modified_content,
            'original_tokens': len(tokens),
            'new_tokens': len(new_tokens)
        }

    def _generate_multi_doc_context(self) -> Dict:
        """
        Generate multi-document context with each needle inserted into a different text file.

        Returns:
            dict: {
                "files": [list of ALL files, with needles inserted where applicable],
                "needle_locations": [{"needle": ..., "filename": ..., "depth": ...}]
            }
        """
        # Randomly select files for each needle
        selected_files = random.sample(self.txt_files, len(self.needles))

        # Create a mapping of which files have needles
        needle_file_map = {}
        for needle, file_data in zip(self.needles, selected_files):
            needle_file_map[file_data['filename']] = needle

        all_files = []
        needle_locations = []

        # Process all files
        for file_data in self.txt_files:
            filename = file_data['filename']

            if filename in needle_file_map:
                # This file gets a needle inserted
                needle = needle_file_map[filename]
                modified = self._insert_needle_into_file(file_data, needle)
                all_files.append(modified)

                needle_locations.append({
                    'needle': needle.strip(),
                    'filename': modified['filename'],
                    'depth_percent': modified['depth_percent']
                })

                if self.print_ongoing_status:
                    print(
                        f"  Inserted '{needle.strip()[:50]}...' into {modified['filename']} at {modified['depth_percent']:.1f}%")
            else:
                # This file remains unchanged
                all_files.append({
                    'filename': file_data['filename'],
                    'depth_percent': 0,  # No needle inserted
                    'modified_content': file_data['content'],
                    'original_tokens': file_data['token_count'],
                    'new_tokens': file_data['token_count']
                })

        return {
            'files': all_files,  # Now includes ALL files
            'needle_locations': needle_locations
        }

    async def evaluate_and_log(self, test_number: int):
        """
        Execute one test.

        Args:
            test_number: Test number
        """
        if self.print_ongoing_status:
            print(f"\n{'=' * 60}")
            print(f"Test #{test_number}")
            print(f"{'=' * 60}")

        # Generate context with needles
        context_data = self._generate_multi_doc_context()

        # Generate prompt
        prompt = self.model_to_test.generate_prompt(
            context_data=context_data,
            question=self.question
        )

        # Run model
        test_start_time = time.time()
        response = await self.model_to_test.evaluate_model(prompt)
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Evaluate response
        score = self.evaluator.evaluate_response(response)

        # Collect results
        results = {
            'test_number': test_number,
            'model': self.model_to_test.model_name,
            'version': self.results_version,
            'needles': self.needles,
            'needle_locations': context_data['needle_locations'],
            'total_files': len(self.txt_files),
            'files_with_needles': len(context_data['files']),
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print(f"\n-- Test Summary --")
            print(f"Duration: {test_elapsed_time:.1f} seconds")
            print(f"Score: {score}/1")  # Show raw score (0/1 or 1/1)
            print(f"Response: {response}\n")

        # Save results
        if self.save_results:
            if not os.path.exists('results'):
                os.makedirs('results')

            result_file = f'results/{self.model_to_test.model_name}_test_{test_number}_results.json'
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)

        # Save contexts
        if self.save_contexts:
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            for modified_file in context_data['files']:
                context_file = f"contexts/test_{test_number}_{modified_file['filename']}"
                with open(context_file, 'w') as f:
                    f.write(modified_file['modified_content'])

    async def run_test(self):
        """Run all tests."""
        for i in range(1, self.num_tests + 1):
            await self.evaluate_and_log(i)

    def print_start_test_summary(self):
        """Print test configuration summary."""
        print("\n" + "=" * 60)
        print("Starting Multi-Document Needle Retrieval Testing")
        print("=" * 60)
        print(f"Model: {self.model_to_test.model_name}")
        print(f"Total text files: {len(self.txt_files)}")
        print(f"Needles to insert: {len(self.needles)}")
        print(f"Number of tests: {self.num_tests}")
        print(f"\nNeedles:")
        for i, needle in enumerate(self.needles, 1):
            print(f"  {i}. {needle.strip()}")
        print("=" * 60 + "\n")

    def start_test(self):
        """Start testing."""
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())

    def get_results(self):
        """Get all test results."""
        return self.testing_results
