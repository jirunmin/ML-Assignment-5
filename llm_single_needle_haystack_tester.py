import asyncio
import glob
import json
import os
import time

import numpy as np

from evaluators.evaluator import Evaluator
from model import ModelProvider

from asyncio import Semaphore
from datetime import datetime, timezone


class LLMSingleNeedleHaystackTester:
    """
    Test framework for evaluating LLM's ability to retrieve specific information
    (needle) from a large context (haystack).
    """

    def __init__(self,
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None,
                 needle=None,
                 haystack_dir="PaulGrahamEssays",
                 question=None,
                 results_version=1,
                 context_lengths_min=1000,
                 context_lengths_max=100000,
                 context_lengths_num_intervals=10,
                 context_lengths=None,
                 document_depth_percent_min=0,
                 document_depth_percent_max=100,
                 document_depth_percent_intervals=10,
                 document_depth_percents=None,
                 document_depth_percent_interval_type="linear",
                 num_concurrent_requests=1,
                 save_results=True,
                 save_contexts=False,
                 final_context_length_buffer=200,
                 seconds_to_sleep_between_completions=None,
                 print_ongoing_status=True,
                 **kwargs):

        if not model_to_test:
            raise ValueError("A language model must be provided to test.")
        if not needle or not haystack_dir or not question:
            raise ValueError("Needle, haystack, and question must be provided.")

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.question = question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError(
                    "Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(
                    np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals,
                                endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'.")

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError(
                    "Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")

            if document_depth_percent_interval_type == 'linear':
                self.document_depth_percents = np.round(
                    np.linspace(document_depth_percent_min, document_depth_percent_max,
                                num=document_depth_percent_intervals, endpoint=True)).astype(int)
            elif document_depth_percent_interval_type == 'sigmoid':
                self.document_depth_percents = [self.logistic(x) for x in
                                                np.linspace(document_depth_percent_min, document_depth_percent_max,
                                                            document_depth_percent_intervals)]
            else:
                raise ValueError(
                    "document_depth_percent_interval_type must be either 'sigmoid' or 'linear' if document_depth_percents is None.")
        else:
            self.document_depth_percents = document_depth_percents

        self.model_to_test = model_to_test
        self.model_name = self.model_to_test.model_name
        self.evaluator = evaluator

    def logistic(self, x, L=100, x0=50, k=.1):
        """Apply logistic function for sigmoid distribution."""
        if x in [0, 100]:
            return x
        x = -k * (x - x0)
        return np.round(L * self.sigmoid(x), 3)

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    async def bound_evaluate_and_log(self, sem, *args):
        """Execute evaluate_and_log with semaphore control."""
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        """Run all test combinations of context lengths and depths."""
        sem = Semaphore(self.num_concurrent_requests)

        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        await asyncio.gather(*tasks)

    async def evaluate_and_log(self, context_length, depth_percent):
        """
        Evaluate model at specific context length and depth, then log results.
        """
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        context = await self.generate_context(context_length, depth_percent)

        prompt = self.model_to_test.generate_prompt(
            context=context,
            question=self.question
        )

        test_start_time = time.time()

        response = await self.model_to_test.evaluate_model(prompt)

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        score = self.evaluator.evaluate_response(response)

        results = {
            'model': self.model_name,
            'context_length': int(context_length),
            'depth_percent': float(depth_percent),
            'version': self.results_version,
            'needle': self.needle,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print(f"-- Test Summary -- ")
            print(f"Duration: {test_elapsed_time:.1f} seconds")
            print(f"Context: {context_length} tokens")
            print(f"Depth: {depth_percent}%")
            print(f"Score: {score}/1")  # Show raw score (0/1 or 1/1)
            print(f"Response: {response}\n")

        context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent * 100)}'

        if self.save_contexts:
            results['file_name'] = context_file_location

            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            with open(f'contexts/{context_file_location}_context.txt', 'w') as f:
                f.write(context)

        if self.save_results:
            if not os.path.exists('results'):
                os.makedirs('results')

            with open(f'results/{context_file_location}_results.json', 'w') as f:
                json.dump(results, f)

        if self.seconds_to_sleep_between_completions:
            await asyncio.sleep(self.seconds_to_sleep_between_completions)

    def result_exists(self, context_length, depth_percent):
        """Check if result already exists for given parameters."""
        results_dir = 'results/'
        if not os.path.exists(results_dir):
            return False

        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    async def generate_context(self, context_length, depth_percent):
        """Generate context with needle inserted at specified depth."""
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = self.insert_needle(context, depth_percent, context_length)
        return context

    def insert_needle(self, context, depth_percent, context_length):
        """Insert needle into context at specified depth percentage."""
        tokens_needle = self.model_to_test.encode_text_to_tokens(self.needle)
        tokens_context = self.model_to_test.encode_text_to_tokens(context)

        context_length -= self.final_context_length_buffer

        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            tokens_new_context = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_new_context = tokens_context[:insertion_point]
            period_tokens = self.model_to_test.encode_text_to_tokens('.')

            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        new_context = self.model_to_test.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        """Get token count of context."""
        return len(self.model_to_test.encode_text_to_tokens(context))

    def read_context_files(self):
        """Read and concatenate all haystack files."""
        context = ""
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def encode_and_trim(self, context, context_length):
        """Encode context to tokens and trim to specified length."""
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(tokens, context_length)
        return context

    def get_results(self):
        """Get all testing results."""
        return self.testing_results

    def print_start_test_summary(self):
        """Print summary of test configuration."""
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print(f"- Needle: {self.needle.strip()}")
        print("\n\n")

    def start_test(self):
        """Start the testing process."""
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())
