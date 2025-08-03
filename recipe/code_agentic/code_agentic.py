# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

import datasets
from omegaconf import OmegaConf

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.reward_score import sandbox_fusion
from recipe.code_agentic.code_agentic_utils import *

logger = logging.getLogger(__name__)


class TestResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TestCase:
    """Represents a test case with input and expected output"""
    input_data: str
    expected_output: str
    test_id: str = ""


@dataclass
class CodeExecutionResult:
    """Result of code execution with test case validation"""
    success: bool
    output: str
    test_result: TestResult
    error_message: Optional[str] = None
    execution_time: float = 0.0
    debug_info: Dict[str, Any] = None


class SandboxFusionTestCaseTool(SandboxFusionTool):
    """Enhanced code execution tool with input/output test case validation"""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = extract_code_pattern()
        self.max_debug_attempts = 5
        self.debug_history: List[Dict[str, Any]] = []
        self.instance_test_cases: Dict[str, List[dict]] = {}
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code to fix common syntax errors"""
        return preprocess_code(code)
    
    def _prepare_code_for_testing(self, code: str, test_cases: List[dict]) -> str:
        return prepare_code_for_testing(code, test_cases)
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract code from model response using the new prompt format
        
        Args:
            response: Model response text
            
        Returns:
            Extracted code
        """
        return extract_final_code_from_response(response)
    
    
    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[dict] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
            
        if ground_truth is None and "ground_truth" in kwargs:
            ground_truth = kwargs["ground_truth"]
            
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        if ground_truth is not None:
            self.instance_test_cases[instance_id] = ground_truth
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Stored ground_truth for instance_id {instance_id}: {len(ground_truth)} cases")
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"No ground_truth provided for instance_id {instance_id}")
                
        return instance_id


    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        """Execute code and return execution result with test case validation"""
        

        code = parameters.get("code", "")
        if not code:
            return "No code provided", 0.0, {"error": "no_code"}
        
        # Extract code from response using new prompt format
        extracted_code = self._extract_code_from_response(code)
        if extracted_code:
            code = extracted_code
        else:
            # Fallback to traditional markdown extraction
            matches = self.code_pattern.findall(code)
            if matches:
                code = matches[0].strip()
        
        code = self._preprocess_code(code)
        import ipdb; ipdb.set_trace()
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        
        run_result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        
        test_cases = self.instance_test_cases.get(instance_id, [])
        #test case format: {'inputs': ['3 4\n1 2 3\n4 5 6\n1 1 1\n0 1 2 3', '5 5\n3 4 6 8 4\n8 3 4 9 3\n10 20 30 40 50\n5 55 13 1000 113', '1 1\n3\n4\n5\n0'], 'outputs': ['2 3 3 3', '2 7 3 7 7', '7']}
        
        if not test_cases:
            metadata = {
                "output": run_result,
                "score": 0.0,
                "pass_rate": 0.0,
                "passed_tests": 0,
                "total_tests": 0,
                "failed_tests": [],
                "execution_result": "computed_without_test_cases"
            }
            return run_result, 0.0, metadata
        
        try:
            processed_code = self._prepare_code_for_testing(code, test_cases)
            formatted_code = processed_code
            
            if not ("```python" in processed_code or "```" in processed_code):
                formatted_code = f"```python\n{processed_code}\n```"
            
            import ipdb; ipdb.set_trace()
            _, metadata_list = sandbox_fusion.compute_score(
                sandbox_fusion_url=self.sandbox_fusion_url,
                concurrent_semaphore=None,
                memory_limit_mb=self.memory_limit_mb,
                completion=formatted_code,
                test_cases=test_cases,
                continuous=True
            )
            
            total_samples = 0
            for input_str in test_cases.get("inputs", []):
                lines = input_str.strip().split('\n')
                if lines and lines[0].isdigit():
                    total_samples += int(lines[0])
            
            passed_samples = 0
            failed_samples = 0
            passed_tests = []
            failed_tests = []
            failed_samples_details = [] 
            import ipdb; ipdb.set_trace()
            for i, meta in enumerate(metadata_list):
                if meta.get("status") == "success":
                    input_str = test_cases.get("inputs", [])[meta.get("case_index", 0)]
                    lines = input_str.strip().split('\n')
                    if lines and lines[0].isdigit():
                        test_cases_in_this_input = int(lines[0])
                        passed_samples += test_cases_in_this_input
                    
                    passed_tests.append({
                        "stdout": meta.get("stdout", ""),
                        "stderr": meta.get("stderr", ""),
                        "execution_time": meta.get("duration", 0.0)
                    })
                else:
                    input_str = test_cases.get("inputs", [])[meta.get("case_index", 0)]
                    lines = input_str.strip().split('\n')
                    if lines and lines[0].isdigit():
                        test_cases_in_this_input = int(lines[0])
                        failed_samples += test_cases_in_this_input
                        
                        if len(failed_samples_details) < 5:
                            failed_samples_details.append({
                                "case_index": meta.get("case_index", 0),
                                "input": input_str,
                                "expected_output": test_cases.get("outputs", [])[meta.get("case_index", 0)],
                                "actual_output": meta.get("stdout", ""),
                                "stderr": meta.get("stderr", ""),
                                "execution_time": meta.get("duration", 0.0),
                                "status": meta.get("status", "unknown"),
                                "error": meta.get("api_request_error", "Unknown error")
                            })
                    
                    failed_test_info = {
                        "status": meta.get("status", "unknown"),
                        "stdout": meta.get("stdout", ""),
                        "stderr": meta.get("stderr", ""),
                        "execution_time": meta.get("duration", 0.0),
                        "error": meta.get("api_request_error", "Unknown error"),
                        "input": input_str,
                        "expected_output": test_cases.get("outputs", [])[meta.get("case_index", 0)],
                        "actual_output": meta.get("stdout", "")
                    }
                    failed_tests.append(failed_test_info)
            
            if total_samples > 0:
                score = passed_samples / total_samples
            else:
                score = 0.0
            
            metadata = {
                "score": score,
                "raw_execution_output": run_result,
                "execution_result": "computed_with_sandbox_fusion",
                "passed_tests": len(passed_tests),
                "passed_samples": passed_samples,
                "failed_tests": len(failed_tests),
                "failed_samples": failed_samples,
                "failed_samples_details": failed_samples_details, 
            }

            output_lines = []
            output_lines.append(f"Output:")
            output_lines.append(run_result)
            output_lines.append(f"\nTest Case Evaluation:")
            output_lines.append(f"Score: {score:.3f}")
            output_lines.append(f"Tests Passed: {passed_samples}/{total_samples}")
            output_lines.append(f"Failed Tests: {failed_samples}")
            output_lines.append(f"Failed Samples: {failed_samples_details}")
            
            return "\n".join(output_lines), score, metadata
            
        except Exception as e:
            logger.warning(f"Failed to evaluate test cases: {e}")
            metadata = {
                "output": run_result,
                "score": 0.0,
                "pass_rate": 0.0,
                "passed_tests": 0,
                "total_tests": len(test_cases),
                "failed_tests": [],
                "execution_result": "computed_with_error",
                "error": str(e)
            }
            return f"Code executed successfully:\n{run_result}\n\nTest case evaluation failed: {e}", 0.0, metadata

class CodeAgenticDataset(RLHFDataset):
    """Dataset class for enhanced code agentic training with input/output test cases"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _read_files_and_tokenize(self):
        """Load and process dataset files"""
        dataframes = []
        
        for data_file in self.data_files:
            dataframe = None
            
            try:
                dataset = datasets.load_dataset(data_file)
                if "train" in dataset:
                    dataframe = dataset["train"]
                else:
                    dataframe = list(dataset.values())[0]  # Take first split
            except Exception as e:
                try:
                    dataset = datasets.load_dataset("json", data_files=data_file)
                    if "train" in dataset:
                        dataframe = dataset["train"]
                    else:
                        dataframe = list(dataset.values())[0]  # Take first split
                except Exception as e2:
                    print(f"Failed to load {data_file} as parquet: {e}")
                    print(f"Failed to load {data_file} as JSONL: {e2}")
                    continue
            
            if dataframe is None:
                print(f"Could not load dataframe from {data_file}")
                continue
            
            print(f"Original dataset size: {len(dataframe)}")
            
            def filter_large_samples(example):
                """filter out large samples"""
                question = example.get("question", [""])[0] if isinstance(example.get("question"), list) else example.get("question", "")
                input_output = example.get("input_output", [""])[0] if isinstance(example.get("input_output"), list) else example.get("input_output", "")
                
                # check question size
                if len(str(question)) > 50000:
                    return False
                
                # check input output size
                if len(str(input_output)) > 100000:
                    return False
                
                # try to parse input_output and check test case size
                if isinstance(input_output, str):
                    try:
                        import json
                        parsed = json.loads(input_output)
                        inputs = parsed.get("inputs", [])
                        outputs = parsed.get("outputs", [])
                        
                        # check total input output size
                        total_input_size = sum(len(str(inp)) for inp in inputs)
                        total_output_size = sum(len(str(out)) for out in outputs)
                        
                        if total_input_size > 200000 or total_output_size > 200000:
                            return False
                            
                    except (json.JSONDecodeError, Exception):
                        # if parse failed, check original string size
                        if len(input_output) > 100000:
                            return False
                
                return True
            
            # apply filter
            dataframe = dataframe.filter(filter_large_samples)
            print(f"After filtering large samples: {len(dataframe)}")
            
            # NOTE: Limit to 1% of data for debugging
            # debug_size = max(256, int(len(dataframe) * 0.01))
            # dataframe = dataframe.select(range(min(debug_size, len(dataframe))))
            
            data_source = "/".join(data_file.split("/")[-1:])
            # NOTE: you can add more data source and map_fn
            if data_source in ["codeforces"]:
                dataframe = dataframe.map(self.map_fn, remove_columns=dataframe.column_names, fn_kwargs={"data_source": data_source})
            else:
                dataframe = dataframe.map(self.map_fn, num_proc=1, fn_kwargs={"data_source": data_source})
            
            dataframes.append(dataframe)
        
        if not dataframes:
            raise ValueError("No valid data files could be loaded")
        
        self.dataframe = datasets.concatenate_datasets(dataframes)
        print(f"Dataset loaded with {len(self.dataframe)} samples")
        print(self.dataframe[0])
        
    
    def map_fn(self, row: dict, data_source: str = "code_agentic"):
        """Map dataset row to training format with code agentic prompt"""
        
        import os
        worker_id = os.getpid()
        
        question = row.get("question", [""])[0] if isinstance(row.get("question"), list) else row.get("question", "")
        input_output = row.get("input_output", [""])[0] if isinstance(row.get("input_output"), list) else row.get("input_output", "")
        
        question_size = len(str(question))
        input_output_size = len(str(input_output))
        
        if question_size > 10000 or input_output_size > 10000:
            print(f"WARNING: Large data detected at worker {worker_id}")
            print(f"  question_size: {question_size}")
            print(f"  input_output_size: {input_output_size}")
            print(f"  question preview: {str(question)[:200]}...")
            print(f"  input_output preview: {str(input_output)[:200]}...")
        
        prompt_content = get_code_problem_prompt(question)
        answer_format = get_answer_format()
        if isinstance(input_output, str):
            try:
                import json
                input_output = json.loads(input_output)
            except:
                input_output = {}
        
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        
        total_input_size = sum(len(str(inp)) for inp in inputs)
        total_output_size = sum(len(str(out)) for out in outputs)
        
        if total_input_size > 50000 or total_output_size > 50000:
            print(f"WARNING: Very large test cases detected at worker {worker_id}")
            print(f"  total_input_size: {total_input_size}")
            print(f"  total_output_size: {total_output_size}")
            print(f"  num_inputs: {len(inputs)}")
            print(f"  num_outputs: {len(outputs)}")
        
        # Create test cases in the format expected by prime_code
        test_cases_dict = {
            "inputs": inputs,
            "outputs": outputs
        }
        
        # Also create the detailed format for our custom tool
        test_cases_detailed = []
        for i, (input_data, expected_output) in enumerate(zip(inputs, outputs)):
            test_case = {
                "input_data": input_data,
                "expected_output": expected_output,
                "test_id": f"test_{i}"
            }
            test_cases_detailed.append(test_case)

        result = {
            "prompt": [{"role": "user", "content": prompt_content+answer_format}],
            "ability": "CODE_DEBUG",
            "reward_model": {"ground_truth": test_cases_dict},  
            "agent_name": "tool_agent",
            "question": question,
            "input_output": input_output,
            "data_source": data_source,
            "extra_info": {
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "code_interpreter": {
                        "create_kwargs": {"ground_truth": test_cases_dict},
                    },
                },
            }
        }
        
        result_size = len(str(result))
        if result_size > 100000:
            print(f"WARNING: Very large result data at worker {worker_id}: {result_size} chars")
        
        return result
        


def compute_code_score(data_source: str, solution_str: str, ground_truth: dict, extra_info: dict) -> dict:

    score = 0.0
    passed_tests = []
    failed_tests = []
    pred = "0/0"
    
    try:
        # 这些函数已经通过 from recipe.code_agentic.code_agentic_utils import * 导入了
        
        code = extract_final_code_from_response(solution_str)
        wrapped_code = wrap_code_for_execution(code)

        _, metadata_list = sandbox_fusion.compute_score(
            sandbox_fusion_url="http://10.68.171.9:8080/run_code",
            concurrent_semaphore=None,
            memory_limit_mb=1024,
            completion=wrapped_code,
            test_cases=ground_truth,
            continuous=True
        )

        
        total_samples = 0
        for input_str in ground_truth.get("inputs", []):
            lines = input_str.strip().split('\n')
            if lines and lines[0].isdigit():
                total_samples += int(lines[0])
        
        passed_samples = 0
        failed_samples = 0
        
        for meta in metadata_list:
            if meta.get("status") == "success":
                input_str = ground_truth.get("inputs", [])[meta.get("case_index", 0)]
                lines = input_str.strip().split('\n')
                if lines and lines[0].isdigit():
                    test_cases_in_this_input = int(lines[0])
                    passed_samples += test_cases_in_this_input
                
                passed_tests.append({
                    "stdout": meta.get("stdout", ""),
                    "stderr": meta.get("stderr", ""),
                    "execution_time": meta.get("duration", 0.0)
                })
            else:
                input_str = ground_truth.get("inputs", [])[meta.get("case_index", 0)]
                lines = input_str.strip().split('\n')
                if lines and lines[0].isdigit():
                    test_cases_in_this_input = int(lines[0])
                    failed_samples += test_cases_in_this_input
                
                failed_tests.append({
                    "status": meta.get("status", "unknown"),
                    "stdout": meta.get("stdout", ""),
                    "stderr": meta.get("stderr", ""),
                    "execution_time": meta.get("duration", 0.0),
                    "error": meta.get("api_request_error", "Unknown error")
                })
        
        if total_samples > 0:
            score = passed_samples / total_samples
        else:
            score = 0.0
        
    except Exception as e:
        logger.warning(f"Failed to evaluate test cases: {e}")

    num_turns = extra_info.get("num_turns", 1)
    if score < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        score = min(0, score + tool_call_reward)

    result = {
        "score": float(score),
        "data_source": data_source,
        "num_turns": num_turns
    }
    
    return result