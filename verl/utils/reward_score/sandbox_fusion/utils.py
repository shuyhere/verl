# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import concurrent.futures  # <-- Import concurrent.futures
import json
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Any, Optional

import requests

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 5  # Increased from 3 to 5 for better resilience
INITIAL_RETRY_DELAY = 2  # Increased from 1 to 2 seconds
API_TIMEOUT = 60  # Increased from 10 to 60 seconds for better reliability

logger = logging.getLogger(__name__)

# Define supported languages list (optional, for documentation or validation)
SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]


def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
) -> tuple[Optional[dict[str, Any]], Optional[str]]:  # <-- Remove request_id parameter
    """
    Calls the remote sandbox API to execute code with retry logic for Gateway Timeout,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        code: The code string to execute.
        stdin: The standard input string.
        compile_timeout: Compile timeout in seconds.
        run_timeout: Run timeout in seconds.
        language: The programming language of the code (e.g., "python", "cpp", "java"). Defaults to "python".

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())  # <-- Generate request_id internally
    log_prefix = f"[Request ID: {request_id}] "  # <-- Create log prefix

    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,  # Use the passed language parameter
            "files": {},
            "fetch_files": [],
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # Calculate a reasonable request timeout based on compile/run timeouts plus a buffer
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT
    
    # Set both connect and read timeout for better reliability
    # connect_timeout: time to establish connection
    # read_timeout: time to read response
    timeout_config = (30, request_timeout)  # (connect_timeout, read_timeout)

    last_error = None  # Store the last error encountered

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {sandbox_fusion_url} with timeout {timeout_config}"
            )  # <-- Use internal log_prefix
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=timeout_config,  # Use tuple timeout for connect and read
            )

            # Check for Gateway Timeout (504) specifically for retrying
            if response.status_code == 504:
                last_error = (
                    f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )  # <-- Use internal log_prefix
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                    # Calculate increasing delay (e.g., 1s, 2s, 4s, ...) or (1s, 2s, 3s, ...)
                    # Simple linear increase: delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    # Exponential backoff: delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)  # Using linear increase for simplicity
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")  # <-- Use internal log_prefix
                    time.sleep(delay)
                continue  # Go to the next retry attempt

            # Check for other HTTP errors (e.g., 4xx, other 5xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(
                f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}"
            )  # <-- Use internal log_prefix
            return response.json(), None

        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on non-504 request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  # <-- Use internal log_prefix
    # Return the error message without the prefix, as the caller doesn't need the internal ID
    # Ensure API call failure returns error message, leading to -1 in check_correctness
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _process_single_case(
    case_index: int,
    stdin_data: Any,
    expected_output: Any,
    sandbox_fusion_url: str,
    generation: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    fn_name: Optional[str] = None,
) -> tuple[int, dict[str, Any]]:
    """Helper function to process a single test case."""
    api_response = None
    error_msg = None
    logger.info(f"Processing test case {case_index + 1}.")

    current_generation_code = generation
    
    # Auto-extract function name if not provided and language is Python
    if not fn_name and language == "python":
        import re
        
        # Find all function definitions with their indentation levels
        lines = generation.split('\n')
        top_level_functions = []
        
        for i, line in enumerate(lines):
            # Match function definitions and check indentation
            match = re.match(r'^(\s*)def\s+(\w+)\s*\(', line)
            if match:
                indent = len(match.group(1))
                func_name = match.group(2)
                
                # Consider functions with 0 indentation as top-level
                if indent == 0:
                    top_level_functions.append((func_name, i))
        
        if top_level_functions:
            # Prefer the first top-level function, or the last one if multiple
            # This handles cases where there might be helper functions after the main one
            fn_name = top_level_functions[0][0]  # Take first top-level function
            logger.info(f"Auto-extracted top-level function name: {fn_name}")
        else:
            # Fallback to the old method if no top-level functions found
            func_matches = re.findall(r'def\s+(\w+)\s*\(', generation)
            if func_matches:
                fn_name = func_matches[0]  # Take first function instead of last
                logger.info(f"Auto-extracted function name (fallback): {fn_name}")
    
    if fn_name and language == "python":
        # Wrapper assumes stdin_data is a JSON string for function arguments.
        wrapper_code = f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{generation}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    # --- Input Parsing ---
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip(): # If there's input
        # Parse input intelligently based on structure
        lines = _raw_input_str.strip().split('\\n')
        
        if len(lines) >= 2 and lines[0].strip().isdigit():
            # Looks like: n followed by n lines of data
            n = int(lines[0].strip())
            if len(lines) == n + 1:
                # This pattern: n followed by exactly n lines
                _args.append(n)
                picture_lines = [lines[i+1].strip() for i in range(n)]
                _args.append(picture_lines)
            else:
                # Fall back to parsing each line individually
                _args = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if line contains space-separated values
                    parts = line.split()
                    if len(parts) > 1:
                        # Multiple values in one line - parse each separately
                        for part in parts:
                            part = part.strip()
                            if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                                _args.append(int(part))
                            else:
                                try:
                                    if '.' in part:
                                        _args.append(float(part))
                                    else:
                                        _args.append(part)
                                except ValueError:
                                    _args.append(part)
                    else:
                        # Single value in the line
                        if line.isdigit() or (line.startswith('-') and line[1:].isdigit()):
                            _args.append(int(line))
                        else:
                            _args.append(line)
        else:
            # Fall back to parsing each line individually
            _args = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line contains space-separated values
                parts = line.split()
                if len(parts) > 1:
                    # Multiple values in one line - parse each separately
                    for part in parts:
                        part = part.strip()
                        if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                            _args.append(int(part))
                        else:
                            try:
                                if '.' in part:
                                    _args.append(float(part))
                                else:
                                    _args.append(part)
                            except ValueError:
                                _args.append(part)
                else:
                    # Single value in the line
                    if line.isdigit() or (line.startswith('-') and line[1:].isdigit()):
                        _args.append(int(line))
                    else:
                        _args.append(line)
        
        # If no arguments were parsed, fallback to the original string
        if not _args:
            _args = [_raw_input_str.strip()]

    # --- Function Location and Execution ---
    try:
        _target_callable = None
        # Try global scope first
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        # Else, if 'Solution' class exists, try to get its method
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            # Attempt to instantiate and get method.
            # Errors (e.g., Solution not a class, instantiation fails, method missing)
            # will be caught by the broad except block below.
            _solution_instance = _Solution_class()
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)

        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True # result, error_occurred

        _fn_result = _target_callable(*_args)
        return _fn_result, False # result, no_error
    except Exception: # Catches errors from Solution instantiation, getattr, or function call
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True # result, error_occurred

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred and _result is not None:
        # Serialize non-None result to stdout
        if isinstance(_result, (dict, list, tuple)) or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result))  # Ensure string conversion for print
        else:
            # For other types, default to string representation.
            print(str(_result))
    # If result is None, assume the user code already printed to stdout; do not print 'null'.
"""
        current_generation_code = wrapper_code
    else:
        # If no function name found or not Python, use original code
        current_generation_code = generation
    
    # stdin debug print removed
    stdin = None if stdin_data is None else str(stdin_data)
    try:
        if concurrent_semaphore:
            # logger.debug(f"Case {case_index + 1}: Attempting to acquire semaphore.")
            with concurrent_semaphore:
                # logger.debug(f"Case {case_index + 1}: Semaphore acquired. Calling API.")
                api_response, error_msg = call_sandbox_api(
                    sandbox_fusion_url=sandbox_fusion_url,
                    code=current_generation_code,
                    stdin=stdin,
                    compile_timeout=timeout,
                    run_timeout=timeout,
                    memory_limit_mb=memory_limit_mb,
                    language=language,
                )
            # logger.debug(f"Case {case_index + 1}: Semaphore released.")
        else:
            api_response, error_msg = call_sandbox_api(
                sandbox_fusion_url=sandbox_fusion_url,
                code=current_generation_code,
                stdin=stdin,
                compile_timeout=timeout,
                run_timeout=timeout,
                memory_limit_mb=memory_limit_mb,
                language=language,
            )
    except Exception as e:
        error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
        logger.error(f"Case {case_index + 1}: {error_msg}")
        traceback.print_exc()
        
    metadata = {
        "case_index": case_index,
        "input": stdin,
        "expected_output": str(expected_output) if expected_output else None,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
    }
    result_status = -1  # Default error: API request error or unknown sandbox error

    if error_msg:
        metadata["status"] = "api_error"
        result_status = -1  # API request itself failed (includes timeout after retries)
        logger.error(f"Case {case_index}: API error occurred: {error_msg}")
        # Log code and input only on error for brevity
        generation_to_log = generation[:200] + "..." if len(generation) > 200 else generation
        logger.error(f"Case {case_index}: code: {generation_to_log}")
        logger.error(f"Case {case_index}: input: {stdin}")
    elif api_response:
        # --- Add debug logging ---
        logger.debug(f"Case {case_index}: API Response: {api_response}")
        metadata["api_response"] = api_response
        metadata["api_status"] = api_response.get("status")
        compile_result = api_response.get("compile_result")
        run_result = api_response.get("run_result")

        # Extract compile information
        if compile_result:
            metadata["compile_status"] = compile_result.get("status")
            metadata["compile_duration"] = compile_result.get("execution_time")
            metadata["compile_stderr"] = compile_result.get("stderr")

        # Extract run information
        if run_result:
            metadata["run_status"] = run_result.get("status")
            metadata["stdout"] = run_result.get("stdout")
            metadata["stderr"] = run_result.get("stderr")  # stderr during runtime
            metadata["exit_code"] = run_result.get("return_code")
            metadata["duration"] = run_result.get("execution_time")

        # --- Determine status based on API response ---
        api_status = metadata["api_status"]

        if api_status == "SandboxError":
            metadata["status"] = "sandbox_error"
            result_status = -1  # Internal sandbox error
        elif api_status == "Failed":
            # --- Add debug logging ---
            logger.debug(f"API returned Failed status. Response: {api_response}")
            logger.debug(f"Compile Result: {compile_result}")
            logger.debug(f"Run Result: {run_result}")
            # --- Check the logic here ---
            # Compile failed or timed out
            is_compile_error = compile_result and (
                metadata["compile_status"] in ["Error", "TimeLimitExceeded"]
                or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0)
            )
            if is_compile_error:
                # Differentiate between compile_error and compile_timeout based on specific status
                if metadata["compile_status"] == "TimeLimitExceeded":
                    metadata["status"] = "compile_timeout"
                else:  # Includes Error and Finished but return_code != 0 cases
                    metadata["status"] = "compile_error"
                result_status = -4
            # Run failed or timed out
            elif run_result:
                # Modified condition: Check for TimeLimitExceeded OR (Finished with non-zero exit code) OR Error status
                is_runtime_error = (
                    metadata["run_status"] == "TimeLimitExceeded"
                    or metadata["run_status"] == "Error"
                    or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
                )
                if is_runtime_error:
                    if metadata["run_status"] == "TimeLimitExceeded":
                        metadata["status"] = "timeout"  # Runtime timeout
                        result_status = -3
                    else:  # Includes Error and Finished with non-zero return_code
                        metadata["status"] = "runtime_error"
                        result_status = -2
                else:
                    # Other Failed status with run_result, classify as unknown failure
                    logger.warning(f"Unknown run_status '{metadata['run_status']}' or state within Failed API status.")
                    metadata["status"] = "unknown_failure"
                    result_status = -1  # Default to -1
            else:
                # Status is Failed but neither a clear compile error nor run_result exists
                logger.warning("API status Failed but cannot determine specific error type (compile/run).")
                metadata["status"] = "unknown_failure_state"
                result_status = -1  # Default to -1
        elif api_status == "Success":
            # Run completed successfully, now check the answer
            if run_result and metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                # Note: Output might contain trailing newlines, need normalization
                if expected_output is None or str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n"):
                    result_status = True
                    metadata["status"] = "success"
                else:
                    result_status = False
                    metadata["status"] = "wrong_answer"
            else:
                # Status is Success but run_result status is not Finished, this is unexpected
                metadata["status"] = "unexpected_success_state"
                result_status = -1  # Classify as unknown error
        else:
            # API returned an unknown top-level status
            logger.warning(f"Unknown API status received: {api_status}")
            metadata["status"] = f"unknown_api_status_{api_status}"
            result_status = -1  # Default to -1
    else:  # api_response is None and no error_msg (Should not happen with current call_sandbox_api logic)
        metadata["status"] = "unknown_api_state"
        result_status = -1
        logger.error(f"Case {case_index}: Unknown API state (no response and no error message).")
    return result_status, metadata


def check_correctness(
    sandbox_fusion_url: str,
    in_outs: Optional[dict],
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = 1024,
    language: str = "python",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Checks the correctness of code generation using the remote sandbox API,
    processing test cases concurrently.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        in_outs: Dictionary containing "inputs" and "outputs" lists.
        generation: The generated code string.
        timeout: Timeout for each test case (compile and run share this timeout).
        language: The programming language of the code.

    Returns:
        A tuple (results, metadata_list).
        results: A list containing the test result for each input/output pair
                 (True/False/-1 api/sandbox err, -2 runtime err, -3 timeout, -4 compile err).
                 Results are ordered corresponding to the inputs.
        metadata_list: A list containing metadata dictionaries for each test case,
                       ordered corresponding to the inputs.
    """
    logger.info("Starting correctness check for generation.")
    # import ipdb; ipdb.set_trace()

    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        logger.warning("Invalid in_outs format provided.")
        return [-1], [{"error": "Invalid input/output data"}]
    
    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)
    
    
    assert_cases = in_outs.get("assert_case", [""] * num_cases)  # Default to empty strings if not provided
    results = [None] * num_cases  # Initialize with placeholders
    metadata_list = [None] * num_cases  # Initialize with placeholders

    if num_cases == 0:
        logger.warning("Empty inputs provided.")
        return [], []

    if len(inputs) != len(expected_outputs):
        logger.warning(f"Mismatch between number of inputs ({len(inputs)}) and outputs ({len(expected_outputs)}).")
        # Return error based on the number of inputs provided
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    # If assert_cases is provided, it overrides inputs and outputs
    if len(assert_cases) != num_cases:
        logger.warning(
            f"Mismatch between number of assert cases ({len(assert_cases)}) and inputs/outputs ({num_cases})."
        )
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    first_compile_error_index = -1

    # max_workers is limited by sandbox_fusion_max_concurrent from concurrent_semaphore
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 5)) as executor:
        # Submit all tasks, passing the concurrent_semaphore to _process_single_case
        future_to_index = {}
        for i, stdin_data in enumerate(inputs):
            future = executor.submit(
                _process_single_case,
                i,
                stdin_data,
                expected_outputs[i],
                sandbox_fusion_url,
                generation + "\n\n" + assert_cases[i],  # Append assert case to generation
                timeout,
                memory_limit_mb,
                language,
                concurrent_semaphore,
                fn_name,
            )
            future_to_index[future] = i

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata

                # Check for compile error (-4)
                if result_status == -4:
                    if first_compile_error_index == -1 or index < first_compile_error_index:
                        first_compile_error_index = index
                    # Optimization: could potentially cancel futures for index > first_compile_error_index
                    # However, cancellation is not guaranteed. Post-processing is safer.

            except Exception as exc:
                logger.error(f"Test case {index} generated an exception: {exc}")
                traceback.print_exc()
                results[index] = -1  # Mark as API/internal error
                metadata_list[index] = {
                    "case_index": index,
                    "input": str(inputs[index]),
                    "expected_output": str(expected_outputs[index]) if expected_outputs[index] else None,
                    "api_request_error": f"Internal execution error: {exc}",
                    "status": "internal_error",
                }

    # Post-processing for compile errors
    if first_compile_error_index != -1:
        logger.warning(
            f"Compile error detected in case {first_compile_error_index}. Marking subsequent cases as compile errors."
        )
        for i in range(first_compile_error_index + 1, num_cases):
            # Only update if not already processed (though it should be None or have a result)
            if results[i] != -4:  # Avoid overwriting if it somehow already got -4
                results[i] = -4
                # Update or create metadata for skipped cases due to compile error
                if metadata_list[i] is None:  # If future failed before returning metadata
                    metadata_list[i] = {
                        "case_index": i,
                        "input": str(inputs[i]),
                        "expected_output": str(expected_outputs[i]) if expected_outputs[i] else None,
                        "api_request_error": None,
                        "status": "compile_error_skipped",  # Indicate skipped due to prior compile error
                    }
                else:  # If future completed but result is overridden
                    metadata_list[i]["status"] = "compile_error_skipped"

    logger.info(f"Correctness check finished. Results: {results}")
    return results, metadata_list
