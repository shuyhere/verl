#!/usr/bin/env python3
"""
Test script for code extraction utilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_singularity.verl.recipe.code_agentic.code_agentic_utils.extract_process_code import (
    preprocess_code,
    extract_code_from_markdown,
    extract_function_code,
    get_function_name,
    prepare_code_for_testing,
    extract_code_block_from_solution,
    wrap_code_for_execution
)


def test_preprocess_code():
    """Test code preprocessing functionality"""
    print("Testing preprocess_code...")
    
    # Test syntax fix
    code = "x = 2^3\ny = 5^2"
    result = preprocess_code(code)
    expected = "x = 2**3\ny = 5**2"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ preprocess_code: syntax fix works")
    
    # Test empty code
    result = preprocess_code("")
    assert result == "", f"Expected empty string, got {result}"
    print("✓ preprocess_code: empty code handled")


def test_extract_code_from_markdown():
    """Test markdown code extraction"""
    print("Testing extract_code_from_markdown...")
    
    # Test Python code block
    code = "```python\ndef test():\n    return 42\n```"
    result = extract_code_from_markdown(code)
    expected = "def test():\n    return 42"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ extract_code_from_markdown: Python code block")
    
    # Test generic code block
    code = "```\ndef test():\n    return 42\n```"
    result = extract_code_from_markdown(code)
    expected = "def test():\n    return 42"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ extract_code_from_markdown: generic code block")


def test_extract_function_code():
    """Test function code extraction"""
    print("Testing extract_function_code...")
    
    code = """
import sys

def solution(input_data):
    return input_data.upper()

print("test")
"""
    result = extract_function_code(code)
    expected = "def solution(input_data):\n    return input_data.upper()"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ extract_function_code: function extraction")


def test_get_function_name():
    """Test function name extraction"""
    print("Testing get_function_name...")
    
    code = "def my_function(x):\n    return x * 2"
    result = get_function_name(code)
    expected = "my_function"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ get_function_name: function name extraction")
    
    # Test default case
    code = "x = 1\ny = 2"
    result = get_function_name(code)
    expected = "solution"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ get_function_name: default name")


def test_prepare_code_for_testing():
    """Test code preparation for testing"""
    print("Testing prepare_code_for_testing...")
    
    code = "```python\ndef solution(x):\n    return x * 2\n```"
    test_cases = []
    result = prepare_code_for_testing(code, test_cases)
    
    expected_import = "import sys"
    expected_function = "def solution(x):\n    return x * 2"
    expected_call = "result = solution(input_str)"
    
    assert expected_import in result, "Missing import sys"
    assert expected_function in result, "Missing function definition"
    assert expected_call in result, "Missing function call"
    print("✓ prepare_code_for_testing: code wrapping")


def test_extract_code_block_from_solution():
    """Test solution code block extraction"""
    print("Testing extract_code_block_from_solution...")
    
    solution = "Here is my solution:\n```python\ndef answer(x):\n    return x + 1\n```"
    result = extract_code_block_from_solution(solution)
    expected = "def answer(x):\n    return x + 1"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ extract_code_block_from_solution: markdown extraction")


def test_wrap_code_for_execution():
    """Test code wrapping for execution"""
    print("Testing wrap_code_for_execution...")
    
    code = "def test(x):\n    return x * 2"
    result = wrap_code_for_execution(code)
    
    expected_import = "import sys"
    expected_function = "def test(x):\n    return x * 2"
    expected_call = "result = test(input_str)"
    expected_markdown = "```python"
    
    assert expected_import in result, "Missing import sys"
    assert expected_function in result, "Missing function definition"
    assert expected_call in result, "Missing function call"
    assert expected_markdown in result, "Missing markdown formatting"
    print("✓ wrap_code_for_execution: code wrapping")


def run_all_tests():
    """Run all tests"""
    print("Running code extraction utility tests...\n")
    
    try:
        test_preprocess_code()
        test_extract_code_from_markdown()
        test_extract_function_code()
        test_get_function_name()
        test_prepare_code_for_testing()
        test_extract_code_block_from_solution()
        test_wrap_code_for_execution()
        
        print("\n🎉 All tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 