#!/usr/bin/env python3
"""
Test script for code extraction utilities with new prompt format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_process_code import (
    extract_code_from_tool_call,
    extract_code_from_code_tags,
    extract_final_code_from_response,
    extract_debug_code_from_response
)


def test_extract_code_from_tool_call():
    """Test extraction from tool_call format"""
    print("Testing extract_code_from_tool_call...")
    
    # Test with quoted code
    text = '''Here is my debug code:
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "print('Hello World')"}}
</tool_call>'''
    
    result = extract_code_from_tool_call(text)
    expected = "print('Hello World')"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✓ extract_code_from_tool_call: quoted code")
    
    # Test with unquoted code
    text = '''Debug code:
<tool_call>
{"name": "code_interpreter", "arguments": {"code": def test(): return 42}}
</tool_call>'''
    
    result = extract_code_from_tool_call(text)
    expected = "def test(): return 42"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✓ extract_code_from_tool_call: unquoted code")


def test_extract_code_from_code_tags():
    """Test extraction from code tags"""
    print("Testing extract_code_from_code_tags...")
    
    text = '''Here is my final solution:
<code>
def solution(input_data):
    return input_data.upper()
</code>'''
    
    result = extract_code_from_code_tags(text)
    expected = "def solution(input_data):\n    return input_data.upper()"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✓ extract_code_from_code_tags: code extraction")


def test_extract_final_code_from_response():
    """Test final code extraction with priority"""
    print("Testing extract_final_code_from_response...")
    
    # Test with code tags (should be prioritized)
    text = '''Debug code:
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "print('debug')"}}
</tool_call>

Final solution:
<code>
def solution(x):
    return x * 2
</code>'''
    
    result = extract_final_code_from_response(text)
    expected = "def solution(x):\n    return x * 2"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✓ extract_final_code_from_response: code tags prioritized")
    
    # Test with only tool_call
    text = '''Only debug code:
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "def test(): return 42"}}
</tool_call>'''
    
    result = extract_final_code_from_response(text)
    expected = "def test(): return 42"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✓ extract_final_code_from_response: tool_call fallback")


def test_extract_debug_code_from_response():
    """Test debug code extraction"""
    print("Testing extract_debug_code_from_response...")
    
    text = '''Debug code:
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "print('debugging')"}}
</tool_call>

Final code:
<code>
def final():
    return "done"
</code>'''
    
    result = extract_debug_code_from_response(text)
    expected = "print('debugging')"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✓ extract_debug_code_from_response: debug code extraction")


def test_complex_response():
    """Test complex response with multiple formats"""
    print("Testing complex response...")
    
    text = '''Let me debug this step by step:

First, let me test the input:
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "print('Testing input')"}}
</tool_call>

Now let me try another approach:
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "x = 5; print(x)"}}
</tool_call>

Finally, here's my solution:
<code>
def solve_problem(data):
    result = data * 2
    return result
</code>'''
    
    # Test final code extraction
    final_result = extract_final_code_from_response(text)
    expected_final = "def solve_problem(data):\n    result = data * 2\n    return result"
    assert final_result == expected_final, f"Expected '{expected_final}', got '{final_result}'"
    
    # Test debug code extraction (should get the last tool_call)
    debug_result = extract_debug_code_from_response(text)
    expected_debug = "x = 5; print(x)"
    assert debug_result == expected_debug, f"Expected '{expected_debug}', got '{debug_result}'"
    
    print("✓ complex_response: multiple formats handled correctly")


def run_all_tests():
    """Run all tests"""
    print("Running code extraction utility tests with new prompt format...\n")
    
    try:
        test_extract_code_from_tool_call()
        test_extract_code_from_code_tags()
        test_extract_final_code_from_response()
        test_extract_debug_code_from_response()
        test_complex_response()
        
        print("\n🎉 All tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 