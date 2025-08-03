import re
from typing import List, Optional, Tuple


def preprocess_code(code: str) -> str:
    """
    Preprocess code to fix common syntax errors
    
    Args:
        code: Raw code string
        
    Returns:
        Preprocessed code with common syntax fixes
    """
    if not code:
        return code
    lines = code.split("\n")
    processed_lines = []
    
    for line in lines:
        # Fix common syntax errors
        line = re.sub(r'(\w+)\^(\w+)', r'\1**\2', line)  # Fix ^ to **
        processed_lines.append(line)
    
    return "\n".join(processed_lines)


def extract_code_from_markdown(code: str) -> str:
    """
    Extract code from markdown code blocks
    
    Args:
        code: Code string that may contain markdown formatting
        
    Returns:
        Extracted code without markdown formatting
    """
    if "```python" in code:
        code = code.split("```python")[-1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if "\n" in code:
                first_line, rest = code.split("\n", 1)
                if first_line.strip().isalpha():
                    code = rest
    return code.strip()


def extract_function_code(code: str) -> str:
    """
    Extract function code from the beginning of def to the last return statement
    
    Args:
        code: Code string that may contain function definitions
        
    Returns:
        Extracted function code
    """
    if "def " in code:
        def_match = re.search(r'def\s+\w+\s*\([^)]*\)[^:]*:', code, re.DOTALL)
        if def_match:
            def_start = def_match.start()
            return_matches = list(re.finditer(r'return\s+[^\\n]+', code, re.DOTALL))
            if return_matches:
                last_return = return_matches[-1]
                return_end = last_return.end()
                code = code[def_start:return_end].strip()
            else:
                code = code[def_start:].strip()
    
    return code


def get_function_name(code: str) -> str:
    """
    Extract function name from code
    
    Args:
        code: Code string containing function definition
        
    Returns:
        Function name or "solution" as default
    """
    if "def " in code:
        function_match = re.search(r'def\s+(\w+)\s*\(', code)
        if function_match:
            return function_match.group(1)
    return "solution"


def prepare_code_for_testing(code: str, test_cases: List[dict]) -> str:
    """
    Prepare code for testing by wrapping it with input/output handling
    
    Args:
        code: Raw code string
        test_cases: List of test cases
        
    Returns:
        Code wrapped with input/output handling for testing
    """
    # Extract code from markdown if present
    code = extract_code_from_markdown(code)
    
    # Extract function code if present
    code = extract_function_code(code)
    
    # Get function name
    function_name = get_function_name(code)
    
    # Wrap code with input/output handling
    complete_code = f"""import sys

{code}

input_str = sys.stdin.read().strip()
result = {function_name}(input_str)
if isinstance(result, list):
    print("\\n".join(result) + "\\n")
else:
    print(result)
"""
    return complete_code


def extract_code_block_from_solution(solution_str: str) -> str:
    """
    Extract code block from solution string with various formats
    
    Args:
        solution_str: Solution string that may contain code blocks
        
    Returns:
        Extracted code block
    """
    code = solution_str
    
    # Try to extract from markdown code blocks
    code_block = None
    match = re.search(r"```python(.*?)(```|$)", code, re.DOTALL | re.IGNORECASE)
    if match:
        code_block = match.group(1).strip()
    else:
        match = re.search(r"```(.*?)(```|$)", code, re.DOTALL)
        if match:
            code_block = match.group(1).strip()
    
    if code_block:
        code = code_block
    else:
        # Extract function code if no markdown blocks found
        def_match = re.search(r'def\s+\w+\s*\([^)]*\)[^:]*:', code, re.DOTALL)
        if def_match:
            # Find the position of def
            def_start = def_match.start()
            # Find the last return statement
            return_matches = list(re.finditer(r'return\s+[^\\n]+', code, re.DOTALL))
            if return_matches:
                # Take the position of the last return
                last_return = return_matches[-1]
                return_end = last_return.end()
                # Extract code from def start to last return end
                code = code[def_start:return_end].strip()
            else:
                # If no return found, from def start to code end
                code = code[def_start:].strip()
        else:
            code = code.strip()
    
    return code


def wrap_code_for_execution(code: str) -> str:
    """
    Wrap code with input/output handling for execution
    
    Args:
        code: Raw code string
        
    Returns:
        Wrapped code ready for execution
    """
    function_match = re.search(r'def\s+(\w+)\s*\(', code)
    if function_match:
        function_name = function_match.group(1)
        wrapped_code = f"""import sys

{code}

input_str = sys.stdin.read().strip()
result = {function_name}(input_str)
if isinstance(result, list):
    print("\\n".join(result) + "\\n")
else:
    print(result)
"""
    else:
        wrapped_code = code
    
    # Add markdown formatting if not present
    if not (wrapped_code.strip().startswith("```python") or wrapped_code.strip().startswith("```")):
        wrapped_code = f"```python\n{wrapped_code}\n```"
    
    return wrapped_code


def extract_code_pattern() -> re.Pattern:
    """
    Get compiled regex pattern for extracting code from markdown
    
    Returns:
        Compiled regex pattern
    """
    return re.compile(r"```python(.*?)```", re.DOTALL)


def extract_code_from_tool_call(text: str) -> str:
    """
    Extract code from tool_call format: {"name": "code_interpreter", "arguments": {"code": "..."}}
    
    Args:
        text: Text containing tool_call format
        
    Returns:
        Extracted code from tool_call
    """
    # Pattern to match tool_call format
    tool_call_pattern = r'<tool_call>\s*\{\s*"name"\s*:\s*"code_interpreter"\s*,\s*"arguments"\s*:\s*\{\s*"code"\s*:\s*"([^"]*)"\s*\}\s*\}\s*</tool_call>'
    match = re.search(tool_call_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Alternative pattern for JSON without quotes around code
    alt_pattern = r'<tool_call>\s*\{\s*"name"\s*:\s*"code_interpreter"\s*,\s*"arguments"\s*:\s*\{\s*"code"\s*:\s*([^}]+)\s*\}\s*\}\s*</tool_call>'
    match = re.search(alt_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    return ""


def extract_code_from_code_tags(text: str) -> str:
    """
    Extract code from <code>your FINAL code here</code> format
    
    Args:
        text: Text containing code tags
        
    Returns:
        Extracted code from code tags
    """
    # Pattern to match <code>...</code> format
    code_pattern = r'<code>(.*?)</code>'
    match = re.search(code_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    return ""


def extract_final_code_from_response(response: str) -> str:
    """
    Extract final code from model response, prioritizing code tags over tool_call
    
    Args:
        response: Model response text
        
    Returns:
        Extracted final code
    """
    # First try to extract from <code> tags (final code)
    final_code = extract_code_from_code_tags(response)
    if final_code:
        return final_code
    
    # If no code tags found, try tool_call format
    tool_call_code = extract_code_from_tool_call(response)
    if tool_call_code:
        return tool_call_code
    
    # Fallback to traditional markdown extraction
    return extract_code_from_markdown(response)


def extract_debug_code_from_response(response: str) -> str:
    """
    Extract debug code from tool_call format in model response
    
    Args:
        response: Model response text
        
    Returns:
        Extracted debug code from tool_call
    """
    return extract_code_from_tool_call(response)
