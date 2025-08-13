#!/usr/bin/env python3
"""
Test SandboxFusionTestCaseTool with real examples from training data
"""

import asyncio
import sys
import os
import json

# Add the correct paths to ensure imports work
sys.path.insert(0, '/ibex/project/c2328/verl_singularity/verl')
sys.path.insert(0, '/ibex/project/c2328/verl_singularity')

# Set working directory to the verl root
os.chdir('/ibex/project/c2328/verl_singularity/verl')

async def test_sandbox_tool():
    """Test SandboxFusionTestCaseTool with real examples"""
    
    # Import from the actual codebase 
    from recipe.code_agentic.code_agentic import SandboxFusionTestCaseTool
    
    # Test configuration
    config = {
        "sandbox_fusion_url": "http://10.68.171.9:8080/run_code",
        "default_timeout": 30,
        "default_language": "python", 
        "memory_limit_mb": 1024
    }
    
    # Create tool instance
    tool = SandboxFusionTestCaseTool(config, None)
    
    # Test with a simple math problem
    math_solution = '''
def solution(input_str):
    s, p = map(int, input_str.strip().split())
    discriminant = s * s - 4 * p
    
    if discriminant < 0:
        return "No"
    
    sqrt_discriminant = int(discriminant ** 0.5)
    if sqrt_discriminant * sqrt_discriminant != discriminant:
        return "No"
    
    n1 = (s + sqrt_discriminant) // 2
    n2 = (s - sqrt_discriminant) // 2
    
    if n1 > 0 and n2 > 0 and n1 + n2 == s and n1 * n2 == p:
        return "Yes"
    else:
        return "No"
'''

    math_ground_truth = {
        "inputs": ["3 2\n", "2 1\n"],
        "outputs": ["Yes\n", "Yes\n"]
    }

    print("=== Testing SandboxFusionTestCaseTool ===")
    print(f"Solution code:\n{math_solution}")
    print(f"Test cases: {len(math_ground_truth['inputs'])} inputs")
    
    try:
        # Create instance
        instance_id = await tool.create(ground_truth=math_ground_truth)
        print(f"Created instance: {instance_id}")
        
        # Execute
        parameters = {"code": math_solution}
        result_text, score, metadata = await tool.execute(instance_id, parameters)
        
        print(f"\n=== Tool Execution Result ===")
        print(f"Score: {score:.3f}")
        print(f"Pass Rate: {metadata.get('pass_rate', 0):.3f}")
        print(f"Passed Tests: {metadata.get('passed_tests', 0)}")
        print(f"Total Tests: {metadata.get('total_tests', 0)}")
        print(f"Execution Result: {metadata.get('execution_result', 'unknown')}")
        
        if metadata.get('failed_test_details'):
            print(f"Failed Test Details: {metadata['failed_test_details']}")
        
        print(f"\nResult text preview:")
        print(f"{result_text[:500]}...")
        
    except Exception as e:
        print(f"Error during tool execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing SandboxFusionTestCaseTool with real examples\n")
    
    # Run async test
    asyncio.run(test_sandbox_tool())
    
    print("\n=== Test completed ===")