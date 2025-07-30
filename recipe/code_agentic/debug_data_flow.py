#!/usr/bin/env python3
"""
调试脚本：验证数据传递流程
"""

import asyncio
import json
import logging
from code_agentic import SandboxFusionTestCaseTool

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 模拟用户的数据格式
user_data = {
    'prompt': [{'role': 'user', 'content': 'Test prompt'}], 
    'ability': 'CODE_DEBUG', 
    'reward_model': [{'input_data': '6\nabc\nacb\nbac\nbca\ncab\ncba\n', 'expected_output': 'YES\nYES\nYES\nNO\nNO\nYES\n', 'test_id': 'test_0'}], 
    'agent_name': 'tool_agent', 
    'test_cases': [{'input_data': '6\nabc\nacb\nbac\nbca\ncab\ncba\n', 'expected_output': 'YES\nYES\nYES\nNO\nNO\nYES\n', 'test_id': 'test_0'}], 
    'question': 'Test question', 
    'input_output': {'inputs': ['6\nabc\nacb\nbac\nbca\ncab\ncba\n'], 'outputs': ['YES\nYES\nYES\nNO\nNO\nYES\n'], 'remote': False}, 
    'data_source': 'codeagentic_training'
}

async def debug_data_flow():
    """调试数据传递流程"""
    
    print("=== 调试数据传递流程 ===")
    
    # 工具配置
    config = {
        "sandbox_fusion_url": "http://10.68.171.9:8080/run_code",
        "num_workers": 1,
        "enable_global_rate_limit": False,
        "rate_limit": 1,
        "default_timeout": 30,
        "default_language": "python",
        "memory_limit_mb": 1024,
        "type": "native"
    }
    
    # 工具模式定义
    tool_schema = {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "Execute Python code and validate against test cases",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }
    
    # 创建工具实例
    tool = SandboxFusionTestCaseTool(config, tool_schema)
    
    print(f"用户数据中的测试用例数量: {len(user_data['test_cases'])}")
    print(f"用户数据中的测试用例: {user_data['test_cases']}")
    
    # 方法1：通过 create 方法传递测试用例
    print("\n--- 方法1：通过 create 方法传递测试用例 ---")
    instance_id1 = await tool.create("test_instance_1", test_cases=user_data['test_cases'])
    print(f"创建实例: {instance_id1}")
    
    # 检查测试用例是否正确存储
    if instance_id1 in tool.instance_test_cases:
        stored_test_cases = tool.instance_test_cases[instance_id1]
        print(f"存储的测试用例数量: {len(stored_test_cases)}")
        print(f"存储的测试用例: {stored_test_cases}")
    else:
        print("错误：测试用例未正确存储")
    
    # 方法2：设置全局测试用例
    print("\n--- 方法2：设置全局测试用例 ---")
    tool.set_global_test_cases(user_data['test_cases'])
    
    # 方法3：使用 add_test_cases 方法
    print("\n--- 方法3：使用 add_test_cases 方法 ---")
    tool.add_test_cases(user_data['test_cases'])
    
    # 测试获取测试用例的方法
    print("\n--- 测试获取测试用例的方法 ---")
    test_cases1 = tool.get_test_cases_for_instance(instance_id1)
    print(f"实例 {instance_id1} 的测试用例数量: {len(test_cases1)}")
    
    # 测试执行
    print("\n--- 测试执行 ---")
    test_code = """
t = int(input())
for _ in range(t):
    s = input().strip()
    if s == "abc":
        print("YES")
    elif s in ["acb", "bac", "cba"]:
        print("YES")
    else:
        print("NO")
"""
    
    try:
        result, score, metadata = await tool.execute_with_testing(
            instance_id1, 
            {"code": test_code}
        )
        
        print(f"执行结果: {result}")
        print(f"分数: {score}")
        print(f"元数据: {json.dumps(metadata, indent=2)}")
        
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    await tool.release(instance_id1)
    print(f"\n释放实例: {instance_id1}")

if __name__ == "__main__":
    asyncio.run(debug_data_flow()) 