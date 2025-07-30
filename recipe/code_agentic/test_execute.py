#!/usr/bin/env python3
"""
测试脚本：测试 SandboxFusionTestCaseTool 的 execute 函数
"""

import asyncio
import logging
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from recipe.code_agentic.code_agentic import SandboxFusionTestCaseTool
    from verl.tools.base_tool import OpenAIFunctionToolSchema
    print("✅ 成功导入 SandboxFusionTestCaseTool")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    # 尝试直接导入
    sys.path.insert(0, os.path.dirname(__file__))
    from code_agentic import SandboxFusionTestCaseTool
    from verl.tools.base_tool import OpenAIFunctionToolSchema

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_execute():
    """测试 execute 函数"""
    
    # 创建工具实例
    config = {
        "sandbox_fusion_url": "http://10.68.171.9:8080/run_code",
        "memory_limit_mb": 1024,
        "default_timeout": 30,
        "default_language": "python"
    }
    
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "code_interpreter",
            "description": "Execute Python code with test case validation",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    )
    
    tool = SandboxFusionTestCaseTool(config, tool_schema)
    
    ground_truth = {
        "inputs": [
            "6\nabc\nacb\nbac\nbca\ncab\ncba\n",
            "1\nabc\n",
            "3\nabc\nabc\nabc\n",
            "5\ncab\nacb\ncba\nbac\nbca\n",
            "6\nabc\nabc\nabc\nabc\nabc\nabc\n"
        ],
        "outputs": [
            "YES\nYES\nYES\nNO\nNO\nYES\n",
            "YES\n",
            "YES\nYES\nYES\n",
            "NO\nYES\nYES\nYES\nNO\n",
            "YES\nYES\nYES\nYES\nYES\nYES\n"
        ]
    }
    
    print("=" * 80)
    print("测试 SandboxFusionTestCaseTool.execute 函数")
    print("=" * 80)
    
    # 测试用例1：正确的解决方案
    correct_solution = """```python
def solution(input_str):
    lines = input_str.strip().split('\\n')
    t = int(lines[0])
    results = []
    
    for i in range(1, t + 1):
        s = lines[i]
        if s == "abc":
            results.append("YES")
        elif s in ["acb", "bac", "cba"]:
            results.append("YES")
        else:
            results.append("NO")
    
    return results
```"""
    
    print("\\n" + "=" * 60)
    print("测试用例1：正确的解决方案")
    print("=" * 60)
    
    # 创建实例
    instance_id = await tool.create(ground_truth=ground_truth)
    print(f"创建的实例ID: {instance_id}")
    
    # 执行代码
    result, score, metadata = await tool.execute(instance_id, {"code": correct_solution})
    
    print(f"执行结果: {result}")
    print(f"分数: {score}")
    print(f"元数据: {metadata}")

    
    # 测试用例2：部分正确的解决方案
    partial_solution = """```python
def solution(input_str):
    lines = input_str.strip().split('\\n')
    t = int(lines[0])
    results = []
    
    for i in range(1, t + 1):
        s = lines[i]
        if s == "abc":
            results.append("YES")
        elif s in ["acb", "bac"]:  # 缺少 cba
            results.append("YES")
        else:
            results.append("NO")
    
    return results
```"""
    
    print("\\n" + "=" * 60)
    print("测试用例2：部分正确的解决方案")
    print("=" * 60)
    
    # 创建新实例
    instance_id2 = await tool.create(ground_truth=ground_truth)
    print(f"创建的实例ID: {instance_id2}")
    
    # 执行代码
    result2, score2, metadata2 = await tool.execute(instance_id2, {"code": partial_solution})
    
    print(f"执行结果: {result2}")
    print(f"分数: {score2}")
    print(f"元数据: {metadata2}")
    
    # 测试用例3：错误的解决方案
    wrong_solution = """```python
def solution(input_str):
    lines = input_str.strip().split('\\n')
    t = int(lines[0])
    results = []
    
    for i in range(1, t + 1):
        s = lines[i]
        results.append("NO")  # 总是返回 NO
    
    return results
```"""
    
    print("\\n" + "=" * 60)
    print("测试用例3：错误的解决方案")
    print("=" * 60)
    
    # 创建新实例
    instance_id3 = await tool.create(ground_truth=ground_truth)
    print(f"创建的实例ID: {instance_id3}")
    
    # 执行代码
    result3, score3, metadata3 = await tool.execute(instance_id3, {"code": wrong_solution})
    
    print(f"执行结果: {result3}")
    print(f"分数: {score3}")
    print(f"通过样本数: {metadata3.get('passed_samples', 0)}")
    print(f"失败样本数: {metadata3.get('failed_samples', 0)}")
    
    # 测试用例4：语法错误的解决方案
    syntax_error_solution = """```python
def solution(input_str):
    lines = input_str.strip().split('\\n')
    t = int(lines[0])
    results = []
    
    for i in range(1, t + 1):
        s = lines[i]
        if s == "abc":
            results.append("YES")
        elif s in ["acb", "bac", "cba"]:
            results.append("YES")
        else:
            results.append("NO")
    
    return results  # 缺少右括号
```"""
    
    print("\\n" + "=" * 60)
    print("测试用例4：语法错误的解决方案")
    print("=" * 60)
    
    # 创建新实例
    instance_id4 = await tool.create(ground_truth=ground_truth)
    print(f"创建的实例ID: {instance_id4}")
    
    # 执行代码
    result4, score4, metadata4 = await tool.execute(instance_id4, {"code": syntax_error_solution})
    
    print(f"执行结果: {result4}")
    print(f"分数: {score4}")
    print(f"通过样本数: {metadata4.get('passed_samples', 0)}")
    print(f"失败样本数: {metadata4.get('failed_samples', 0)}")
    
    # 总结
    print("\\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    print(f"正确解决方案分数: {score}")
    print(f"部分正确解决方案分数: {score2}")
    print(f"错误解决方案分数: {score3}")
    print(f"语法错误解决方案分数: {score4}")
    
    # 验证分数合理性
    if score > score2 > score3:
        print("✅ 分数排序正确：正确 > 部分正确 > 错误")
    else:
        print("⚠️ 分数排序可能有问题")
    
    print("\\n测试完成！")

if __name__ == "__main__":
    asyncio.run(test_execute()) 