#!/usr/bin/env python3
"""
测试脚本：测试 compute_code_score 函数
使用真实的 codeforces 样本数据
"""

import asyncio
import logging
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from recipe.code_agentic.code_agentic import compute_code_score
    print("✅ 成功导入 compute_code_score")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    # 尝试直接导入
    sys.path.insert(0, os.path.dirname(__file__))
    from code_agentic import compute_code_score

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_compute_code_score():
    """测试 compute_code_score 函数"""
    
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

    extra_info = {		
        "num_turns": 3,
        "sandbox_fusion_url": "http://10.68.171.9:8080/run_code",
        "memory_limit_mb": 1024
    }

    correct_solution = """<code>
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
</code>"""

    print("=" * 80)
    print("测试用例1：正确的解决方案")
    print("=" * 80)
    
    result1 = compute_code_score("test_source", correct_solution, ground_truth, extra_info)
    
    print(f"分数: {result1.get('score')}")
    
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

    print("\n" + "=" * 80)
    print("测试用例2：部分正确的解决方案")
    print("=" * 80)
    
    result2 = compute_code_score("test_source", partial_solution, ground_truth, extra_info)
    
    print(f"分数: {result2.get('score')}")

    
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

    print("\n" + "=" * 80)
    print("测试用例3：错误的解决方案")
    print("=" * 80)
    
    result3 = compute_code_score("test_source", wrong_solution, ground_truth, extra_info)
    
    print(f"分数: {result3.get('score')}")
    
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

    print("\n" + "=" * 80)
    print("测试用例4：语法错误的解决方案")
    print("=" * 80)
    
    result4 = compute_code_score("test_source", syntax_error_solution, ground_truth, extra_info)
    
    print(f"分数: {result4.get('score')}")

    
    # 测试用例5：空字符串
    print("\n" + "=" * 80)
    print("测试用例5：空字符串")
    print("=" * 80)
    
    result5 = compute_code_score("test_source", "", ground_truth, extra_info)
    
    print(f"分数: {result5.get('score')}")
    print(f"预测: {result5.get('pred')}")
    print(f"通过测试数: {result5.get('passed_count')}")
    print(f"失败测试数: {result5.get('failed_count')}")
    print(f"总测试数: {result5.get('total_tests')}")
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    print(f"正确解决方案分数: {result1.get('score')}")
    print(f"部分正确解决方案分数: {result2.get('score')}")
    print(f"错误解决方案分数: {result3.get('score')}")
    print(f"语法错误解决方案分数: {result4.get('score')}")
    print(f"空字符串分数: {result5.get('score')}")
    
    # 验证分数合理性
    if result1.get('score') > result2.get('score') > result3.get('score'):
        print("✅ 分数排序正确：正确 > 部分正确 > 错误")
    else:
        print("⚠️ 分数排序可能有问题")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_compute_code_score() 