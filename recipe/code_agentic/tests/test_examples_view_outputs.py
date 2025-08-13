import json
import types
import sys

import pytest

from recipe.code_agentic.tests.test_sandbox_and_score import _install_stub_modules, _import_code_agentic


def test_example_compute_code_score_output(monkeypatch):
    _install_stub_modules()
    mod = _import_code_agentic()

    monkeypatch.setattr(mod, 'extract_final_code_from_response', lambda s: "print('ok')\n")
    monkeypatch.setattr(mod, 'wrap_code_for_execution', lambda c: f"```python\n{c}\n```")

    def _mock_compute_score(**kwargs):
        metadata_list = [
            {"status": "success", "case_index": 0, "stdout": "ok", "duration": 0.1},
            {"status": "error", "case_index": 1, "stdout": "", "duration": 0.1},
        ]
        return None, metadata_list

    monkeypatch.setattr(mod.sandbox_fusion, 'compute_score', _mock_compute_score)

    # Original ground_truth for regular examples
    ground_truth = {
        "inputs": [
            "2\nfoo\nbar",
            "1\nbaz",
        ],
        "outputs": [
            "out_foo\nout_bar",
            "out_baz",
        ],
    }
    
    # Ground truth that matches the real_solution function
    # solution(N, Q, P_list) -> (total_sum, result)
    real_ground_truth = {
        "inputs": [
            "5\n3\n[1, 3, 5]",  # N=5, Q=3, P_list=[1,3,5] -> A=[1,1,2,2,3], total=9, result=[1,2,3]
            "3\n2\n[2, 3]",     # N=3, Q=2, P_list=[2,3] -> A=[1,1,2], total=4, result=[1,2]
        ],
        "outputs": [
            "(9, [1, 2, 3])",
            "(4, [1, 2])",
        ],
    }
    extra_info = {
        "num_turns": 2,
        "num_tool_calls": 3,
        "used_tools": ["code_interpreter"],
    }

    # Test with user's real example first
    real_solution = """def solution(N, Q, P_list):
    A = [1] * N
    for i in range(1, N):
        # Calculate the value based on previous calculations to minimize total sum
        A[i] = A[i-1] if A[i] == 1 else A[i-1] + 1
    total_sum = sum(A)
    result = [A[p-1] for p in P_list]
    return total_sum, result"""
    
    print("=== TESTING REAL USER EXAMPLE ===")
    real_result = mod.compute_code_score("user_example", real_solution, real_ground_truth, extra_info)
    print("REAL_USER_COMPUTE_CODE_SCORE_RESULT=")
    print(json.dumps(real_result, ensure_ascii=False, indent=2))
    
    # Test with regular example
    print("\n=== TESTING REGULAR EXAMPLE ===")
    result = mod.compute_code_score("example", "```python\nprint('ok')\n```", ground_truth, extra_info)

    print("COMPUTE_CODE_SCORE_RESULT=")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    assert isinstance(result, dict)
    assert "score" in result
    assert "details" in result


@pytest.mark.asyncio
async def test_example_tool_output(monkeypatch):
    _install_stub_modules()
    mod = _import_code_agentic()

    schema = sys.modules['verl.tools.base_tool'].OpenAIFunctionToolSchema()
    tool = mod.SandboxFusionTestCaseTool(
        config={
            'sandbox_fusion_url': 'http://10.68.171.9:8080/run_code',
            'default_timeout': 5,
            'default_language': 'python',
            'memory_limit_mb': 256,
        },
        tool_schema=schema,
    )

    class _Exec:
        @staticmethod
        async def remote(*args, **kwargs):
            return types.SimpleNamespace(text='example run output')
    tool.execution_pool = types.SimpleNamespace(execute=_Exec())

    def _mock_compute_score(**kwargs):
        metadata_list = [
            {"status": "success", "case_index": 0, "stdout": "ok", "duration": 0.1},
            {"status": "error", "case_index": 1, "stdout": "", "duration": 0.1},
        ]
        return None, metadata_list
    monkeypatch.setattr(mod.sandbox_fusion, 'compute_score', _mock_compute_score)

    # Ground truth for regular tool test
    gt = {
        'inputs': [
            '2\na\nb',
            '1\nc',
        ],
        'outputs': [
            'x\ny',
            'z',
        ],
    }
    
    # Ground truth that matches the real_solution function for tool test
    real_gt = {
        'inputs': [
            '5\n3\n[1, 3, 5]',  # N=5, Q=3, P_list=[1,3,5] -> A=[1,1,2,2,3], total=9, result=[1,2,3]
            '3\n2\n[2, 3]',     # N=3, Q=2, P_list=[2,3] -> A=[1,1,2], total=4, result=[1,2]
        ],
        'outputs': [
            '(9, [1, 2, 3])',
            '(4, [1, 2])',
        ],
    }

    instance_id = await tool.create(ground_truth=gt)
    real_instance_id = await tool.create(ground_truth=real_gt)
    # Test with user's real example first
    real_solution = """def solution(N, Q, P_list):
    A = [1] * N
    for i in range(1, N):
        # Calculate the value based on previous calculations to minimize total sum
        A[i] = A[i-1] if A[i] == 1 else A[i-1] + 1
    total_sum = sum(A)
    result = [A[p-1] for p in P_list]
    return total_sum, result"""
    
    print("\n=== TESTING REAL USER EXAMPLE WITH TOOL ===")
    real_output, real_score, real_meta = await tool.execute(
        real_instance_id,
        parameters={
            'code': real_solution,
            'timeout': 5,
            'language': 'python',
        },
    )
    
    print("REAL_USER_SANDBOX_TOOL_OUTPUT=")
    print(real_output)
    print("REAL_USER_SANDBOX_TOOL_METADATA=")
    print(json.dumps(real_meta, ensure_ascii=False, indent=2))
    
    print("\n=== TESTING REGULAR EXAMPLE WITH TOOL ===")
    output, score, meta = await tool.execute(
        instance_id,
        parameters={
            'code': "print('x')",
            'timeout': 5,
            'language': 'python',
        },
    )

    print("SANDBOX_TOOL_OUTPUT=")
    print(output)
    print("SANDBOX_TOOL_METADATA=")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    assert isinstance(output, str)
    assert isinstance(meta, dict)
    assert "score" in meta

