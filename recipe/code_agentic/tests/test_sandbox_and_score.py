import sys
import types
import importlib
import builtins

import pytest


def _install_stub_modules():
    # Minimal stubs for missing modules to import the target module
    stubs = {
        'verl.tools.base_tool': types.ModuleType('verl.tools.base_tool'),
        'verl.tools.sandbox_fusion_tools': types.ModuleType('verl.tools.sandbox_fusion_tools'),
        'verl.utils.dataset': types.ModuleType('verl.utils.dataset'),
        'verl.utils.rollout_trace': types.ModuleType('verl.utils.rollout_trace'),
        'verl.utils.reward_score': types.ModuleType('verl.utils.reward_score'),
        'recipe.code_agentic.code_agentic_utils': types.ModuleType('recipe.code_agentic.code_agentic_utils'),
        'datasets': types.ModuleType('datasets'),
        'omegaconf': types.ModuleType('omegaconf'),
    }

    # Ensure __spec__ exists to satisfy importlib expectations
    try:
        from importlib.machinery import ModuleSpec, BuiltinImporter
        def _ensure_spec(name: str, mod: types.ModuleType):
            if getattr(mod, '__spec__', None) is None:
                mod.__spec__ = ModuleSpec(name, BuiltinImporter)
    except Exception:
        def _ensure_spec(name: str, mod: types.ModuleType):
            setattr(mod, '__spec__', object())

    # Base tool schema stub
    class _OpenAIFunctionToolSchema:  # noqa: N801
        pass
    stubs['verl.tools.base_tool'].OpenAIFunctionToolSchema = _OpenAIFunctionToolSchema

    # SandboxFusionTool base class stub
    class _SandboxFusionTool:  # noqa: N801
        def __init__(self, config, tool_schema):
            self.config = config
            self.tool_schema = tool_schema
            self.sandbox_fusion_url = config.get('sandbox_fusion_url', 'http://stub/run')
            self.default_timeout = config.get('default_timeout', 5)
            self.default_language = config.get('default_language', 'python')
            self.memory_limit_mb = config.get('memory_limit_mb', 256)
            self._instance_dict = {}

            class _ExecPool:
                async def execute(self, *args, **kwargs):  # pragma: no cover
                    return ""

                class remote:  # noqa: N801
                    @staticmethod
                    async def __call__(*args, **kwargs):
                        return ""

            self.execution_pool = types.SimpleNamespace(execute=types.SimpleNamespace(remote=lambda *a, **k: None))

        async def execute_code(self, *args, **kwargs):  # pragma: no cover
            return ""

    stubs['verl.tools.sandbox_fusion_tools'].SandboxFusionTool = _SandboxFusionTool

    # RLHFDataset stub
    class _RLHFDataset:  # noqa: N801
        def __init__(self, *args, **kwargs):
            self.data_files = []
    stubs['verl.utils.dataset'].RLHFDataset = _RLHFDataset

    # rollout_trace decorator stub
    def _rollout_trace_op(func):
        return func
    stubs['verl.utils.rollout_trace'].rollout_trace_op = _rollout_trace_op

    # reward_score.sandbox_fusion placeholder
    class _SandboxFusionNS:
        @staticmethod
        def compute_score(**kwargs):  # will be monkeypatched per test
            return None, []
    stubs['verl.utils.reward_score'].sandbox_fusion = _SandboxFusionNS

    # code_agentic_utils functions stub
    def _extract_final_code_from_response(s):
        return ""

    def _wrap_code_for_execution(c):
        return c

    def _extract_code_pattern():
        import re as _re
        return _re.compile(r"```(?:python)?\n([\s\S]*?)```", _re.MULTILINE)

    def _preprocess_code(c):
        return c

    def _prepare_code_for_testing(c, t):
        return c

    def _get_code_problem_prompt(q):
        return str(q)

    def _get_answer_format():
        return ""

    cu = stubs['recipe.code_agentic.code_agentic_utils']
    cu.extract_final_code_from_response = _extract_final_code_from_response
    cu.wrap_code_for_execution = _wrap_code_for_execution
    cu.extract_code_pattern = _extract_code_pattern
    cu.preprocess_code = _preprocess_code
    cu.prepare_code_for_testing = _prepare_code_for_testing
    cu.get_code_problem_prompt = _get_code_problem_prompt
    cu.get_answer_format = _get_answer_format

    # datasets.load_dataset stub
    def _load_dataset(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("stub")
    stubs['datasets'].load_dataset = _load_dataset

    # OmegaConf stub
    class _OmegaConf:  # pragma: no cover
        pass
    stubs['omegaconf'].OmegaConf = _OmegaConf

    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)
        _ensure_spec(name, mod)


def _import_code_agentic():
    _install_stub_modules()
    return importlib.import_module('recipe.code_agentic.code_agentic')


def test_compute_code_score_shaping_rewards(monkeypatch):
    mod = _import_code_agentic()

    # Force code extraction and wrapping
    monkeypatch.setattr(mod, 'extract_final_code_from_response', lambda s: "print('ok')\n")
    monkeypatch.setattr(mod, 'wrap_code_for_execution', lambda c: f"```python\n{c}\n```")

    # Mock sandbox fusion compute_score to simulate pass/fail
    def _mock_compute_score(**kwargs):
        metadata_list = [
            {"status": "success", "case_index": 0, "stdout": "ok", "duration": 0.1},
            {"status": "error", "case_index": 1, "stdout": "", "duration": 0.1},
        ]
        return None, metadata_list
    monkeypatch.setattr(mod.sandbox_fusion, 'compute_score', _mock_compute_score)

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
    extra_info = {
        "num_turns": 2,
        "num_tool_calls": 3,
        "used_tools": ["code_interpreter"],
    }
    result = mod.compute_code_score("unit", "```python\nprint('ok')\n```", ground_truth, extra_info)
    assert "score" in result and isinstance(result["score"], float)
    # Passed samples = 2 (case_index 0 has 2), total = 3 -> base pass_rate ≈ 0.6667
    # With shaping, final score should be >= base pass_rate and <= 1
    assert 0.66 <= result["score"] <= 1.0


def test_compute_code_score_looks_like_code_without_extraction(monkeypatch):
    mod = _import_code_agentic()

    # Force extraction to fail but looks-like-code present
    monkeypatch.setattr(mod, 'extract_final_code_from_response', lambda s: "")
    monkeypatch.setattr(mod, 'wrap_code_for_execution', lambda c: c)

    # Make compute_score return empty results (no passes)
    monkeypatch.setattr(mod.sandbox_fusion, 'compute_score', lambda **k: (None, []))

    ground_truth = {
        "inputs": [
            "1\nabc",
        ],
        "outputs": [
            "cba",
        ],
    }
    extra_info = {
        "num_turns": 2,
        "num_tool_calls": 0,
        "used_tools": [],
    }
    # Include a code fence so it is treated as looks-like-code
    solution = "```\nprint('x')\n```"
    result = mod.compute_code_score("unit", solution, ground_truth, extra_info)
    assert 0.0 <= result["score"] <= 1.0
    # Should have non-zero shaping from format and turns even if pass_rate is 0
    assert result["score"] > 0.0


@pytest.mark.asyncio
async def test_sandbox_tool_execute_without_tests(monkeypatch):
    mod = _import_code_agentic()

    # Build tool
    schema = sys.modules['verl.tools.base_tool'].OpenAIFunctionToolSchema()
    tool = mod.SandboxFusionTestCaseTool(
        config={
            'sandbox_fusion_url': 'http://stub/run',
            'default_timeout': 5,
            'default_language': 'python',
            'memory_limit_mb': 256,
        },
        tool_schema=schema,
    )

    # Patch execution pool to return an object with a text field
    class _Exec:
        @staticmethod
        async def remote(*args, **kwargs):
            return types.SimpleNamespace(text='ok')
    tool.execution_pool = types.SimpleNamespace(execute=_Exec())

    instance_id = await tool.create()
    output, score, meta = await tool.execute(
        instance_id,
        parameters={
            'code': "```python\nprint('x')\n```",
            'timeout': 5,
            'language': 'python',
        },
    )

    assert output == 'ok'
    assert score == 0.0
    assert meta['execution_result'] == 'computed_without_test_cases'
    assert meta['total_tests'] == 0


@pytest.mark.asyncio
async def test_sandbox_tool_execute_with_tests(monkeypatch):
    mod = _import_code_agentic()

    # Build tool
    schema = sys.modules['verl.tools.base_tool'].OpenAIFunctionToolSchema()
    tool = mod.SandboxFusionTestCaseTool(
        config={
            'sandbox_fusion_url': 'http://stub/run',
            'default_timeout': 5,
            'default_language': 'python',
            'memory_limit_mb': 256,
        },
        tool_schema=schema,
    )

    # Patch execution pool to return string
    class _Exec2:
        @staticmethod
        async def remote(*args, **kwargs):
            return "run out"
    tool.execution_pool = types.SimpleNamespace(execute=_Exec2())

    # Mock compute_score
    def _mock_compute_score(**kwargs):
        metadata_list = [
            {"status": "success", "case_index": 0, "stdout": "ok", "duration": 0.1},
            {"status": "error", "case_index": 1, "stdout": "", "duration": 0.1},
        ]
        return None, metadata_list
    monkeypatch.setattr(mod.sandbox_fusion, 'compute_score', _mock_compute_score)

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

    instance_id = await tool.create(ground_truth=gt)
    output, score, meta = await tool.execute(
        instance_id,
        parameters={
            'code': "print('x')",
            'timeout': 5,
            'language': 'python',
        },
    )

    assert isinstance(output, str)
    assert 'Output:' in output
    assert 'Test Case Evaluation:' in output
    assert meta['execution_result'] == 'computed_with_sandbox_fusion'
    assert meta['passed_samples'] == 2
    assert meta['failed_samples'] == 1
    assert abs(meta['score'] - (2/3)) < 1e-6


