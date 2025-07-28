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

"""
示例：如何在 ToolAgentLoop 和其他使用 @rollout_trace_op 的类中集成自定义 scorer

这个示例展示了：
1. 如何定义自定义 scorer
2. 如何在配置中设置 scorer
3. 如何在训练过程中使用 scorer
"""

import os
from typing import List, Any

from verl.utils.rollout_trace import RolloutTraceConfig
from verl.utils.scorer_examples import (
    HallucinationScorer, 
    FactualAccuracyScorer, 
    ResponseLengthScorer,
    get_scorer_by_name,
    create_custom_scorer
)


def setup_scorers_for_training():
    """设置训练时使用的 scorer。"""
    
    # 方法1：在初始化时直接传入 scorer 列表
    scorers = [
        HallucinationScorer(model="gpt-4o-mini"),
        FactualAccuracyScorer(),
        ResponseLengthScorer(min_length=20, max_length=500)
    ]
    
    # 初始化 RolloutTraceConfig 时包含 scorer
    RolloutTraceConfig.init(
        project_name="my_project",
        experiment_name="my_experiment", 
        backend="weave",
        token2text=True,
        scorers=scorers
    )


def setup_scorers_dynamically():
    """动态添加 scorer。"""
    
    # 先初始化配置
    RolloutTraceConfig.init(
        project_name="my_project",
        experiment_name="my_experiment",
        backend="weave", 
        token2text=True
    )
    
    # 然后动态添加 scorer
    RolloutTraceConfig.add_scorer(HallucinationScorer())
    RolloutTraceConfig.add_scorer(FactualAccuracyScorer())
    RolloutTraceConfig.add_scorer(ResponseLengthScorer())


def setup_scorers_from_config():
    """从配置文件设置 scorer。"""
    
    # 假设从配置文件读取 scorer 配置
    scorer_configs = [
        {"type": "hallucination", "model": "gpt-4o-mini"},
        {"type": "factual_accuracy"},
        {"type": "response_length", "min_length": 30, "max_length": 800}
    ]
    
    scorers = []
    for config in scorer_configs:
        scorer_type = config.pop("type")
        scorer = create_custom_scorer(scorer_type, **config)
        scorers.append(scorer)
    
    RolloutTraceConfig.init(
        project_name="my_project",
        experiment_name="my_experiment",
        backend="weave",
        token2text=True,
        scorers=scorers
    )


class CustomToolAgentLoop:
    """自定义 ToolAgentLoop 示例，展示如何集成 scorer。"""
    
    def __init__(self, scorer_names: List[str] = None):
        self.scorer_names = scorer_names or ["hallucination", "factual_accuracy"]
        self._setup_scorers()
    
    def _setup_scorers(self):
        """设置 scorer。"""
        scorers = []
        for name in self.scorer_names:
            scorer = get_scorer_by_name(name, use_weave=True)
            if scorer:
                scorers.append(scorer)
        
        # 初始化 trace 配置
        RolloutTraceConfig.init(
            project_name="custom_tool_agent",
            experiment_name="with_scorers",
            backend="weave",
            token2text=True,
            scorers=scorers
        )
    
    def run_with_scorers(self, messages, sampling_params):
        """运行带有 scorer 的 agent loop。"""
        # 这里会调用被 @rollout_trace_op 装饰的方法
        # scorer 会自动应用到每个 trace 调用上
        pass


def example_usage_in_training_script():
    """在训练脚本中的使用示例。"""
    
    # 1. 设置环境变量
    os.environ["WANDB_API_KEY"] = "your_wandb_api_key"
    
    # 2. 配置 scorer
    setup_scorers_for_training()
    
    # 3. 运行训练（scorer 会自动应用到所有 @rollout_trace_op 装饰的方法）
    # python -m verl.trainer.main_ppo --config-path=configs --config-name=your_config


def example_custom_scorer():
    """自定义 scorer 示例。"""
    
    class CustomMathAccuracyScorer:
        """自定义数学准确性 scorer。"""
        
        def __init__(self):
            self.name = "math_accuracy"
        
        def score(self, output: str, ground_truth: str, **kwargs) -> dict:
            """评估数学答案的准确性。"""
            try:
                # 简单的数学表达式评估
                import re
                
                # 提取数字和运算符
                output_numbers = re.findall(r'\d+\.?\d*', output)
                ground_truth_numbers = re.findall(r'\d+\.?\d*', ground_truth)
                
                if not ground_truth_numbers:
                    return {"score": 0.0, "error": "No numbers in ground truth"}
                
                # 检查数字是否匹配
                matches = 0
                for num in ground_truth_numbers:
                    if num in output_numbers:
                        matches += 1
                
                accuracy = matches / len(ground_truth_numbers)
                
                return {
                    "scorer_name": self.name,
                    "score": accuracy,
                    "details": {
                        "output_numbers": output_numbers,
                        "ground_truth_numbers": ground_truth_numbers,
                        "matches": matches
                    }
                }
                
            except Exception as e:
                return {
                    "scorer_name": self.name,
                    "score": 0.0,
                    "error": str(e)
                }
    
    # 使用自定义 scorer
    custom_scorer = CustomMathAccuracyScorer()
    RolloutTraceConfig.add_scorer(custom_scorer)


if __name__ == "__main__":
    # 运行示例
    print("设置 scorer 示例...")
    setup_scorers_for_training()
    
    print("当前配置的 scorer:")
    scorers = RolloutTraceConfig.get_scorers()
    for scorer in scorers:
        print(f"  - {scorer}")
    
    print("\n示例完成！") 