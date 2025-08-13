# ToolAgentLoop Scorer 集成指南

本指南介绍如何在 VERL 的 `ToolAgentLoop` 中集成和使用 scorer 功能。

## 概述

Scorer 功能允许你在训练过程中自动评估模型输出的质量。当使用 Weave 作为 trace 后端时，scorer 会自动应用到每个 trace 调用上，提供额外的评估指标。

## 快速开始

### 1. 基本设置

```python
from verl.utils.rollout_trace import RolloutTraceConfig
from verl.utils.scorer_examples import HallucinationScorer, FactualAccuracyScorer

# 设置 scorer
scorers = [
    HallucinationScorer(model="gpt-4o-mini"),
    FactualAccuracyScorer()
]

# 初始化配置
RolloutTraceConfig.init(
    project_name="my_project",
    experiment_name="my_experiment",
    backend="weave",
    token2text=True,
    scorers=scorers
)
```

### 2. 使用配置文件

创建配置文件 `configs/tool_agent_with_scorers.yaml`：

```yaml
actor_rollout_ref:
  rollout:
    trace:
      backend: "weave"
      token2text: true
      scorers:
        enabled: true
        scorer_list:
          - name: "hallucination"
            type: "hallucination"
            model: "gpt-4o-mini"
            enabled: true
          - name: "factual_accuracy"
            type: "factual_accuracy"
            enabled: true
          - name: "response_length"
            type: "response_length"
            min_length: 20
            max_length: 500
            enabled: true
```

然后在训练脚本中加载：

```bash
python -m verl.trainer.main_ppo --config-path=configs --config-name=tool_agent_with_scorers
```

## 内置 Scorer

### 1. HallucinationScorer

检测模型输出是否包含幻觉信息。

```python
from verl.utils.scorer_examples import HallucinationScorer

scorer = HallucinationScorer(model="gpt-4o-mini")
```

**参数：**
- `model`: 用于评估的模型名称（默认：gpt-4o-mini）

### 2. FactualAccuracyScorer

评估输出的事实准确性。

```python
from verl.utils.scorer_examples import FactualAccuracyScorer

scorer = FactualAccuracyScorer()
```

### 3. ResponseLengthScorer

评估响应长度是否合适。

```python
from verl.utils.scorer_examples import ResponseLengthScorer

scorer = ResponseLengthScorer(min_length=20, max_length=500)
```

**参数：**
- `min_length`: 最小长度（默认：10）
- `max_length`: 最大长度（默认：1000）

### 4. MathAccuracyScorer

评估数学答案的准确性。

```python
from verl.utils.scorer_examples import MathAccuracyScorer

scorer = MathAccuracyScorer()
```

## 自定义 Scorer

### 1. 创建自定义 Scorer

```python
class CustomCodeQualityScorer:
    """自定义代码质量 scorer。"""
    
    def __init__(self):
        self.name = "code_quality"
    
    def score(self, output: str, ground_truth: str, **kwargs) -> dict:
        """评估代码质量。"""
        try:
            score = 0.0
            details = {}
            
            # 检查代码长度
            if len(output) > 50:
                score += 0.3
                details["length_ok"] = True
            
            # 检查是否包含函数定义
            if "def " in output:
                score += 0.4
                details["has_function"] = True
            
            # 检查是否包含注释
            if "#" in output:
                score += 0.3
                details["has_comments"] = True
            
            return {
                "scorer_name": self.name,
                "score": score,
                "details": details
            }
            
        except Exception as e:
            return {
                "scorer_name": self.name,
                "score": 0.0,
                "error": str(e)
            }
```

### 2. 注册自定义 Scorer

```python
from verl.utils.rollout_trace import RolloutTraceConfig

custom_scorer = CustomCodeQualityScorer()
RolloutTraceConfig.add_scorer(custom_scorer)
```

### 3. 在配置文件中使用

```yaml
actor_rollout_ref:
  rollout:
    trace:
      scorers:
        enabled: true
        scorer_list:
          - name: "code_quality"
            type: "custom"
            class_path: "my_module.CustomCodeQualityScorer"
            enabled: true
```

## 在 ToolAgentLoop 中使用

### 1. 自动集成

当你在 `ToolAgentLoop` 中使用 `@rollout_trace_op` 装饰器时，scorer 会自动应用：

```python
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

# ToolAgentLoop 的 run 方法已经被 @rollout_trace_op 装饰
# scorer 会自动应用到每个调用
```

### 2. 查看结果

在 Weave 界面中，你可以看到：
- 每个 trace 调用的 scorer 结果
- 详细的评分信息
- 评分趋势和分布

## 配置选项

### Trace 配置

```yaml
actor_rollout_ref:
  rollout:
    trace:
      # 必需：trace 后端
      backend: "weave"  # 或 "mlflow"
      
      # 可选：是否转换 token 为文本
      token2text: true
      
      # 可选：scorer 配置
      scorers:
        enabled: true
        scorer_list:
          - name: "scorer_name"
            type: "scorer_type"
            enabled: true
            # 其他 scorer 特定参数
```

### Scorer 配置参数

每个 scorer 可以包含以下参数：

- `name`: scorer 名称（必需）
- `type`: scorer 类型（必需）
- `enabled`: 是否启用（默认：true）
- `class_path`: 自定义 scorer 的类路径（仅用于 custom 类型）
- 其他 scorer 特定参数

## 故障排除

### 常见问题

1. **Scorer 未生效**
   - 检查 `backend` 是否设置为 "weave"
   - 确认 scorer 已正确添加到配置中
   - 验证 Weave API 密钥是否正确设置

2. **自定义 Scorer 加载失败**
   - 检查 `class_path` 是否正确
   - 确认类实现了正确的接口
   - 查看日志中的错误信息

3. **性能问题**
   - 考虑使用异步 scorer
   - 优化 scorer 的实现逻辑
   - 减少不必要的 API 调用

### 调试技巧

```python
# 启用详细日志
import logging
logging.getLogger("verl.utils.rollout_trace").setLevel(logging.DEBUG)

# 检查当前配置的 scorer
scorers = RolloutTraceConfig.get_scorers()
print(f"Configured scorers: {scorers}")
```

## 示例

查看 `verl/examples/tool_agent_with_scorers_example.py` 获取完整的使用示例。

## 注意事项

1. 确保已安装 weave: `pip install weave`
2. 设置正确的 `WANDB_API_KEY` 环境变量
3. 如果使用 OpenAI API，确保设置了 `OPENAI_API_KEY`
4. 在 Weave 界面中查看 scorer 结果

## 许可证

本功能遵循 VERL 项目的 Apache 2.0 许可证。 