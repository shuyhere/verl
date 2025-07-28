# VERL Scorer 集成指南

本指南介绍如何在 VERL 的 `tool_agent_loop` 和其他使用 `@rollout_trace_op` 装饰器的类中集成自定义 scorer。

## 概述

Scorer 功能允许你在训练过程中自动评估模型输出的质量。当使用 Weave 作为 trace 后端时，scorer 会自动应用到每个 trace 调用上，提供额外的评估指标。

## 功能特性

- ✅ 支持多种 scorer 类型（幻觉检测、事实准确性、响应长度等）
- ✅ 与 Weave trace 系统无缝集成
- ✅ 支持自定义 scorer 开发
- ✅ 配置文件驱动的 scorer 管理
- ✅ 动态 scorer 添加和移除

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

### 2. 在训练脚本中使用

```python
# 在你的训练脚本中
from verl.utils.scorer_config_loader import quick_setup_scorers

# 快速设置常用 scorer
quick_setup_scorers(
    ["hallucination", "factual_accuracy", "response_length"],
    project_name="my_training_project",
    experiment_name="with_scorers",
    backend="weave"
)

# 运行训练 - scorer 会自动应用到所有 @rollout_trace_op 装饰的方法
```

### 3. 使用配置文件

创建配置文件 `configs/rollout_trace_with_scorers.yaml`：

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
```

然后在训练脚本中加载：

```python
from verl.utils.scorer_config_loader import setup_scorers_from_yaml_config

setup_scorers_from_yaml_config("configs/rollout_trace_with_scorers.yaml")
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

## 自定义 Scorer

### 1. 创建自定义 Scorer

```python
class CustomMathAccuracyScorer:
    """自定义数学准确性 scorer。"""
    
    def __init__(self):
        self.name = "math_accuracy"
    
    def score(self, output: str, ground_truth: str, **kwargs) -> dict:
        """评估数学答案的准确性。"""
        try:
            # 你的评分逻辑
            import re
            
            # 提取数字
            output_numbers = re.findall(r'\d+\.?\d*', output)
            ground_truth_numbers = re.findall(r'\d+\.?\d*', ground_truth)
            
            if not ground_truth_numbers:
                return {"score": 0.0, "error": "No numbers in ground truth"}
            
            # 计算匹配率
            matches = sum(1 for num in ground_truth_numbers if num in output_numbers)
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
```

### 2. 注册自定义 Scorer

```python
from verl.utils.rollout_trace import RolloutTraceConfig

custom_scorer = CustomMathAccuracyScorer()
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
          - name: "math_accuracy"
            type: "custom"
            class_path: "my_module.CustomMathAccuracyScorer"
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

## 高级用法

### 1. 动态添加 Scorer

```python
from verl.utils.rollout_trace import RolloutTraceConfig

# 在运行时添加 scorer
RolloutTraceConfig.add_scorer(NewScorer())
```

### 2. 条件性 Scorer

```python
def conditional_scorer_setup(condition):
    if condition:
        RolloutTraceConfig.add_scorer(HallucinationScorer())
    else:
        RolloutTraceConfig.add_scorer(FactualAccuracyScorer())
```

### 3. 批量 Scorer 管理

```python
from verl.utils.scorer_config_loader import ScorerConfigLoader

loader = ScorerConfigLoader()

# 从配置加载多个 scorer
scorers = loader.load_scorers_from_config(config_dict)

# 批量设置
RolloutTraceConfig.init(
    project_name="batch_project",
    experiment_name="batch_experiment",
    backend="weave",
    scorers=scorers
)
```

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

## 最佳实践

1. **选择合适的 Scorer**
   - 根据任务类型选择相关的 scorer
   - 避免使用过多 scorer 影响性能

2. **配置管理**
   - 使用配置文件管理 scorer 设置
   - 为不同实验使用不同的 scorer 配置

3. **性能优化**
   - 缓存 scorer 结果
   - 使用批量处理
   - 考虑异步执行

4. **监控和维护**
   - 定期检查 scorer 性能
   - 更新 scorer 逻辑
   - 监控 API 使用量

## 示例项目

查看 `verl/examples/scorer_usage_example.py` 获取完整的使用示例。

## 贡献

欢迎贡献新的 scorer 类型和改进建议！

## 许可证

本功能遵循 VERL 项目的 Apache 2.0 许可证。 