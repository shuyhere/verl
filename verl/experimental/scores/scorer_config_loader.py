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
Scorer 配置加载器

这个模块提供了从配置文件加载和初始化 scorer 的功能。
"""

import importlib
import logging
from typing import Any, Dict, List, Optional

from verl.utils.rollout_trace import RolloutTraceConfig
from verl.experimental.scores.scorer_examples import (
    HallucinationScorer,
    FactualAccuracyScorer, 
    ResponseLengthScorer,
    create_custom_scorer
)

logger = logging.getLogger(__name__)


class ScorerConfigLoader:
    """Scorer 配置加载器类。"""
    
    def __init__(self):
        self.available_scorers = {
            "hallucination": HallucinationScorer,
            "factual_accuracy": FactualAccuracyScorer,
            "response_length": ResponseLengthScorer,
        }
    
    def load_scorer_from_config(self, scorer_config: Dict[str, Any]) -> Optional[Any]:
        """从配置字典加载单个 scorer。"""
        try:
            scorer_type = scorer_config.get("type")
            name = scorer_config.get("name", scorer_type)
            enabled = scorer_config.get("enabled", True)
            
            if not enabled:
                logger.info(f"Scorer {name} is disabled, skipping...")
                return None
            
            if scorer_type == "custom":
                return self._load_custom_scorer(scorer_config)
            elif scorer_type in self.available_scorers:
                scorer_class = self.available_scorers[scorer_type]
                # 移除已知参数，其余作为 kwargs
                kwargs = {k: v for k, v in scorer_config.items() 
                         if k not in ["type", "name", "enabled"]}
                return scorer_class(**kwargs)
            else:
                logger.warning(f"Unknown scorer type: {scorer_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load scorer {scorer_config.get('name', 'unknown')}: {e}")
            return None
    
    def _load_custom_scorer(self, scorer_config: Dict[str, Any]) -> Optional[Any]:
        """加载自定义 scorer。"""
        try:
            class_path = scorer_config.get("class_path")
            if not class_path:
                logger.error("Custom scorer must specify class_path")
                return None
            
            # 动态导入类
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            scorer_class = getattr(module, class_name)
            
            # 创建实例
            kwargs = {k: v for k, v in scorer_config.items() 
                     if k not in ["type", "name", "enabled", "class_path"]}
            return scorer_class(**kwargs)
            
        except Exception as e:
            logger.error(f"Failed to load custom scorer {scorer_config.get('name', 'unknown')}: {e}")
            return None
    
    def load_scorers_from_config(self, config: Dict[str, Any]) -> List[Any]:
        """从完整配置加载所有 scorer。"""
        scorers = []
        
        # 检查是否启用了 scorer 功能
        trace_config = config.get("actor_rollout_ref", {}).get("rollout", {}).get("trace", {})
        scorer_config = trace_config.get("scorers", {})
        
        if not scorer_config.get("enabled", False):
            logger.info("Scorer functionality is disabled in config")
            return scorers
        
        scorer_list = scorer_config.get("scorer_list", [])
        
        for scorer_config_item in scorer_list:
            scorer = self.load_scorer_from_config(scorer_config_item)
            if scorer:
                scorers.append(scorer)
                logger.info(f"Loaded scorer: {scorer}")
        
        return scorers
    
    def setup_scorers_from_config(self, config: Dict[str, Any]) -> None:
        """从配置设置 scorer 并初始化 RolloutTraceConfig。"""
        scorers = self.load_scorers_from_config(config)
        
        if not scorers:
            logger.info("No scorers loaded from config")
            return
        
        # 获取 trace 配置
        trace_config = config.get("actor_rollout_ref", {}).get("rollout", {}).get("trace", {})
        trainer_config = config.get("trainer", {})
        
        # 初始化 RolloutTraceConfig
        RolloutTraceConfig.init(
            project_name=trainer_config.get("project_name", "default_project"),
            experiment_name=trainer_config.get("experiment_name", "default_experiment"),
            backend=trace_config.get("backend"),
            token2text=trace_config.get("token2text", False),
            scorers=scorers
        )
        
        logger.info(f"Initialized {len(scorers)} scorers for rollout trace")


def setup_scorers_from_yaml_config(config_path: str) -> None:
    """从 YAML 配置文件设置 scorer。"""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        loader = ScorerConfigLoader()
        loader.setup_scorers_from_config(config)
        
    except Exception as e:
        logger.error(f"Failed to load scorer config from {config_path}: {e}")


def add_scorer_to_existing_config(scorer_config: Dict[str, Any]) -> None:
    """向现有的 RolloutTraceConfig 添加 scorer。"""
    loader = ScorerConfigLoader()
    scorer = loader.load_scorer_from_config(scorer_config)
    
    if scorer:
        RolloutTraceConfig.add_scorer(scorer)
        logger.info(f"Added scorer: {scorer}")
    else:
        logger.warning(f"Failed to add scorer from config: {scorer_config}")


# 便捷函数
def quick_setup_scorers(scorer_types: List[str], **kwargs) -> None:
    """快速设置常用 scorer。"""
    scorers = []
    
    for scorer_type in scorer_types:
        if scorer_type == "hallucination":
            model = kwargs.get("model", "gpt-4o-mini")
            scorers.append(HallucinationScorer(model=model))
        elif scorer_type == "factual_accuracy":
            scorers.append(FactualAccuracyScorer())
        elif scorer_type == "response_length":
            min_length = kwargs.get("min_length", 10)
            max_length = kwargs.get("max_length", 1000)
            scorers.append(ResponseLengthScorer(min_length=min_length, max_length=max_length))
        else:
            logger.warning(f"Unknown scorer type: {scorer_type}")
    
    if scorers:
        RolloutTraceConfig.init(
            project_name=kwargs.get("project_name", "quick_setup"),
            experiment_name=kwargs.get("experiment_name", "with_scorers"),
            backend=kwargs.get("backend", "weave"),
            token2text=kwargs.get("token2text", True),
            scorers=scorers
        )
        logger.info(f"Quick setup completed with {len(scorers)} scorers") 