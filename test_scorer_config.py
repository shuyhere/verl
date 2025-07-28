#!/usr/bin/env python3
"""
测试 scorer 配置加载
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'verl'))

from omegaconf import OmegaConf
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.utils.rollout_trace import RolloutTraceConfig

def test_scorer_config():
    """测试 scorer 配置加载。"""
    
    # 模拟你的配置
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "trace": {
                    "backend": "weave",
                    "token2text": True,
                    "scorers": {
                        "enabled": True,
                        "scorer_list": {
                            "0": {
                                "name": "rewardhacking",
                                "type": "rewardhacking",
                                "model": "gpt-4o-mini",
                                "enabled": True
                            }
                        }
                    }
                }
            }
        },
        "trainer": {
            "project_name": "test_project",
            "experiment_name": "test_experiment"
        }
    }
    
    print("测试配置:")
    print(f"scorer_config: {config['actor_rollout_ref']['rollout']['trace']['scorers']}")
    
    # 测试 ToolAgentLoop 的 scorer 加载
    try:
        # 将字典转换为 OmegaConf 对象
        config_obj = OmegaConf.create(config) 
        print(f"config_obj: {config_obj}")
        ToolAgentLoop._init_scorers_from_config(config_obj)
        
        # 检查是否成功加载
        scorers = RolloutTraceConfig.get_scorers()
        print(f"scorers: {scorers}")
        print(f"\n成功加载 {len(scorers)} 个 scorer:")
        for scorer in scorers:
            print(f"  - {scorer}")
        
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试 scorer 配置...")
    success = test_scorer_config()
    if success:
        print("\n✅ 测试通过！")
    else:
        print("\n❌ 测试失败！") 