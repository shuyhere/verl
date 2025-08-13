#!/usr/bin/env python3
"""
test scorer config loading  
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'verl'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'verl/verl'))
from omegaconf import OmegaConf
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.utils.rollout_trace import RolloutTraceConfig

def test_scorer_config():
    """test scorer config loading"""
    
    # mock config
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
                                "name": "reward_hacking",
                                "type": "reward_hacking",
                                "model": "gpt-4o-mini",
                                "enabled": True
                            },
                            "1": {
                                "name": "self_awareness",
                                "type": "self_awareness",
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
    
    print("test config:")
    print(f"scorer_config: {config['actor_rollout_ref']['rollout']['trace']['scorers']}")
    
    # test ToolAgentLoop scorer loading
    try:
        config_obj = OmegaConf.create(config) 
        print(f"config_obj: {config_obj}")
        ToolAgentLoop._init_scorers_from_config(config_obj)
        
 
        scorers = RolloutTraceConfig.get_scorers()
        print(f"scorers: {scorers}")
        print(f"\nsuccessfully loaded {len(scorers)} scorers:")
        for scorer in scorers:
            print(f"  - {scorer}")
        
        return True
        
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("start test scorer config...")
    success = test_scorer_config()
    if success:
        print("\n✅ test passed!")
    else:
        print("\n❌ test failed!") 
        