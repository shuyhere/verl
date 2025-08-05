# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import json
import logging
import os
from typing import Any, List, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op, RolloutTraceConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, trainer_config, server_manager, tokenizer, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, **kwargs)
        self._current_data_context = {}
        
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

        # Initialize scorers from config if available
        cls._init_scorers_from_config(config)

    @classmethod
    def _init_scorers_from_config(cls, config):
        """Initialize scorers from configuration."""
        try:
            # Check if trace config exists and scorers are enabled
            trace_config = config.actor_rollout_ref.rollout.get("trace", {})
            scorer_config = trace_config.get("scorers", {})
            
            if not scorer_config.get("enabled", False):
                logger.info("Scorer functionality is disabled in config")
                return
            
            # Load scorers from config
            scorers = cls._load_scorers_from_config(scorer_config)
            print(f"scorers: {scorers}")
            if scorers:
                print("Scorers loaded")
                # Initialize RolloutTraceConfig with scorers
                trainer_config = config.get("trainer", {})
                RolloutTraceConfig.init(
                    project_name=trainer_config.get("project_name", "tool_agent_project"),
                    experiment_name=trainer_config.get("experiment_name", "tool_agent_experiment"),
                    backend=trace_config.get("backend"),
                    token2text=trace_config.get("token2text", False),
                    scorers=scorers
                )
                logger.info(f"Initialized {len(scorers)} scorers for ToolAgentLoop")
            
        except Exception as e:
            logger.warning(f"Failed to initialize scorers from config: {e}")

    @classmethod
    def _load_scorers_from_config(cls, scorer_config):
        """Load scorers from configuration."""
        scorers = []
        scorer_list = scorer_config.get("scorer_list", {})
        
        print(f"Raw scorer_list: {scorer_list}")
        print(f"Type of scorer_list: {type(scorer_list)}")
        
        if hasattr(scorer_list, 'keys') and hasattr(scorer_list, 'values'):
            try:
                sorted_keys = sorted(scorer_list.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
                scorer_list = [scorer_list[key] for key in sorted_keys]
                print(f"Converted dict to list: {scorer_list}")
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not sort keys numerically: {e}")
                scorer_list = list(scorer_list.values())
                print(f"Using values directly: {scorer_list}")
        
        if not isinstance(scorer_list, list):
            print(f"Warning: scorer_list is not a list after conversion: {type(scorer_list)}")
            scorer_list = []

        for i, scorer_item in enumerate(scorer_list):
            print(f"Processing scorer {i}: {scorer_item}")
            try:
                scorer = cls._create_scorer_from_config(scorer_item)
                if scorer:
                    scorers.append(scorer)
                    logger.info(f"Loaded scorer: {scorer}")
                    print(f"Successfully loaded scorer: {scorer}")
                else:
                    print(f"Scorer {i} was not enabled or failed to create")
            except Exception as e:
                logger.error(f"Failed to load scorer {scorer_item.get('name', 'unknown')}: {e}")
                print(f"Error loading scorer {i}: {e}")
        
        print(f"Total scorers loaded: {len(scorers)}")
        return scorers

    @classmethod
    def _create_scorer_from_config(cls, scorer_config):
        """Create a scorer instance from configuration."""
        print(f"Creating scorer from config: {scorer_config}")
        
        scorer_type = scorer_config.get("type")
        name = scorer_config.get("name", scorer_type)
        enabled = scorer_config.get("enabled", False)
        
        print(f"Scorer type: {scorer_type}, name: {name}, enabled: {enabled}")
        
        if not enabled:
            print(f"Scorer {name} is disabled, skipping")
            return None
        
        if scorer_type == "custom":
            return cls._load_custom_scorer(scorer_config)
        elif scorer_type == "factual_accuracy":
            from verl.experimental.scores.scorer_examples import FactualAccuracyScorer
            print("Creating FactualAccuracyScorer")
            return FactualAccuracyScorer()
        elif scorer_type == "response_length":
            from verl.experimental.scores.scorer_examples import ResponseLengthScorer
            min_length = scorer_config.get("min_length", 10)
            max_length = scorer_config.get("max_length", 1000)
            print(f"Creating ResponseLengthScorer with min_length: {min_length}, max_length: {max_length}")
            return ResponseLengthScorer(min_length=min_length, max_length=max_length)
        elif scorer_type == "math_accuracy":
            from verl.experimental.scores.scorer_examples import MathAccuracyScorer
            print("Creating MathAccuracyScorer")
            return MathAccuracyScorer()
        elif scorer_type == "reward_hacking":
            from verl.experimental.scores.scorer_examples import RewardHackingScorer
            model = scorer_config.get("model", "gpt-4o-mini")
            print(f"Creating RewardHackingScorer with model: {model}")
            return RewardHackingScorer(model=model)
        elif scorer_type == "self_awareness":
            from verl.experimental.scores.scorer_examples import SelfAwarenessScorer
            model = scorer_config.get("model", "gpt-4o-mini")
            print(f"Creating self_awareness with model: {model}")
            return SelfAwarenessScorer(model=model)
        else:
            logger.warning(f"Unknown scorer type: {scorer_type}")
            print(f"Unknown scorer type: {scorer_type}")
            return None

    @classmethod
    def _load_custom_scorer(cls, scorer_config):
        """Load a custom scorer from class path."""
        try:
            import importlib
            class_path = scorer_config.get("class_path")
            if not class_path:
                logger.error("Custom scorer must specify class_path")
                return None
            
            # Dynamic import
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            scorer_class = getattr(module, class_name)
            
            # Create instance
            kwargs = {k: v for k, v in scorer_config.items() 
                     if k not in ["type", "name", "enabled", "class_path"]}
            return scorer_class(**kwargs)
            
        except Exception as e:
            logger.error(f"Failed to load custom scorer: {e}")
            return None

    @classmethod
    def add_scorer(cls, scorer):
        """Add a scorer to the current configuration."""
        try:
            RolloutTraceConfig.add_scorer(scorer)
            logger.info(f"Added scorer: {scorer}")
        except Exception as e:
            logger.error(f"Failed to add scorer: {e}")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
            ),
        )
        response_mask = []
        tools_kwargs = kwargs.get("tools_kwargs", {})

        user_turns, assistant_turns = 0, 0
        while True:
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1

            # reach max response length
            if len(response_mask) >= self.response_length:
                break

            # reach max assistant turns
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break

            # reach max user turns
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # no tool calls
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                break

            # call tools
            tasks = []
            for tool_call in tool_calls[: self.max_parallel_calls]:
                tasks.append(self._call_tool(tool_call, tools_kwargs))
            with simple_timer("tool_calls", metrics):
                tool_responses = await asyncio.gather(*tasks)
            if any(isinstance(item, Exception) for item in tool_responses):
                break

            # append tool_response_ids
            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                ),
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]

            # NOTE: last turn should not be user turn, or the EOS token reward
            # can't be propagated to previous token in GAE.
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            user_turns += 1

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        return output

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> dict[str, str]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]

            create_kwargs = {}
            if hasattr(self, '_current_data_context') and self._current_data_context:
                tools_kwargs = self._current_data_context.get('tools_kwargs', {})
                if tool_name in tools_kwargs:
                    create_kwargs = tools_kwargs[tool_name].get('create_kwargs', {})
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Found create_kwargs for tool {tool_name}: {create_kwargs}")

            instance_id = await tool.create(**create_kwargs)
            tool_response, tool_reward_score, tool_metrics = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        return {
            "role": "tool",
            "content": tool_response_text,
        }
