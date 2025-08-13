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

import asyncio
import contextlib
import functools
import inspect
import os
from typing import Optional, List, Any


class RolloutTraceConfig:
    """Configuration for rollout tracing with various backends.

    Singleton configuration class for managing rollout trace settings across different
    tracing backends like Weave and MLflow.

    Args:
        backend (Optional[str]): Tracing backend to use ('weave', 'mlflow', or None).
        client (Optional[object]): Client instance for the selected backend.
        token2text (bool): Whether to convert tokens to text in traces. Defaults to False.
        project_name (str): Name of the project for tracing.
        experiment_name (str): Name of the experiment for tracing.
        scorers (List[Any]): List of scorers to apply to traces.
    """

    _instance: Optional["RolloutTraceConfig"] = None
    backend: Optional[str] = None
    client: Optional[object] = None
    token2text: bool = False
    _initialized: bool = False
    project_name: str = None
    experiment_name: str = None
    scorers: List[Any] = []

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RolloutTraceConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def init(cls, project_name: str, experiment_name: str, backend: str, token2text: bool = False, scorers: List[Any] = None):
        config = cls.get_instance()
        
        # 如果已经初始化，只更新 scorers（如果提供了的话）
        if config._initialized:
            if scorers is not None:
                config.scorers = scorers
                print(f"Updated scorers in existing config: {len(scorers)} scorers")
            return

        config.backend = backend
        config.token2text = token2text
        config.project_name = project_name
        config.experiment_name = experiment_name
        config.scorers = scorers or []

        if backend == "weave":
            import weave

            config.client = weave.init(project_name)
        elif backend == "mlflow":
            import mlflow

            mlflow.config.enable_async_logging()
            config.client = mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            mlflow.set_experiment(project_name)
        else:
            config.client = None

        config._initialized = True

    @classmethod
    def get_backend(cls) -> Optional[str]:
        return cls.get_instance().backend

    @classmethod
    def get_client(cls) -> Optional[object]:
        return cls.get_instance().client

    @classmethod
    def enable_token2text(cls) -> Optional[bool]:
        return cls.get_instance().token2text

    @classmethod
    def get_scorers(cls) -> List[Any]:
        scorers = cls.get_instance().scorers
        print(f"RolloutTraceConfig.get_scorers() called, returning {len(scorers)} scorers: {scorers}")
        return scorers

    @classmethod
    def add_scorer(cls, scorer: Any):
        """Add a scorer to the configuration."""
        config = cls.get_instance()
        config.scorers.append(scorer)

    @classmethod
    def reset(cls):
        cls._instance = None


@contextlib.contextmanager
def rollout_trace_attr(sample_index=None, step=None, rollout_n=None, name="rollout_trace", validate=False):
    """A context manager to add attributes to a trace for the configured backend."""
    backend = RolloutTraceConfig.get_backend()
    attributes = {}
    if backend:
        if sample_index is not None:
            # Handle different types of sample_index (int, str, numpy.int64, etc.)
            if hasattr(sample_index, 'item'):  # numpy types
                attributes["sample_index"] = sample_index.item()
            elif isinstance(sample_index, str):
                # Try to convert string to int if possible, otherwise keep as string
                try:
                    attributes["sample_index"] = int(sample_index)
                except (ValueError, TypeError):
                    attributes["sample_index"] = sample_index
            else:
                attributes["sample_index"] = sample_index
        if step is not None:
            # Handle different types of step (int, str, numpy.int64, etc.)
            if hasattr(step, 'item'):  # numpy types
                attributes["step"] = step.item()
            elif isinstance(step, str):
                # Try to convert string to int if possible, otherwise keep as string
                try:
                    attributes["step"] = int(step)
                except (ValueError, TypeError):
                    attributes["step"] = step
            else:
                attributes["step"] = step
        if rollout_n is not None:
            # Handle different types of rollout_n (int, str, numpy.int64, etc.)
            if hasattr(rollout_n, 'item'):  # numpy types
                attributes["rollout_n"] = rollout_n.item()
            elif isinstance(rollout_n, str):
                # Try to convert string to int if possible, otherwise keep as string
                try:
                    attributes["rollout_n"] = int(rollout_n)
                except (ValueError, TypeError):
                    attributes["rollout_n"] = rollout_n
            else:
                attributes["rollout_n"] = rollout_n
        attributes["validate"] = validate
        attributes["experiment_name"] = RolloutTraceConfig.get_instance().experiment_name

    if not attributes or backend is None:
        yield
        return

    if backend == "weave":
        import weave

        with weave.attributes(attributes):
            yield
    elif backend == "mlflow":
        import mlflow

        with mlflow.start_span(name=name) as span:
            trace_id = span.trace_id
            for key, value in attributes.items():
                mlflow.set_trace_tag(trace_id, str(key), str(value))
            yield
    else:
        yield


async def apply_scorers_to_call(call, result, scorers, instance=None):
    """Apply scorers to a weave call."""
    if not scorers:
        print("No scorers to apply")
        return
    
    try:
        for scorer in scorers:
            print(f"Applying scorer {scorer.__class__.__name__}")
            
            prompt_text = ""
            response_text = ""
            
            if hasattr(result, 'prompt_ids') and hasattr(result, 'response_ids'):
                try:
                    if instance and hasattr(instance, 'tokenizer') and hasattr(instance.tokenizer, 'decode'):
                        loop = asyncio.get_running_loop()
                        prompt_text = await loop.run_in_executor(None, instance.tokenizer.decode, result.prompt_ids)
                        response_text = await loop.run_in_executor(None, instance.tokenizer.decode, result.response_ids)
                    else:
                        prompt_text = "prompt_text"
                        response_text = "response_text"
                except Exception as decode_error:
                    print(f"Error decoding tokens: {decode_error}")
                    prompt_text = "prompt_text" 
                    response_text = "response_text"  
            

            try:
                score_result = await scorer.score(response_text, prompt_text)
                print(f"Scorer {scorer.__class__.__name__} result: {score_result}")
                
                if hasattr(call, 'add_attribute'):
                    call.add_attribute(f"scorer_{scorer.name}", score_result)
                
            except Exception as scorer_error:
                print(f"Error calling scorer {scorer.__class__.__name__}: {scorer_error}")
                
    except Exception as e:
        # Log error but don't fail the main operation
        print(f"Warning: Failed to apply scorer {scorer.__class__.__name__}: {e}")


def rollout_trace_op(func):
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        backend = RolloutTraceConfig.get_backend()
        enable_token2text = RolloutTraceConfig.enable_token2text()
        scorers = RolloutTraceConfig.get_scorers()
        print(f"use scorers: {scorers}")
        if backend is None:
            return await func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        async def add_token2text(self, result):
            if hasattr(result, "prompt_ids") and hasattr(self, "tokenizer") and hasattr(self.tokenizer, "decode"):
                _result = vars(result)
                loop = asyncio.get_running_loop()
                if hasattr(result, "prompt_ids"):
                    prompt_text = await loop.run_in_executor(None, self.tokenizer.decode, result.prompt_ids)
                    _result["prompt_text"] = prompt_text

                if hasattr(result, "response_ids"):
                    response_text = await loop.run_in_executor(None, self.tokenizer.decode, result.response_ids)
                    _result["response_text"] = response_text
                return _result
            return result

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context

            cur_attributes = {**call_context.call_attributes.get()}
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await add_token2text(self, result)
                    tracer.finish_call(call, output=_result)
                else:
                    tracer.finish_call(call, output=result)

                if scorers:
                    print(f"Applying scorers: {scorers}")
                    await apply_scorers_to_call(call, result, scorers, self)

                return result

            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            with mlflow.start_span(name=func.__qualname__) as span:
                span.set_inputs(inputs)
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await add_token2text(self, result)
                    span.set_outputs(_result)
                else:
                    span.set_outputs(result)

            return result

        else:
            return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        backend = RolloutTraceConfig.get_backend()
        scorers = RolloutTraceConfig.get_scorers()
        
        if backend is None:
            return func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context

            cur_attributes = {**call_context.call_attributes.get()}
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = func(self, *args, **kwargs)
                tracer.finish_call(call, output=result)
                
                # Apply scorers after finishing the call
                asyncio.create_task(apply_scorers_to_call(call, result, scorers))
                
                return result
            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            return mlflow.trace(func)(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
