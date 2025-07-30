# Code Agentic Training

A multi-turn code debugging training system based on the retool implementation, designed to train models that can write code, run test cases, and iteratively debug based on feedback.

## Overview

This system extends the original retool implementation to focus specifically on code debugging scenarios where:

1. **Multi-turn conversations**: Models engage in back-and-forth debugging sessions
2. **Test case validation**: Code is automatically tested against predefined test cases
3. **Intelligent debugging**: Models receive specific suggestions based on error types
4. **Iterative improvement**: Models learn to fix bugs based on feedback

## Key Features

### Enhanced Code Execution Tool (`CodeAgenticTool`)
- **Test case validation**: Automatically validates code against predefined test cases
- **Debug suggestions**: Provides specific debugging hints based on error types
- **Multi-turn support**: Tracks conversation history for context
- **Syntax error fixing**: Automatically fixes common syntax issues
- **Debug prints**: Adds automatic debug output for variable tracking

### Test Case System
- **Predefined test cases**: Factorial, Fibonacci, prime checking, array operations, string manipulation
- **Input/output validation**: Compares actual vs expected results
- **Error categorization**: Classifies errors (syntax, logic, runtime, etc.)
- **Timeout handling**: Prevents infinite loops and long-running code

### Training Data Generation
- **Realistic bug patterns**: Generates common programming mistakes
- **Debug conversations**: Creates multi-turn debugging dialogues
- **Progressive difficulty**: Easy to hard problems across different categories
- **Diverse error types**: Syntax errors, logic errors, edge cases

## Architecture

```
User Problem → Model generates code → Tool executes with test case
     ↓
Tool validates result → Provides feedback/suggestions
     ↓
Model fixes code → Tool re-executes → Success or continue debugging
```

## Usage

### 1. Data Preparation

Generate training data with debug conversations:

```bash
python3 recipe/retool/code_agentic_preprocess.py
```

This creates a dataset with:
- 6 problem categories (mathematics, arrays, strings, algorithms)
- 3 debug conversation patterns per problem
- Multi-turn conversations with tool calls and responses

### 2. SFT Training

Train the base model on code debugging conversations:

```bash
bash recipe/retool/run_code_agentic_sft.sh
```

Training parameters:
- Model: Qwen2.5-32B-Instruct
- Batch size: 1 (gradient accumulation: 8)
- Learning rate: 5e-6
- Epochs: 3
- Max length: 4096

### 3. PPO Training

Fine-tune with reinforcement learning:

```bash
bash recipe/retool/run_code_agentic_ppo.sh
```

PPO parameters:
- Learning rate: 1e-6
- Max steps: 1000
- Reward scale: 1.0
- Debug penalty: 0.1
- Tool bonus: 0.05

## Test Cases

The system includes predefined test cases:

### Mathematics
- **Factorial**: Calculate n! with edge cases (0, 1, 5)
- **Fibonacci**: Calculate nth Fibonacci number
- **Prime Check**: Determine if number is prime

### Arrays
- **Array Sum**: Sum all elements in array
- **Bubble Sort**: Sort array in ascending order

### Strings
- **String Reverse**: Reverse a string

## Debug Suggestions

The system provides targeted debugging suggestions:

- **NameError**: Variable not defined
- **SyntaxError**: Check parentheses, brackets, indentation
- **TypeError**: Data type mismatches
- **IndexError**: Array bounds issues
- **ZeroDivisionError**: Division by zero
- **AttributeError**: Object method issues

## Configuration

### Tool Configuration (`code_agentic_tool_config.yaml`)
```yaml
tools:
  - name: code_agentic_tool
    tool_schema:
      name: code_agentic_tool
      description: Execute Python code with test case validation
      parameters:
        code: Python code to execute
        test_case: Test case name
        timeout: Execution timeout
        language: Programming language
    config:
      default_timeout: 30
      max_debug_attempts: 5
      enable_debug_prints: true
```

### Training Configuration
- **SFT**: Focus on learning code generation and debugging patterns
- **PPO**: Optimize for efficient debugging with minimal attempts
- **Reward function**: Balances correctness, tool usage, and debug efficiency

## Expected Results

After training, the model should:

1. **Generate correct code** for given problems
2. **Use tools effectively** for code execution and testing
3. **Interpret error messages** and apply debugging suggestions
4. **Fix bugs iteratively** based on feedback
5. **Minimize debug attempts** while ensuring correctness

## File Structure

```
recipe/retool/
├── code_agentic_training.py      # Main training implementation
├── code_agentic_preprocess.py    # Data generation script
├── code_agentic_tool_config.yaml # Tool configuration
├── run_code_agentic_sft.sh       # SFT training script
├── run_code_agentic_ppo.sh       # PPO training script
└── CODE_AGENTIC_README.md        # This documentation
```

## Differences from Original Retool

1. **Focus**: Code debugging vs general tool use
2. **Test cases**: Automatic validation vs manual verification
3. **Debug suggestions**: Specific error-based hints vs general feedback
4. **Multi-turn**: Iterative debugging vs single-shot execution
5. **Reward function**: Debugging efficiency vs general correctness

## Future Enhancements

1. **More test cases**: Expand problem categories and difficulty levels
2. **Dynamic test generation**: Generate test cases based on problem description
3. **Code quality metrics**: Consider code readability and efficiency
4. **Multi-language support**: Extend beyond Python
5. **Real-world scenarios**: Include actual programming problems from competitions 