# Enhanced Code Agentic Training

A multi-turn code debugging training system that adapts to your data format, using sandbox fusion tools to run input/output test cases and provide iterative debugging feedback.

## Overview

This enhanced system is specifically designed to work with your training data format and provides:

1. **Input/Output Test Case Validation**: Automatically validates code against your provided test cases
2. **Multi-turn Debugging**: Models engage in back-and-forth debugging sessions
3. **Sandbox Fusion Integration**: Uses your existing sandbox fusion tools for code execution
4. **Retool-style Prompts**: Maintains the same prompt format as your retool implementation
5. **Iterative Improvement**: Models learn to fix bugs based on specific feedback

## Key Features

### Enhanced Code Execution Tool (`EnhancedCodeAgenticTool`)
- **Input/Output Testing**: Validates code against your specific test cases
- **Test Case Indexing**: Supports multiple test cases per problem
- **Debug Suggestions**: Provides targeted debugging hints
- **Sandbox Integration**: Uses your existing sandbox fusion infrastructure
- **Error Categorization**: Classifies different types of errors

### Data Format Support
- **Your Data Format**: Processes your specific JSON/parquet format
- **Question/Input/Output**: Handles your problem specification format
- **Multiple Test Cases**: Supports multiple input/output pairs per problem
- **Conversation Generation**: Creates realistic debugging dialogues

### Training Pipeline
- **SFT Stage**: Learn code generation and debugging patterns
- **PPO Stage**: Optimize for efficient debugging with minimal attempts
- **Reward Function**: Balances correctness, tool usage, and debug efficiency

## Your Data Format

The system is designed to work with your data format:

```json
{
  "prompt": "You are an expert Python programmer...",
  "question": "Problem description...",
  "query_id": "unique_id",
  "input_output": {
    "inputs": ["input1", "input2", ...],
    "outputs": ["output1", "output2", ...],
    "remote": false
  }
}
```

## Architecture

```
User Question → Model generates code → Tool executes with test case
     ↓
Tool validates I/O → Provides feedback/suggestions
     ↓
Model fixes code → Tool re-executes → Success or continue debugging
```

## Usage

### 1. Data Preparation

Generate training data from your format:

```bash
python3 recipe/code_agentic/enhanced_code_agentic_preprocess.py
```

This script:
- Loads your data format (JSON/parquet)
- Parses question and input/output test cases
- Generates realistic debug conversations
- Creates multi-turn training examples

### 2. SFT Training

Train the base model on code debugging conversations:

```bash
bash recipe/code_agentic/run_enhanced_code_agentic_sft.sh
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
bash recipe/code_agentic/run_enhanced_code_agentic_ppo.sh
```

PPO parameters:
- Learning rate: 1e-6
- Max steps: 1000
- Reward scale: 1.0
- Debug penalty: 0.1
- Tool bonus: 0.05
- Test case bonus: 0.3

## Configuration

### Tool Configuration (`enhanced_code_agentic_tool_config.yaml`)
```yaml
tools:
  - name: enhanced_code_agentic_tool
    tool_schema:
      name: enhanced_code_agentic_tool
      description: Execute Python code with input/output test case validation
      parameters:
        code: Python code to execute
        test_case_index: Index of test case to validate
        timeout: Execution timeout
        language: Programming language
    config:
      default_timeout: 30
      max_debug_attempts: 5
      enable_debug_prints: true
```

### Dataset Processing
The system automatically:
1. **Parses your data format** into structured problems
2. **Extracts test cases** from input_output field
3. **Generates debug conversations** with realistic bugs
4. **Creates multi-turn dialogues** with tool calls and responses

## Debug Suggestions

The system provides targeted debugging suggestions:

- **NameError**: Variable not defined
- **SyntaxError**: Check parentheses, brackets, indentation
- **TypeError**: Data type mismatches
- **IndexError**: Array bounds issues
- **ZeroDivisionError**: Division by zero
- **AttributeError**: Object method issues
- **Test failed**: Output doesn't match expected result

## Reward Function

The enhanced reward function considers:

1. **Base Score**: Correctness of final solution
2. **Tool Bonus**: Encourages appropriate tool usage
3. **Test Case Bonus**: Rewards passing multiple test cases
4. **Debug Penalty**: Penalizes excessive debugging attempts

## File Structure

```
recipe/code_agentic/
├── enhanced_code_agentic.py              # Main enhanced implementation
├── enhanced_code_agentic_preprocess.py   # Data generation script
├── enhanced_code_agentic_tool_config.yaml # Tool configuration
├── run_enhanced_code_agentic_sft.sh      # SFT training script
├── run_enhanced_code_agentic_ppo.sh      # PPO training script
└── ENHANCED_CODE_AGENTIC_README.md       # This documentation
```

## Differences from Original Retool

1. **Data Format**: Adapts to your specific input/output format
2. **Test Case Validation**: Uses your test cases instead of predefined ones
3. **Sandbox Integration**: Leverages your existing sandbox fusion tools
4. **Multi-test Support**: Handles multiple test cases per problem
5. **Enhanced Rewards**: Considers test case passing rates

## Expected Results

After training, the model should:

1. **Generate correct code** for your problem specifications
2. **Use sandbox fusion tools** effectively for code execution
3. **Interpret error messages** and apply debugging suggestions
4. **Fix bugs iteratively** based on test case feedback
5. **Minimize debug attempts** while ensuring correctness
6. **Handle multiple test cases** efficiently

## Integration with Your System

The enhanced system is designed to:

1. **Use your existing sandbox fusion tools** without modification
2. **Process your data format** directly
3. **Maintain retool-style prompts** for consistency
4. **Support your training pipeline** with minimal changes
5. **Provide debugging feedback** specific to your test cases

## Future Enhancements

1. **Dynamic test generation**: Generate additional test cases
2. **Code quality metrics**: Consider code readability and efficiency
3. **Multi-language support**: Extend beyond Python
4. **Real-time feedback**: Provide immediate debugging suggestions
5. **Performance optimization**: Improve training efficiency 