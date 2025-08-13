def get_code_problem_prompt(question: str) -> str:
    """
    Generate prompt template for code problems
    
    Args:
        question: Code problem description
        
    Returns:
        Formatted prompt content
    """
    prompt_content = """You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

		### Code problem:
		{question}
        
        You can use code_interpreter tool to test your code by generating the following format:
		<tool_call>
		{{"name": "code_interpreter", "arguments": {{"code": "your DEBUG code here"}}}}
		</tool_call>

		""".format(question=question)
    return prompt_content


def get_answer_format() -> str:
    """
    Get answer format requirements
    
    Returns:
        Answer format description
    """
    return """\nThe final code solution format must be a COMPLETE python function start with def and a return value inside <code>your FINAL code here</code>, for example:
    <code>
    def solution(input_data):
        return "your return value here"
    </code>
    You don't need to add any other code before or after the function, I will help you to add the input data and return value."""
