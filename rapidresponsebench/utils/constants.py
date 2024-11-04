import os

INTERNAL_REFUSAL = "<MockLLM> I'm sorry, but I can't respond to that request. </MockLLM>"
INTERNAL_ACCEPT = "<MockLLM> I am going to fulfil your request. </MockLLM>"

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']