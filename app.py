from dotenv import load_dotenv
load_dotenv()

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

# Create an RLM module
rlm = dspy.RLM("context, query -> answer")

# Call it like any other module
result = rlm(
    context="...very long document or data...",
    query="What is the total revenue mentioned?"
)
print(result.answer)