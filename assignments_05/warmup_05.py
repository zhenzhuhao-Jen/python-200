import os
from dotenv import load_dotenv
from openai import OpenAI
import json

#------------The Chat Completions API-------------

#------API Question 1-----




if load_dotenv():
    print("Successfully loaded api key")
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)
#Print just the text of the response (not the whole object). Then print the name of the model that responded and the total number of tokens used. Label each output.
print("response: ", response.choices[0].message.content)
print("Name of the Model: ", response.model)
print("Total Tokens: ", response.usage.total_tokens)


#------API Question 2-----
# Run the same prompt three times with three different temperature settings: 0, 0.7, and 1.5. Print each response, labeled with its temperature.
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures: 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature = temperature
    )
    print(f"{temperature}: {response.choices[0].message.content}")
# When the temperature is 0 and 0.7, it has the same response: "DataForge Solutions". When the temperature is 1.5, it provided the answer and explanations, which makes is sound more like a human.
# I would use temperature 1.5. It's more persuasive.


#------API Question 3-----
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
for choice in response.choices:
  print(choice)


#------API Question 4-----
# Set max_tokens=15 and send a prompt that would normally produce a long response (for example, "Explain how neural networks work."). 
# Print the result. 
# Add a comment: What happened, and why might you want to use max_tokens in a real application?
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)
print(response.choices[0].message.content)
# The result wasn't complete. max_tokens limits the lenght of the response. This can be a useful way to keep costs under control.




#-----------------System Messages and Personas--------------------

#--------System Question 1-----------
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
)

print(response.choices[0].message.content)

print("The grumpy tutor:")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a impatient Python tutor. You only want to  explain things as simple as it can be."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
)
print(response.choices[0].message.content)
# After changing the system personality from patient to impatient. the response also changed correspondingly with a less detailed and short answer.


#--------System Question 2-----------
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
)
print(response.choices[0].message.content)
# The model reads the previous user and assistant turns from the `messages` array 
# and processes them as part of its current input context.


#-----------------Prompt Engineering--------------------
print("Prompt Engineering")
#------Prompt Question 1 — Zero-Shot-------
def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]


prompt = f"""
classify the sentiment of each review below as positive, negative, or mixed.
Print each result labeled with the review number.

Review:
'''
{reviews}
'''

"""

result= get_completion(prompt, temperature=0)
print(result)


#------Prompt Question 2 — one-Shot-------
prompt = f"""
classify the sentiment of each review below as positive, negative, or mixed.
Print each result labeled with the review number.

Review:
'''
{reviews}
'''

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

"""

result= get_completion(prompt, temperature=0)
print(result)
# By adding the example, the format of the output changed compared to Q1


#------Prompt Question 3 — few-Shot-------

prompt = f"""
classify the sentiment of each review below as positive, negative, or mixed.
Print each result labeled with the review number.

Review:
'''
{reviews}
'''

Example1:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Example2:
Review: "I'm very satisfied with my purchase."
Sentiment: positve

Example3:
Review: "The product broke in a week, won't buy again."
Sentiment: negative
"""

result= get_completion(prompt, temperature=0)
print(result)
# Q2 has better output format than Q1 cause it listed the review with corresponding sentiment. 
# Q3 has the exact same output with Q2.
# I would choose Zero-shot if I only want a answer from the model's pre-trained knowledge.
# I would choose One-shot if I want to clarify the expected format and improves accuracy.
# I would choose Few-shot if I want the model recognize the pattern and handle the task more reliably.


#------Prompt Question 4 — Chain of Thought-------

prompt = """
Show your step-by-step reasoning, then give the final answer on its own line labelled: Final annual salary: <value>

Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

result = get_completion(prompt, temperature=0)
print(result)
# You can inspect the steps for errors.
# Labeling a final answer (e.g., "Final answer: $25") makes it easy to parse in code.
# Shows the model's logic, building trust in the result.


#------Prompt Question 5 — Structured Output-------
prompt = """
Classify the sentiment of the review and respond ONLY with valid JSON. don't include any words outside the curly brackets.
Keys: sentiment (positive/negative/mixed), confidence (0–1 scale), reason (one short sentence).

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
"""

response = get_completion(prompt, temperature=0)
print("Raw response:", response)

# Parse JSON safely
try:
    result = json.loads(response)
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")


 #------Prompt Question 6 — Delimiters-------

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
result = get_completion(prompt, temperature=0)
print(result)


user_text = "Today we went to the market and got three little pigs. One for Mom, one for Dad and one for me."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
result = get_completion(prompt, temperature=0)
print(result)

# Delimiters are simple markers (like triple backticks or dashes) that clearly separate these sections so the model doesn't get confused.
# Delimiters help prevent prompt injection and keep results predictable.


#-----------Local Models with Ollama-------------
#------Ollama Question 1--------

prompt = "Explain what a large language model is in two sentences."
result = get_completion(prompt, temperature=0)
print(result)
# A large language model is a type of AI model capable of understanding and generating human language, trained on vast 
# datasets of text to recognize patterns and produce coherent responses. It processes and analyzes large volumes of 
# information to perform tasks like answering questions or writing articles, leveraging complex algorithms to achieve high accuracy.

# differences: the OpenAI had a faster response than Ollama.
# Advantage: Privacy and Cost. Your data never leaves your computer (making it secure for sensitive or private information), 
# and once downloaded, you can run it completely offline without paying per-token API fees.

# Disadvantage: Hardware Requirements. Running large language models locally demands significant computational resources (a powerful GPU, high VRAM, and RAM), 
# whereas cloud APIs handle all the heavy lifting on remote servers.