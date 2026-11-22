from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


docs_dir = Path("./lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Load the Documents. How many documents were loaded. The file name of each document
docs = SimpleDirectoryReader(docs_dir).load_data()
print(len(docs))
for doc in docs:
    print(doc.metadata["file_name"])


# Step 3: Build the Index and Query Engine
# Specify the new model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

#build a VectorStoreIndex from the loaded documents and create a query engine with similarity_top_k=3. 
# Print a short confirmation message once the index is ready
llm = OpenAI(temperature=0.1, model="gpt-5.6-terra")

index = VectorStoreIndex.from_documents(docs)
if index:
    print("Index built successfully. Ready to answer questions.")

query_engine = index.as_query_engine(similarity_top_k=3, llm=llm)



# Step 4: Query the Assistant
print("Step 4: Query the Assistant")

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
     print(f"Document Name: {node_with_score.node.metadata["file_name"]}")
     print(f"Similarity Score: {node_with_score.score:.4f}")
     print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
     print("-" * 30)


# the assistant sound confident and accurate and there's no surprises.



# Step 5: Find a Failure
questions = [
    "What kind of beans are used for cold brew coffee?"
]
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
     print(f"Document Name: {node_with_score.node.metadata["file_name"]}")
     print(f"Similarity Score: {node_with_score.score:.4f}")
     print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
     print("-" * 30)
# What you asked and why you expected it to be hard
# I asked : "What kind of beans are used for cold brew coffee?". I expected it to be hard because the documents don't contain that informaton.

# What went wrong — wrong retrieval, missing information, the model guessed anyway?
# Nothing went wrong, it told me there's no related information.

# When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? What does this suggest about trusting AI-generated responses?
# For my question, the assistant has a certain tone about the response.

# What you would change about the system to improve it
# I would probably offer solution to help the user solve the problem by providing source information or contact information.



# Step 6: Reflection
# 1. The LlamaIndex lesson lists the steps that LlamaIndex automates: chunking, embedding, and indexing.
#  How many lines of code did your LlamaIndex implementation take in your project?
#  Compare that with the keyword RAG code from the Keyword-based RAG lesson,
#  or with the code you would need to write each of those steps yourself. What does that tell you about the value of using a framework?

# A: In my project, I used three lines of code for my LlamaIndex implementation: specify the embedding model; define llm and build VectorStoreIndex.



# 2. You have now built a system that answers questions from real documents. 
# Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.

# A: For company, It could read long internal documents and save people time to find the information they need in seconds.



# 3. What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?

# A: Even when retrieval works perfectly and surfaces the exact right source chunks, generation-phase hallucination (unfaithful synthesis) remains a failure mode that RAG cannot fully prevent.