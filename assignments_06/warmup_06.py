
from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---
# Concepts Question 1


# Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.
# Best approach: fine-tuning
# Reason: Fine-tuning is a process designed to help a pre-trained LLM excel at specific tasks. It facilitates LLM grounding by exposing the model to additional data relevant to the application in question.


# Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online. They have 3,000 examples their in-house writers produced over the years.
# Best approach: prompt engineering
# Reason: Prompt engineering is a technique focused on optimizing the input provided to your large language model. By providing clear instructions, context, and desired formats, prompt engineering can mitigate some of the inherent shortcomings of basic prompts.


# Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does not need this to work for any other document.
# Best approach: RAG
# Reason: RAG intercepts a prompt, identifies relevant information from internal and external sources, and then augments the prompt with the additional information – leading to a better, more relevant response.



# Concepts Question 2

# Q: Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real situation where a confident hallucination could cause harm.
# A: A confidently wrong answer would more likely make user believe the output from model than "I am not sure". For example, if user asks a souce information that model doesn't know. Model then generates response that's from a similar source.
#   And it would cause financial loss if the user provide that response to their customer.

# Q: Think about the tone of the response as well as its content — why does the way the model expresses an answer affect how much we trust it?
# A: low-confidence answers with first-person uncertainty (e.g., "I'm not completely sure, but...") is significantly more effective at calibrating user trust than passive phrasing.
#    Also if the model provides clickable links, source cards, or reference lists—alongside indicators of source trustworthiness and how many sources back a claim—helps users verify facts.



# Concepts Question 3

"""
steps = [
    "Receive the user's query", The system captures the natural language question or prompt inputted by the user.
    "Extract text from source documents", Raw data and content are gathered from external files, databases, or web sources.
    "Split text into chunks", Large documents are broken down into smaller, manageable text segments for precise searching.
    "Convert text chunks into embeddings", Document segments are transformed into dense vector representations using an embedding model.
    "Embed the user's query", The user's input question is converted into the same vector space format as the document chunks.
    "Retrieve the most relevant chunks",Vector search compares the query embedding against the document embeddings to find the closest matches.
    "Inject retrieved chunks into the prompt", The context found in the top chunks is combined with the user's original query into a structured prompt.
    "Generate a response from the LLM", The large language model uses the enriched prompt to generate a grounded, context-aware answer.
]
"""




# --------Keyword RAG-------
# Keyword Question 1

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)
# Print the name of the selected document.
print(result)
# After running the function, add a comment explaining which document was selected and why.
# Selected best match: loyalty.txt
# [('loyalty.txt', 'Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.')]
# the documents are being ranked based on the number of exact keyword matches with the query. The query has one match with "hours.txt","hiring.txt" and "loyalty.txt" three documents.
# Since the scores for relevance are all the same. It randomly chose  'loyalty.txt' as the match document.


# Keyword Question 2
query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)
# Print the name of the selected document.
print(result)
# Which document was selected
# None of the documents was selected because the function didn't find the matching keyword with the query.

# Whether keyword RAG got this right — and why or why not
# The keyword RAG didn't get it right.It should've match the "menu.txt" document. It didn't find any exact matching keywords from the documents.

# What kind of retrieval would do better here
# Semantic (Vector) Retrieval: Uses embeddings to capture the meaning of words rather than their exact letters. It maps queries and documents into a vector space, allowing the system to find relevant chunks even if the vocabulary between the user and the document is completely different.
# Hybrid Retrieval (The Gold Standard): Combines both keyword search (BM25) and semantic vector search, blending their scores together (often using a technique called Reciprocal Rank Fusion).


# Keyword Question 3
query = "How do I sign up for rewards?"
# Predict: there will be no selected document for this query because there're no matching keywords in all the documents.
result = simple_keyword_retrieval(query, documents, verbose=True)
# Print the name of the selected document.
print(result)
# my prediction was right.



# ----------Semantic RAG Concepts---------
# Semantic Question 1

# 1. What is a vector embedding? (1-2 sentences)
#    A vector embedding is a numerical representation of data—such as a word, sentence, or image—expressed as a list of numbers in a high-dimensional space. This format allows computers to measure the semantic similarity and relationships between different pieces of data based on their positions in that space.
# 2. Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, and what does that number tell you about the relationship between the texts?
#.   The scores of 0.85 is more relevant. the higher the cosine similarity, the more relevant the text chunks are.
# 3. Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
#.   Semantic search can find relevant chunks without exact word matches because it translates words and sentences into meaning-based numerical representations (embeddings) rather than matching letters.


# Semantic Question 2

# | Feature                    | Keyword RAG                       | Semantic RAG |
#|----------------------------|-----------------------------------|--------------|
#| What is compared?          | Exact word overlap                | vector embeddings            |
#| What is retrieved?         | Full document                     | relevant text chunks          |
#| Can it handle synonyms?    | No                                | Yes            |
#| Storage format             | Plain text dictionary             | embeddings            |
#| Relevance score            | Number of overlapping keywords    | cosine similarity         |



# ---------LlamaIndex-----------

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# LlamaIndex Question 1

# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader("./lessons/06_AI_augmentation/resources/brightleaf_pdfs").load_data()


llm = OpenAI(temperature=0.1, model="gpt-5.6-terra")
# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3, llm=llm)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.get_content()[:150]}...")
        print("-" * 30)

# For question 1, only the first retrieved chunk look relevant to the question. the model's response sound confident and specific. the secong and third chunks of text look unrelevantto the question.
# For question 2, only the first retrieved chunk look relevant to the question. the model's response sound confident and specific. the secong and third chunks of text look unrelevantto the question.


# LlamaIndex Question 2
print('LlamaIndex Question 2')

# query with similarity_top_k=1
query_engine = index.as_query_engine(similarity_top_k=1, llm=llm)
questions = [
    "What employee benefits does BrightLeaf offer?",
]
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.get_content()[:150]}...")
        print("-" * 30)

# query with similarity_top_k=5
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)
questions = [
    "What employee benefits does BrightLeaf offer?",
]
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.get_content()[:150]}...")
        print("-" * 30)
# the response with similarity_top_k=1 is more detailed and specific than  with similarity_top_k=3 or  with similarity_top_k=1. It shows that in this case more retrieved context doesn't provide better result.


# LlamaIndex Question 3

query_engine = index.as_query_engine(similarity_top_k=3, llm=llm)
questions = [
    "Do brightleaf solar's leaders all get married?",
]
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.get_content()[:150]}...")
        print("-" * 30)

# I was expecting no available answers from the model. And it did say there's no information regarding my question. I think the model is working well.



# LlamaIndex Question 4
print("LlamaIndex " \
"Question 4")

# Create Judge LLM
llm2 = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm2)
relevancy_evaluator = RelevancyEvaluator(llm=llm2)

q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))



#  run the evaluators again on a query you expect to produce a lower-quality response 
q = "What schools do they recommend for toddlers?"
response = query_engine.query(q)

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))


# 1. a faithfulness score of 1.0 mean  the response is faithful to the retrieved contexts. a score of 0.0 indicate the response contains hallucinations or lying.
# 2. a relevancy score measure whether the response is relevant to the query using the retrieved contexts. the relevancy score measure the relevancy between query and the retrieved contexts, while 
# faithfulness score measure how true the response is.
# 3. the scores changed from 1.0 to 0.0. because my second question was not relevant to the docs.
# 4.  the "LLM-as-a-judge" approach meaning the evaluation metrics are computed using an external LLM (different from the RAG framework). 
#  Because for RAG frameworks we're trying to assess accuracy in the natural language space as opposed to the numerical or categorical space for ML models.
#  An external LLM (trained appropriately) will be able to assess these subjective metrics faster compared to multiple human experts. 
