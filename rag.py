# importing libraries
import numpy as np           
from openai import OpenAI


# 1. Initialize OpenAI 

client = OpenAI(api_key="your_api_key_here")  # Replace with your actual API key

# 2. Open your file here
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# creating big data into small chunks
chunk_size = 300
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# embdding chunks in openai database
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


# 5. Create embeddings for all chunks

chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

# user will ask question here
question = input("Ask a question: ")
question_embedding = get_embedding(question)

# searching for the most relevant reply
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine_similarity(question_embedding, emb) for emb in chunk_embeddings]
best_index = int(np.argmax(scores))

context = chunks[best_index]

# generating answer using most relevant data/chunk
prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print("\nAnswer:")
print(response.choices[0].message.content)


# copyrights (c) to Ujjwal Prakash.