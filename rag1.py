#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

@st.cache_resource
def load_models():
    # Load generator model and tokenizer
    generator_model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    generator = AutoModelForCausalLM.from_pretrained(generator_model_name)
    
    # Load embedder
    embedder = SentenceTransformer('all-mpnet-base-v2')
    
    # Create FAISS index
    documents = [
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris.",
        "The Louvre Museum is a famous art museum in Paris.",
        "The French national symbol is the Gallic Rooster."
    ]
    
    d = 768
    index = faiss.IndexFlatL2(d) 
    doc_embeddings = embedder.encode(documents)
    index.add(doc_embeddings)
    
    return tokenizer, generator, embedder, index, documents

tokenizer, generator, embedder, index, documents = load_models()

def retrieve_relevant_documents(query, k=2):
    query_embedding = embedder.encode([query])[0]
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    return [documents[i] for i in indices[0]]

def generate_response(query, retrieved_docs):
    augmented_input = f"Question: {query}\nRelevant Information: {'; '.join(retrieved_docs)}"
    
    inputs = tokenizer(augmented_input, return_tensors="pt", truncation=True, padding=True)
    
    outputs = generator.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_response = full_response.split("Relevant Information:")[-1].strip()
    return generated_response

# Streamlit UI
st.title("🗨️ Document-Augmented Chatbot")
st.write("This chatbot uses retrieved documents to enhance its responses!")

query = st.text_input("Enter your question:", 
                     placeholder="Ask about France...")

if query:
    with st.spinner("Searching documents and generating answer..."):
        relevant_docs = retrieve_relevant_documents(query)
        response = generate_response(query, relevant_docs)
    
    st.subheader("Answer:")
    st.write(response)
    
    with st.expander("See retrieved documents"):
        for i, doc in enumerate(relevant_docs, 1):
            st.write(f"{i}. {doc}")

