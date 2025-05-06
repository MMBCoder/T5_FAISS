import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import faiss
import numpy as np
import PyPDF2

# Initialize models with caching
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='./model_cache')
        tokenizer = T5Tokenizer.from_pretrained('t5-base', cache_dir='./model_cache')
        t5_model = T5ForConditionalGeneration.from_pretrained('t5-base', cache_dir='./model_cache')
        return embed_model, tokenizer, t5_model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

embed_model, tokenizer, t5_model = load_models()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Build FAISS index
def build_faiss_index(text_chunks):
    embeddings = embed_model.encode(text_chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Retrieve relevant context
def retrieve_context(query, index, text_chunks, k=3):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k)
    retrieved = [text_chunks[idx] for idx in indices[0]]
    return retrieved

# Generate response using T5
def generate_response(query, context_docs):
    input_text = f"question: {query} context: {' '.join(context_docs)}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = t5_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit App
st.title("📄 RAG-based GenAI App with FAISS & T5")

if embed_model is None:
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Split text into chunks
    text_chunks = [chunk.strip().replace('\n', ' ') for chunk in pdf_text.split('.') if chunk.strip()]

    # Create FAISS index
    index, embeddings = build_faiss_index(text_chunks)

    st.success("PDF processed successfully!")

    user_query = st.text_input("Enter your question related to the PDF:")

    if user_query:
        with st.spinner("Generating response..."):
            context = retrieve_context(user_query, index, text_chunks, k=3)
            response = generate_response(user_query, context)

        st.subheader("Retrieved Context:")
        for i, ctx in enumerate(context, 1):
            st.write(f"{i}. {ctx}")

        st.subheader("Generated Response:")
        st.write(response)
else:
    st.info("Please upload a PDF to start.")
