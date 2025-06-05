import streamlit as st
from backend import load_and_split, build_vectorstore, get_qa_chain

st.set_page_config(page_title="Local Document Q&A", layout="wide")
st.title("Q&A with your PDF")

st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose LLM", ["OpenAI", "Ollama"])
embedding_type = "openai" if model_choice == "OpenAI" else "huggingface"

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
qa_chain = None

if uploaded_file:
    with st.spinner("Processing document..."):
        filepath = f"data/{uploaded_file.name}"
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        chunks = load_and_split(filepath)
        vectordb = build_vectorstore(chunks, embedding_type=embedding_type)
        qa_chain = get_qa_chain(vectordb, model_choice)
    st.success("Document ready for Q&A!")

if qa_chain:
    query = st.text_input("Ask a question about your document:")
    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
        st.markdown(f"**Answer:** {answer}")