import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

def extract_text(content):
    """Extract plain text from a Gemini response content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)

# Initialize environment
load_dotenv()

# --- Config ---
CHROMA_PATH = "chroma_db/"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3-flash-preview"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# --- Minimal UI Setup ---
st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="centered")

# Hide Streamlit default styling for an extra minimal look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("📄 PDF Chat")
st.markdown("Upload a PDF to start chatting.")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# --- Sidebar for Upload ---
with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file and not st.session_state.db_ready:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Load and Split
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Embed and Store
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
            # We recreate the DB in memory or a temp folder for the web app to keep it simple and clean
            # For a production app, you might want persistent user-specific stores
            st.session_state.db = Chroma.from_documents(chunks, embeddings)
            st.session_state.db_ready = True
            
            # Cleanup temp file
            os.remove(tmp_path)
            
        st.success(f"Loaded {len(documents)} pages!")
    
    if st.session_state.db_ready:
        if st.button("Reset Document"):
            st.session_state.db_ready = False
            st.session_state.messages = []
            if "db" in st.session_state:
                del st.session_state.db
            st.rerun()

# --- Chat Interface ---
if st.session_state.db_ready:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to UI and history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve context
                results = st.session_state.db.similarity_search_with_relevance_scores(prompt, k=3)
                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                
                # Query LLM
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                llm_prompt = prompt_template.format(context=context_text, question=prompt)
                
                model = ChatGoogleGenerativeAI(model=CHAT_MODEL)
                response = model.invoke(llm_prompt)
                
                answer_text = extract_text(response.content)
                st.markdown(answer_text)
                
                # Optionally add sources expandable
                if results:
                     with st.expander("View Sources"):
                         for i, (doc, score) in enumerate(results):
                             st.markdown(f"**Chunk {i+1} (Score: {score:.2f})**\n\n{doc.page_content[:200]}...")

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer_text})

else:
    st.info("👈 Please upload a PDF in the sidebar to begin.")
