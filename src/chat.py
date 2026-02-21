import os
import argparse
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument("query_text", type=str, help="The question you want to ask.")
    args = parser.parse_args()
    query_text = args.query_text

    print(f"Loading ChromaDB from {CHROMA_PATH}/...")
    # 1. Prepare the DB
    # Must use the same embedding model used during ingestion
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 2. Search the DB for relevant chunks
    print(f"Searching for relevance: '{query_text}'")
    # You can adjust 'k' for number of documents retrieved
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0:
        print("Unable to find any context in the database.")
        print("Please ensure you have ingested PDFs using 'python src/ingest.py'")
        return

    # Let the user know the relevance score of the top result
    top_score = results[0][1]
    if top_score < 0.2:
         print(f"Warning: Low relevance score ({top_score:.2f}). The answer may not be accurate.")

    # Combine the context documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # 3. Predict answer using LLM
    print("Generating response with Gemini...")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Initialize the LLM (gemini-1.5-flash is fast and cheap, gemini-1.5-pro is more capable)
    model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview") 
    
    response_text = model.invoke(prompt)

    # Extract plain text from response
    content = response_text.content
    if isinstance(content, list):
        answer = "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    else:
        answer = str(content)

    # Display the sources where the answer came from
    sources = set([doc.metadata.get("source", "Unknown") for doc, _score in results])
    
    print("\n" + "="*50)
    print("RESPONSE:")
    print("="*50)
    print(answer)
    print("\n" + "-"*50)
    print(f"Sources: {', '.join(sources)}")

if __name__ == "__main__":
    main()
