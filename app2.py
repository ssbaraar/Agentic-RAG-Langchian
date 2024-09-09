import os
import logging
import streamlit as st
import pandas as pd
import datasets
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from transformers.agents import Tool, ReactJsonAgent
from huggingface_hub import InferenceClient
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini AI
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

@st.cache_resource
def load_knowledge_base():
    # Load the knowledge base
    knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

    # Convert dataset to Document objects
    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
        for doc in knowledge_base
    ]

    logger.info(f"Loaded {len(source_docs)} documents from the knowledge base")
    return source_docs

@st.cache_resource
def initialize_vectordb(_docs_processed):
    # Initialize the embedding model
    logger.info("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

    # Create the vector database
    logger.info("Creating vector database...")
    vectordb = FAISS.from_documents(
        documents=_docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

    logger.info("Vector database created successfully")
    return vectordb

def process_documents(source_docs):
    # Initialize the text splitter
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Split documents and remove duplicates
    logger.info("Splitting documents...")
    docs_processed = []
    unique_texts = {}
    for doc in tqdm(source_docs):
        new_docs = text_splitter.split_documents([doc])
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)

    logger.info(f"Processed {len(docs_processed)} unique document chunks")
    return docs_processed

class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "text"

    def __init__(self, vectordb, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )

class GeminiEngine:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)

    def __call__(self, messages, stop_sequences=[]):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            stop_sequences=stop_sequences,
            temperature=0.5,
        ))
        return response.text

def run_agentic_rag(question: str, agent) -> str:
    enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
give a comprehensive answer to the question below.
Respond only to the question asked, response should be concise and relevant to the question.
If you cannot find information, do not give up and try calling your retriever again with different arguments!
Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
Your queries should not be questions but affirmative form sentences: e.g. rather than "How do I load a model from the Hub in bf16?", query should be "load a model from the Hub bf16 weights".

Question:
{question}"""

    return agent.run(enhanced_question)

def run_standard_rag(question: str, retriever_tool, llm_engine) -> str:
    context = retriever_tool(question)

    prompt = f"""Given the question and supporting documents below, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.

Question:
{question}

{context}
"""
    response = llm_engine([{"role": "user", "content": prompt}])
    return response

def main():
    st.title("RAG System with PDF Ingestion")

    # Initialize or load the knowledge base and vector database
    if 'source_docs' not in st.session_state:
        st.session_state.source_docs = load_knowledge_base()
    
    if 'docs_processed' not in st.session_state:
        st.session_state.docs_processed = process_documents(st.session_state.source_docs)
    
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = initialize_vectordb(st.session_state.docs_processed)

    # PDF upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            new_doc = Document(page_content=text, metadata={"source": uploaded_file.name})
            st.session_state.source_docs.append(new_doc)
        
        st.session_state.docs_processed = process_documents(st.session_state.source_docs)
        st.session_state.vectordb = initialize_vectordb(st.session_state.docs_processed)
        st.success("PDF added to the knowledge base!")

    # Initialize tools and engines
    retriever_tool = RetrieverTool(st.session_state.vectordb)
    llm_engine = GeminiEngine()
    agent = ReactJsonAgent(tools=[retriever_tool], llm_engine=llm_engine, max_iterations=4, verbose=2)

    # User input
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                agentic_answer = run_agentic_rag(question, agent)
                standard_answer = run_standard_rag(question, retriever_tool, llm_engine)

            st.subheader("Agentic RAG Answer:")
            st.write(agentic_answer)

            st.subheader("Standard RAG Answer:")
            st.write(standard_answer)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()