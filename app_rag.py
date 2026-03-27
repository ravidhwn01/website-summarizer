import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
import hashlib
import os


st.set_page_config(page_title="LangChain: RAG Q&A on Website", page_icon="🦜")
st.title("RAG Q&A on Website")
st.subheader("Ask any question about a webpage")

with st.sidebar:
    # groq_api_key = st.text_input("GROQ_API_KEY", value="", type="password")
    groq_api_key = os.getenv("GROQ_API_KEY")

url = st.text_input("Enter Website URL", placeholder="https://example.com/article")
question = st.text_input("Ask a question about the website", placeholder="What is this page about?")

prompt_template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Be concise and helpful.

Context: {context}

Question: {question}

Helpful Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_sources(docs):
    sources = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        chunk = doc.metadata.get("chunk", index)
        preview = " ".join(doc.page_content.split())[:350]
        sources.append(f"**{index}. Source:** {source}  \n**Chunk:** {chunk}  \n**Preview:** {preview}...")
    return sources


def collection_name_for_url(url):
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"web_{url_hash}"

@st.cache_data(show_spinner=False)
def load_website(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }
    session = requests.Session()
    response = session.get(url, headers=headers, timeout=30, allow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    main_content = (
        soup.find("article") or
        soup.find("main") or
        soup.find(class_=lambda c: c and "content" in c.lower()) or
        soup.find("body")
    )
    text = main_content.get_text(separator=" ", strip=True) if main_content else soup.get_text()
    text = " ".join(text.split())
    return [Document(page_content=text, metadata={"source": url})]


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_retriever(url, docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    for index, chunk in enumerate(splits, start=1):
        chunk.metadata["chunk"] = index

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name_for_url(url),
    )

    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12})


if st.button("Get Answer"):
    if not groq_api_key.strip():
        st.error("Please provide your Groq API key.")
    elif not url.strip():
        st.error("Please enter a website URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL (e.g. https://example.com).")
    elif "youtube.com" in url or "youtu.be" in url:
        st.error("YouTube URLs are not supported. Please enter a website URL.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        try:
            with st.spinner("Fetching content, creating vector store, and generating answer..."):
                llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
                
                docs = load_website(url)
                retriever = build_retriever(url, docs)
                retrieved_docs = retriever.invoke(question)
                context_text = format_docs(retrieved_docs)

                rag_chain = prompt | llm | StrOutputParser()
                answer = rag_chain.invoke({"context": context_text, "question": question})
                st.success(answer)

                with st.expander("Show retrieved sources"):
                    for source_block in format_sources(retrieved_docs):
                        st.markdown(source_block)
                        st.divider()

        except requests.exceptions.Timeout:
            st.error("⏱️ The website took too long to respond. Try a different URL.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Could not connect to the website. Check the URL or your internet.")
        except requests.exceptions.HTTPError as e:
            st.error(f"🚫 Website blocked the request (HTTP {e.response.status_code}). Try another URL.")
        except Exception as e:
            st.exception(f"Exception: {e}")
