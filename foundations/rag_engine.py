from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import sqlite3, os, asyncio
import google.generativeai as genai
from dotenv import load_dotenv

# ─── Environment & AI setup ──────────────────────────────────────────────────
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ─── Chunking & Indexing ─────────────────────────────────────────────────────
def prepare_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def build_vector_index(text, save_path="rag_index"):
    chunks     = prepare_chunks(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs         = FAISS.from_texts(chunks, embeddings)
    vs.save_local(save_path)

def load_vector_index(path="rag_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings)

# ─── SQLite Cache ────────────────────────────────────────────────────────────
def init_cache():
    conn = sqlite3.connect("response_cache.db")
    conn.execute("CREATE TABLE IF NOT EXISTS cache (question TEXT PRIMARY KEY, answer TEXT)")
    conn.commit()
    return conn

def get_cached_answer(conn, q):
    cur = conn.cursor()
    cur.execute("SELECT answer FROM cache WHERE question = ?", (q,))
    row = cur.fetchone()
    return row[0] if row else None

def store_answer(conn, q, a):
    conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (q, a))
    conn.commit()

# ─── Async RAG Query ─────────────────────────────────────────────────────────
async def get_rag_response_async(query: str, full_text: str, history: str):
    conn = init_cache()
    # Cache key includes history to avoid incorrect hits on follow-up questions
    cache_key = f"{query}|{history}"
    cached = get_cached_answer(conn, cache_key)
    if cached:
        return cached

    retriever = load_vector_index().as_retriever(search_kwargs={"k": 5}) # Increased k for more context
    docs      = retriever.invoke(query)
    context   = "\n\n".join(d.page_content for d in docs)

    prompt = (
        "You are a helpful AI assistant for Venkatesh Narra. You MUST speak in the first person, as if you ARE Venkatesh Narra (e.g., use 'I', 'my', 'me'). Your persona is professional, friendly, and an expert in your own career.\n\n"
        "You must answer questions based ONLY on the information in the CONTEXT section and the CONVERSATION HISTORY. Do not use any outside knowledge.\n\n"
        "### INSTRUCTIONS FOR ANSWERING ###\n"
        "1.  **Use Conversation History:** If the user's question is a follow-up (e.g., 'tell me more about that', 'why?'), use the history to understand the original topic.\n"
        "2.  **Synthesize Information:** Combine details from all relevant sections (Professional Experience, Resume, Detailed Projects, GitHub) to form a complete answer. Do not just copy-paste one section.\n"
        "3.  **For 'why hire me' or behavioral questions:** Answer using the STAR method (Situation, Task, Action, Result) by drawing from my project experiences. Be persuasive and confident.\n"
        "4.  **For 'what are your skills' questions:** List technical skills clearly, grouped by category (e.g., Languages, Frameworks, Cloud/DevOps, AI/ML).\n"
        "5.  **For 'recent work/projects' questions:** Focus on the 'Veritis Group Inc.' experience and the 'AI-Powered Testing Agent' and 'Multi-Model AI Chat Platform' projects. Synthesize details from all these sections to give a full picture of my latest work.\n"
        "6.  **For past projects:** Refer to the TCS loan platform and other projects listed in the GitHub section.\n\n"
        "If the information to answer the question is NOT in the context, you MUST respond with EXACTLY this phrase: 'I don't have enough information on that. Please drop your email, and I'll get back to you.'\n\n"
        f"CONVERSATION HISTORY:\n---------------------\n{history}\n---------------------\n\n"
        f"CONTEXT:\n---------------------\n{context}\n---------------------\n\n"
        f"Question: {query}\n"
        "Answer (as Venkatesh Narra):"
    )

    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY is not set. Cannot query the model."

    gem = genai.GenerativeModel("gemini-1.5-flash")
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, gem.generate_content, prompt)
    answer = resp.text.strip()

    store_answer(conn, cache_key, answer)
    return answer
