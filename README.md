# 🎥 YouTube Video RAG Chatbot

A Retrieval-Augmented Generation (RAG) based AI chatbot that can answer questions from YouTube videos using subtitles/transcripts.

This project extracts subtitles from a YouTube video, processes the transcript, creates embeddings using Hugging Face models, stores them in a FAISS vector database, and uses a Groq LLM to answer user questions based only on the video content.

---

# 🚀 Features

- Extract subtitles from YouTube videos
- Clean and preprocess transcripts
- Split transcripts into chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS vector database
- Semantic similarity search
- Ask questions from video content
- Summarize YouTube videos
- Context-aware AI responses using Groq LLM
- Built using LangChain RAG pipeline

---

# 🛠️ Technologies Used

- Python
- LangChain
- FAISS
- Hugging Face Embeddings
- Groq API
- yt-dlp
- YouTube Transcript API

---

# 📂 Project Workflow

```text
YouTube Video
      ↓
Subtitle Extraction
      ↓
Transcript Cleaning
      ↓
Text Splitting
      ↓
Embedding Generation
      ↓
FAISS Vector Database
      ↓
Retriever
      ↓
Groq LLM
      ↓
AI Generated Answer
```

---

# 📦 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/youtube-rag-chatbot.git

cd youtube-rag-chatbot
```

---

## 2️⃣ Install Dependencies

```bash
pip install -q sentence-transformers

pip uninstall -y google-generativeai langchain-google-genai

pip install -U google-generativeai langchain-google-genai

pip install -q youtube-transcript-api langchain-community langchain-openai faiss-cpu tiktoken python-dotenv

pip install -q langchain

pip install -q langchain-text-splitters

pip install -q langchain-huggingface

pip install -q langchain-groq

pip install -q yt-dlp
```

---

# 🔑 Setup API Key

Create a `.env` file in the project root folder.

```env
GROQ_API_KEY=your_groq_api_key
```

Or directly set it in Python:

```python
import os

os.environ["GROQ_API_KEY"] = "your_groq_api_key"
```

---

# ▶️ How to Run the Project

## Step 1: Import Libraries

```python
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
```

---

## Step 2: Download YouTube Subtitles

```python
import yt_dlp

url = "https://www.youtube.com/watch?v=VIDEO_ID"

ydl_opts = {
    'writesubtitles': True,
    'writeautomaticsub': True,
    'skip_download': True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
```

---

## Step 3: Read Subtitle File

```python
subtitle_file = "video.en.vtt"

with open(subtitle_file, "r", encoding="utf-8") as f:
    transcript_text = f.read()
```

---

## Step 4: Clean Transcript

```python
def clean_vtt(vtt_text):
    lines = vtt_text.split("\n")

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if "-->" in line:
            continue

        if line.startswith("WEBVTT"):
            continue

        if line == "":
            continue

        if line.isdigit():
            continue

        cleaned_lines.append(line)

    return " ".join(cleaned_lines)

clean_transcript = clean_vtt(transcript_text)
```

---

## Step 5: Split Text into Chunks

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([clean_transcript])
```

---

## Step 6: Generate Embeddings

```python
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

## Step 7: Store Embeddings in FAISS

```python
vector_store = FAISS.from_documents(
    chunks,
    embedding
)
```

---

## Step 8: Create Retriever

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

---

## Step 9: Initialize Groq LLM

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.environ["GROQ_API_KEY"]
)
```

---

## Step 10: Create Prompt Template

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is not found in context, say:
"I could not find this information in the video."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)
```

---

## Step 11: Build the RAG Chain

```python
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser
```

### Format Retrieved Documents

```python
def format_docs(retrieved_docs):
    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )
    return context_text
```

### Parallel Chain

```python
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
```

### Output Parser

```python
parser = StrOutputParser()
```

### Final Main Chain

```python
main_chain = parallel_chain | prompt | llm | parser
```

---

# 💬 Example Queries

## Ask Questions

```python
main_chain.invoke("Who is Demis?")
```

---

## Summarize Video

```python
main_chain.invoke("Can you summarize the video?")
```

---

## Topic Detection

```python
main_chain.invoke(
    "Is Narendra Modi discussed in this video?"
)
```

---

# 📌 Example Output

```text
Demis Hassabis is the co-founder of DeepMind mentioned in the video.
```

---

# ⚡ Future Improvements

- Streamlit Web UI
- Multi-video support
- Chat history memory
- Hybrid search
- Better chunking strategies
- Voice assistant integration
- PDF and website support
- Deploy on Hugging Face Spaces

---

# 🔒 Security Note

Never expose your API keys publicly.

Use environment variables or `.env` files for storing secrets.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Shashank  
B.Tech CSE | AI & ML Enthusiast

Interested in:
- Artificial Intelligence
- Machine Learning
- Backend Development
- RAG Systems
- NLP & LLM Applications

---
