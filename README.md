# Prism AI: Knowledgebase Q&A Engine

Prism AI is a custom Q&A system designed to provide instant answers based on a specific FAQ knowledge base. It combines the power of **Google Palm LLM** with **HuggingFace Instructor Embeddings** for high-accuracy information retrieval.

##  Tech Stack
* **LLM:** Google Palm
* **Framework:** LangChain
* **Embeddings:** HuggingFace Instructor Embeddings
* **Vector Database:** FAISS
* **Interface:** Streamlit / FastAPI

##  How It Works
The system follows a Retrieval-Augmented Generation (RAG) workflow:
1. **Knowledge Creation:** It processes existing FAQ data and converts it into numerical vectors.
2. **Vector Storage:** These vectors are stored locally in a `faiss_index` directory.
3. **Smart Retrieval:** When a user asks a question, the system finds the most relevant FAQ and uses Google Palm to generate a natural response.

##  Application Preview
![images alt](https://github.com/muzammilrahman-code/prism-ai-chatbot/blob/0fac635787a9c92ee4830ae1e3302552a2372906/Screenshot%202026-02-08%20192142.png)

---

##  Getting Started

### 1. Build the Knowledgebase
Before asking questions, you must initialize the AI's memory:
* Launch the application.
* Click the **"Create Knowledge Base"** button.
* **Wait:** The system is currently generating embeddings (this may take a moment).
* Once finished, a folder named `faiss_index` will appear in your project directory.

### 2. Ask Questions
Once the knowledge base is ready:
* Type your question into the **Question** input box.
* Hit **Enter**.
* Prism AI will retrieve the answer from the stored FAQs instantly.

---

##  Project Structure
* `api.py`: FastAPI implementation for backend requests.
* `langchain_helper.py`: The "brain" of the project (FAISS logic, Google Palm integration, and HuggingFace embeddings).
* `app.py` (or Streamlit file): The user interface.
* `faiss_index/`: Local directory where the processed knowledge is stored (Generated after setup).

---

##  Setup Instructions
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Add your `GOOGLE_API_KEY` to your environment variables or `.env` file.
4. Run the app: `streamlit run main.py` or `uvicorn api:app`.

