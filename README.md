# 📄 Document-Based RAG Application

A Streamlit-based Retrieval-Augmented Generation RAG application that allows you to chat with your PDF documents. The app supports both Online mode using OpenAI models and Offline mode using a local Ollama LLM, making it flexible for experimentation and learning.

---

## 🚀 Features

* Chat with multiple PDF documents
* Retrieval-Augmented Generation using LangChain
* Vector storage with Chroma DB
* HuggingFace sentence transformer embeddings
* Toggle between Online OpenAI and Offline Ollama models
* Clean and simple Streamlit UI with custom CSS

---

## 🧠 Tech Stack

* Python
* Streamlit
* LangChain
* Chroma Vector Database
* HuggingFace Embeddings
* OpenAI API
* Ollama

---

## 📁 Project Structure

```
Document-Based-RAG-Application/
│── data/                 # PDF documents
│── chroma_db/            # Vector database storage
│── style.css             # Custom UI styling
│── doc.py                # Main Streamlit application
│── requirements.txt      # Python dependencies
│── .env.example          # Environment variable template
│── .gitignore
│── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```
git clone https://github.com/gadher80/Document-Based-RAG-Application.git
cd Document-Based-RAG-Application
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Environment Variables

Create a `.env` file in the project root

```
OPENAI_API_KEY=your_openai_api_key
```

For security, never commit `.env` to GitHub

---

## ▶️ Run the Application

```
streamlit run doc.py
```

The app will open in your browser

---

## 🔄 Online vs Offline Mode

Online Mode

* Uses OpenAI Chat models
* Faster responses
* Higher quality answers
* Requires OpenAI API key

Offline Mode

* Uses local Ollama model
* No internet required
* Fully local inference
* Requires Ollama installed

---

## 🧪 Ollama Setup

Install Ollama from

[https://ollama.com](https://ollama.com)

Pull required model

```
ollama pull llama3
```

Ensure Ollama is running before using Offline mode

---

## 📌 Notes

* PDFs must be placed inside the `data/` folder
* Vector database persists in `chroma_db/`
* First run may take time due to embedding creation

---

## 📜 License

This project is for educational and learning purposes

---

## 🙌 Author

Hardik Gadher

Data Engineer | BI Engineer | AI Enthusiast
