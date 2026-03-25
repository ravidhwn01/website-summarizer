# 🦜 LangChain: Summarize Any Webpage

A simple and powerful web application that extracts content from any webpage and generates a concise summary using LLMs.

Built with **Streamlit**, **LangChain**, and **Groq API**.

---

## 🚀 Features

* 🌐 Extracts content from any webpage
* 🧹 Cleans HTML (removes scripts, ads, navigation, etc.)
* 🤖 Generates AI-powered summaries (300 words)
* ⚡ Fast inference using Groq LLM (LLaMA 3.1)
* 🖥️ Simple and interactive UI with Streamlit
* 🔐 Secure API key handling using `.env`

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* Groq API
* BeautifulSoup (Web Scraping)
* Requests

---

## 📂 Project Structure

```
LangChain_TextSummarize_any_webpage/
│
├── app.py                # Main Streamlit application
├── requirements.txt     # Dependencies
├── .env                 # API keys (not included in repo)
└── README.md            # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/LangChain_TextSummarize_any_webpage.git
cd LangChain_TextSummarize_any_webpage
```

---

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 🔑 Setup Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```
python -m streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧠 How It Works

1. User inputs a webpage URL
2. App fetches HTML using `requests`
3. Cleans content using `BeautifulSoup`
4. Extracts main text from page
5. Sends text to LangChain summarization chain
6. Groq LLM generates a concise summary
7. Output is displayed in Streamlit UI

---

## ⚠️ Limitations

* ❌ Does not support YouTube URLs
* 📏 Input text is limited (~8000 characters)
* 🚫 Some websites may block scraping

---

## 🔮 Future Improvements

* Support for YouTube/video summarization
* PDF and document summarization
* Adjustable summary length
* Multi-language support
* Better content extraction (readability-based parsing)

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Developed by Ravi

---
