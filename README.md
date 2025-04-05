
# 🚀 Startup Idea Generator with GPT-4 + LangChain + Vector DB

This is a Streamlit-based AI tool that helps you brainstorm startup ideas based on your interests or skills using GPT-4. It also includes market analysis, MVP planning, landing page content, SWOT, competitor research, and user persona generation.

## ✨ Features

- Generate complete startup ideas from a simple user input prompt.
- VC-style pitch, MVP, SWOT analysis, user persona, and landing page.
- Live web search integration using DuckDuckGo.
- Vector database support with FAISS to store and retrieve similar ideas.
- Interactive "Find Similar Ideas" semantic search.
- Export your idea as a downloadable PDF.
- Session history and refinement options.

## 🛠️ Tech Stack

- Python + Streamlit for the web interface
- GPT-4 via LangChain for all LLM-based features
- FAISS (Facebook AI Similarity Search) for semantic search
- DuckDuckGo API for real-time market research
- FPDF for PDF export

## 🧪 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure to set your OpenAI API Key using a `.env` file:

```bash
OPENAI_API_KEY=your_openai_key_here
```

## 📂 File Structure

```
├── app.py              # Main Streamlit app
├── requirements.txt    # Required Python packages
├── README.md           # This documentation
```

## 📸 Screenshot

Check the output folder for the results

---

Built with ❤️ by Abhilash using GPT-4 + LangChain + Vector DB.
