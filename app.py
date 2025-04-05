import os
import pickle
import requests
import urllib.parse
from uuid import uuid4

from dotenv import load_dotenv
from fpdf import FPDF
import streamlit as st
import faiss

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# -------------------
# Setup & Config
# -------------------
load_dotenv()

llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, request_timeout=60)

st.set_page_config(page_title="Startup Idea Generator", layout="centered")
st.title("🚀 Startup Idea Generator")
st.markdown("Generate startup ideas based on your interests, skills, or industry trends.")

if "idea_history" not in st.session_state:
    st.session_state.idea_history = []

user_input = st.text_area(
    "Describe your interests, skills, or the kind of problem you want to solve:",
    placeholder="AI + Healthcare"
)

# -------------------
# Prompt Templates
# -------------------
def build_prompt(template: str) -> PromptTemplate:
    return PromptTemplate(input_variables=["input"], template=template)

prompt_templates = {
    "idea": """
        You are a startup mentor. Based on the following input, generate a unique startup idea with the following structure:

        1. Problem Statement
        2. Proposed Solution
        3. Target Audience
        4. Monetization Model
        5. Why Now? (Market Relevance)

        User Input: {input}
    """,
    "persona": """
        Create a detailed user persona for the target audience described below:

        {input}

        Include:
        - Name
        - Age
        - Occupation
        - Goals
        - Pains
        - Technology Comfort Level
    """,
    "pitch": """
        Summarize the startup idea based on this input into a concise VC-style pitch (3-4 sentences):

        {input}
    """,
    "swot": """
        Provide a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for a startup idea based on:

        {input}
    """,
    "landing": """
        Write a mock landing page copy for this startup idea:
        - Hero section headline
        - Subheadline
        - 3 bullet point value props
        - CTA (call-to-action)

        Startup Idea Input:
        {input}
    """,
    "mvp": """
        Based on this startup idea, define the MVP (minimum viable product):
        - Key Features
        - User Flow
        - Tech Stack Suggestion

        Startup Input:
        {input}
    """
}

chains = {
    name: LLMChain(llm=llm, prompt=build_prompt(template))
    for name, template in prompt_templates.items()
}

# -------------------
# Tooling
# -------------------
class DuckDuckGoSearch:
    def run(self, query: str) -> str:
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return f"Search failed with status code {resp.status_code}."

            results = [
                f"- {item['Text']}: {item['FirstURL']}"
                for item in resp.json().get("RelatedTopics", [])
                if "Text" in item and "FirstURL" in item
            ]
            return "\n\n".join(results[:3]) or "No relevant search results found."
        except Exception as e:
            return f"Search error: {e}"

agent = initialize_agent(
    tools=[
        Tool(
            name="Search",
            func=DuckDuckGoSearch().run,
            description="Finds trends and competitors"
        )
    ],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# -------------------
# Vector DB (FAISS)
# -------------------
VECTORSTORE_PATH = "vector_store.pkl"

if os.path.exists(VECTORSTORE_PATH):
    with open(VECTORSTORE_PATH, "rb") as f:
        vectorstore = pickle.load(f)
else:
    vectorstore = FAISS.from_texts([""], OpenAIEmbeddings())

# -------------------
# Idea Generation Flow
# -------------------
if st.button("Generate Idea") and user_input:
    with st.spinner("Thinking hard and brainstorming ideas..."):
        try:
            outputs = {
                key: chains[key].run(user_input if key in ["persona"] else None or outputs.get("idea"))
                for key in ["idea", "pitch", "persona", "swot", "mvp", "landing"]
            }

            trends = agent.run(f"startup trends related to {user_input}")
            competitors = agent.run(f"top startups related to {user_input}")

            vectorstore.add_texts([outputs["idea"]])
            with open(VECTORSTORE_PATH, "wb") as f:
                pickle.dump(vectorstore, f)

            idea_id = str(uuid4())[:8]
            st.session_state.idea_history.append({
                "id": idea_id,
                "input": user_input,
                **outputs
            })

            # UI Rendering
            st.subheader("💡 Your Startup Idea")
            st.markdown(outputs["idea"])

            with st.expander("👥 User Persona"):
                st.markdown(outputs["persona"])
            with st.expander("🏋 VC-Style Pitch"):
                st.markdown(outputs["pitch"])
            with st.expander("🛠 SWOT Analysis"):
                st.markdown(outputs["swot"])
            with st.expander("🔧 MVP Definition"):
                st.markdown(outputs["mvp"])
            with st.expander("💻 Landing Page"):
                st.markdown(outputs["landing"])

            st.markdown("### 📈 Market Trends")
            st.markdown(trends)

            st.markdown("### 💼 Competitor Snapshot")
            st.markdown(competitors)

            if st.button("📄 Download as PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, f"Startup Idea Report\n\nInput: {user_input}\n\nIdea:\n{outputs['idea']}\n\nPitch:\n{outputs['pitch']}\n\nSWOT:\n{outputs['swot']}")
                pdf_path = f"idea_{idea_id}.pdf"
                pdf.output(pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", data=f, file_name=pdf_path)

        except Exception as e:
            st.error(f"Error generating idea: {e}")

# -------------------
# Similarity Search
# -------------------
search_query = st.text_input("🔍 Find similar startup ideas")
if st.button("Search Similar Ideas") and search_query:
    similar_docs = vectorstore.similarity_search(search_query, k=3)
    st.markdown("### 📚 Similar Ideas")
    for i, doc in enumerate(similar_docs, 1):
        st.markdown(f"**{i}.** {doc.page_content}")

# -------------------
# Session History
# -------------------
if st.session_state.idea_history:
    with st.expander("🕒 Session History"):
        for entry in reversed(st.session_state.idea_history):
            st.markdown(f"**{entry['id']}** — _{entry['input']}_")
            st.markdown(entry['idea'])
            st.markdown("---")
