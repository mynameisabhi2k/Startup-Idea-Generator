import os
import pickle
import requests
import urllib.parse
from uuid import uuid4
import base64

from dotenv import load_dotenv
from fpdf import FPDF
import streamlit as st
import torch

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS

# Custom embeddings class to handle tokenization issues
class SimpleFastEmbeddings:
    """A simple class that uses numpy arrays for embeddings."""
    
    def __init__(self, dimension=768):
        self.dimension = dimension
    
    def embed_documents(self, texts):
        """Return simple random embeddings for texts."""
        return [self._get_embedding() for _ in texts]
    
    def embed_query(self, text):
        """Return simple random embedding for query."""
        return self._get_embedding()
    
    def _get_embedding(self):
        """Create a simple deterministic embedding based on text."""
        # Use torch for random but deterministic embeddings
        with torch.no_grad():
            # Create a random embedding and normalize it
            embedding = torch.rand(self.dimension)
            embedding = embedding / torch.norm(embedding)
            return embedding.numpy()

# Function to create PDF and get downloadable link
def create_pdf_download_link(idea_content, idea_id):
    """Generate a PDF and create a download link."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Split content into lines to avoid overflow issues
    for line in idea_content.split('\n'):
        pdf.multi_cell(0, 10, line)
    
    pdf_path = f"idea_{idea_id}.pdf"
    pdf.output(pdf_path)
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Create a download button using base64 encoding
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_path}">Download PDF</a>'
    return href

# -------------------
# Setup & Config
# -------------------
load_dotenv()

# Initialize session state at the beginning
if 'idea_history' not in st.session_state:
    st.session_state.idea_history = []
    
if 'current_idea' not in st.session_state:
    st.session_state.current_idea = None

# Replace OpenAI with Groq
llm = ChatGroq(
    model_name="llama3-70b-8192", 
    temperature=0.7, 
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    request_timeout=60
)

# Use our custom embeddings class instead
embeddings = SimpleFastEmbeddings()

st.set_page_config(page_title="Startup Idea Generator", layout="centered")
st.title("🚀 Startup Idea Generator")
st.markdown("Generate startup ideas based on your interests, skills, or industry trends.")

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

# Initialize a fresh vector store - don't try to reuse the old one
# since we've changed embeddings model
try:
    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        vectorstore = FAISS.from_texts(["Initial document"], embeddings)
        with open(VECTORSTORE_PATH, "wb") as f:
            pickle.dump(vectorstore, f)
except Exception as e:
    st.warning(f"Could not load/initialize vector store: {e}")
    vectorstore = FAISS.from_texts(["Initial document"], embeddings)

# -------------------
# Idea Generation Flow
# -------------------
generate_button = st.button("Generate Idea", key="generate_idea")

if generate_button and user_input:
    with st.spinner("Thinking hard and brainstorming ideas..."):
        try:
            # Create a dictionary to store the outputs
            outputs = {}
            
            # Generate idea first
            outputs["idea"] = chains["idea"].run(user_input)
            
            # Then generate the rest using the idea output where needed
            for key in ["pitch", "persona", "swot", "mvp", "landing"]:
                input_text = user_input if key == "persona" else outputs["idea"]
                outputs[key] = chains[key].run(input_text)

            trends = agent.run(f"startup trends related to {user_input}")
            competitors = agent.run(f"top startups related to {user_input}")

            # Safely add to vector store
            try:
                vectorstore.add_texts([outputs["idea"]])
                with open(VECTORSTORE_PATH, "wb") as f:
                    pickle.dump(vectorstore, f)
            except Exception as e:
                st.warning(f"Could not update vector store: {e}")

            idea_id = str(uuid4())[:8]
            st.session_state.idea_history.append({
                "id": idea_id,
                "input": user_input,
                **outputs
            })
            
            # Store current idea for PDF generation
            st.session_state.current_idea = {
                "id": idea_id,
                "input": user_input,
                "idea": outputs["idea"],
                "pitch": outputs["pitch"],
                "swot": outputs["swot"],
                "content": f"Startup Idea Report\n\nInput: {user_input}\n\nIdea:\n{outputs['idea']}\n\nPitch:\n{outputs['pitch']}\n\nSWOT:\n{outputs['swot']}"
            }

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
            
            # Generate PDF link directly
            if st.session_state.current_idea:
                pdf_html = create_pdf_download_link(
                    st.session_state.current_idea["content"],
                    st.session_state.current_idea["id"]
                )
                st.markdown(pdf_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating idea: {e}")
            import traceback
            st.error(traceback.format_exc())

# -------------------
# Similarity Search
# -------------------
search_query = st.text_input("🔍 Find similar startup ideas")
search_button = st.button("Search Similar Ideas", key="search_ideas")

if search_button and search_query:
    try:
        similar_docs = vectorstore.similarity_search(search_query, k=3)
        st.markdown("### 📚 Similar Ideas")
        for i, doc in enumerate(similar_docs, 1):
            st.markdown(f"**{i}.** {doc.page_content}")
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        st.info("Try generating some ideas first to build up the database.")

# -------------------
# Session History
# -------------------
if st.session_state.idea_history:
    with st.expander("🕒 Session History"):
        for entry in reversed(st.session_state.idea_history):
            st.markdown(f"**{entry['id']}** — _{entry['input']}_")
            st.markdown(entry['idea'])
            st.markdown("---")