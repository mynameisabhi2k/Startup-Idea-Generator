import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import urllib.parse
import requests
from uuid import uuid4
from fpdf import FPDF
import faiss
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Initialize GPT-4
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, request_timeout=60)

# Streamlit UI Setup
st.set_page_config(page_title="Startup Idea Generator", layout="centered")
st.title("\U0001F680 Startup Idea Generator")
st.markdown("Generate startup ideas based on your interests, skills, or industry trends.")

# Session state for history
if "idea_history" not in st.session_state:
    st.session_state.idea_history = []

# User input
user_input = st.text_area("Describe your interests, skills, or the kind of problem you want to solve:", "AI + Healthcare")

# Prompt templates
idea_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a startup mentor. Based on the following input, generate a unique startup idea with the following structure:

1. **Problem Statement**
2. **Proposed Solution**
3. **Target Audience**
4. **Monetization Model**
5. **Why Now? (Market Relevance)**

User Input: {input}
"""
)

persona_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Create a detailed user persona for the target audience described below:

{input}

Include:
- Name
- Age
- Occupation
- Goals
- Pains
- Technology Comfort Level
"""
)

pitch_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Summarize the startup idea based on this input into a concise VC-style pitch (3-4 sentences):

{input}
"""
)

swot_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Provide a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for a startup idea based on:

{input}
"""
)

landing_page_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Write a mock landing page copy for this startup idea:
- Hero section headline
- Subheadline
- 3 bullet point value props
- CTA (call-to-action)

Startup Idea Input:
{input}
"""
)

mvp_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Based on this startup idea, define the MVP (minimum viable product):
- Key Features
- User Flow
- Tech Stack Suggestion

Startup Input:
{input}
"""
)

# Chains
idea_chain = LLMChain(llm=llm, prompt=idea_prompt)
persona_chain = LLMChain(llm=llm, prompt=persona_prompt)
pitch_chain = LLMChain(llm=llm, prompt=pitch_prompt)
swot_chain = LLMChain(llm=llm, prompt=swot_prompt)
landing_chain = LLMChain(llm=llm, prompt=landing_page_prompt)
mvp_chain = LLMChain(llm=llm, prompt=mvp_prompt)

# DuckDuckGo Search Tool
class DuckDuckGoSearch:
    def run(self, query: str) -> str:
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = []

                for topic in data.get("RelatedTopics", [])[:3]:
                    if "Text" in topic and "FirstURL" in topic:
                        results.append(f"- {topic['Text']}: {topic['FirstURL']}")
                    elif "Topics" in topic:
                        for subtopic in topic.get("Topics", [])[:2]:
                            if "Text" in subtopic and "FirstURL" in subtopic:
                                results.append(f"- {subtopic['Text']}: {subtopic['FirstURL']}")

                if len(results) < 3 and "Results" in data:
                    for result in data.get("Results", [])[:3 - len(results)]:
                        if "Text" in result and "FirstURL" in result:
                            results.append(f"- {result['Text']}: {result['FirstURL']}")

                if data.get("AbstractText") and data.get("AbstractURL"):
                    results.insert(0, f"**Overview**: {data['AbstractText']} [Source]({data['AbstractURL']})")

                return "\n\n".join(results) if results else "No relevant search results found."
            else:
                return f"Search failed with status code {response.status_code}."
        except Exception as e:
            return f"Search failed: {str(e)}"

search_tool = Tool(
    name="Search",
    func=DuckDuckGoSearch().run,
    description="Useful for finding recent trends or validating market interest."
)

agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Vector Store Setup
if os.path.exists("vector_store.pkl"):
    with open("vector_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)
else:
    vectorstore = FAISS.from_texts([""], OpenAIEmbeddings())

# Generate Idea
if st.button("Generate Idea") and user_input:
    with st.spinner("Thinking hard and brainstorming ideas... \U0001F4A1"):
        try:
            idea = idea_chain.run(user_input)
            try:
                trends = agent.run(f"startup trends related to {user_input}")
            except:
                trends = "Unable to fetch market trends."
            try:
                competitors = agent.run(f"top startups or products related to {user_input}")
            except:
                competitors = "Unable to fetch competitor data."
            try:
                persona = persona_chain.run(user_input)
            except:
                persona = "Unable to generate persona."
            try:
                pitch = pitch_chain.run(idea)
            except:
                pitch = "Unable to generate pitch."
            try:
                swot = swot_chain.run(idea)
            except:
                swot = "Unable to generate SWOT."
            try:
                mvp = mvp_chain.run(idea)
            except:
                mvp = "Unable to generate MVP."
            try:
                landing_copy = landing_chain.run(idea)
            except:
                landing_copy = "Unable to generate landing copy."

            vectorstore.add_texts([idea])
            with open("vector_store.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

            session_id = str(uuid4())[:8]
            st.session_state.idea_history.append({"id": session_id, "input": user_input, "idea": idea, "pitch": pitch})

            st.markdown("### \U0001F4A1 Your Startup Idea")
            st.markdown(idea)

            with st.expander("\U0001F465 Target User Persona"):
                st.markdown(persona)

            with st.expander("\U0001F3CB\ufe0f VC-Style Pitch Summary"):
                st.markdown(pitch)

            with st.expander("\U0001F6E0 SWOT Analysis"):
                st.markdown(swot)

            with st.expander("\U0001F527 MVP Scope and Initial Feature Set"):
                st.markdown(mvp)

            with st.expander("\U0001F4BB Landing Page Copy"):
                st.markdown(landing_copy)

            st.markdown("### \U0001F4CA Market Trends")
            st.markdown(trends)

            st.markdown("### \U0001F4BC Competitor Snapshot")
            st.markdown(competitors)

            st.markdown("### \U0001F44D Was this idea helpful?")
            col1, col2 = st.columns(2)
            if col1.button("\U0001F44D Yes, I like it"):
                st.success("Awesome! Glad it helped.")
            if col2.button("\U0001F44E Needs improvement"):
                try:
                    refined = idea_chain.run(user_input + " but make it more innovative and unique")
                    st.markdown("### \U0001F501 Refined Idea")
                    st.markdown(refined)
                except:
                    st.error("Could not refine idea.")

            if st.button("\U0001F4C4 Download Idea as PDF"):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, f"Startup Idea Generator Result\n\nInput: {user_input}\n\nIdea:\n{idea}\n\nPitch:\n{pitch}\n\nSWOT:\n{swot}")
                    file_path = f"idea_{session_id}.pdf"
                    pdf.output(file_path)
                    with open(file_path, "rb") as f:
                        st.download_button("Download PDF", data=f, file_name=file_path)
                except:
                    st.error("Could not generate PDF.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Similar idea search
search_query = st.text_input("\U0001F50D Find similar startup ideas")
if st.button("Search Similar Ideas") and search_query:
    docs = vectorstore.similarity_search(search_query, k=3)
    st.markdown("### \U0001F4D6 Similar Ideas")
    for i, doc in enumerate(docs, 1):
        st.markdown(f"**{i}.** {doc.page_content}")

# Session history
if st.session_state.idea_history:
    with st.expander("\U0001F552 Session History"):
        for entry in reversed(st.session_state.idea_history):
            st.markdown(f"**{entry['id']}** — _{entry['input']}_")
            st.markdown(entry['idea'])
            st.markdown("---")

