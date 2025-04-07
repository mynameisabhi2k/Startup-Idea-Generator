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

# Load environment variables
load_dotenv()

# Initialize GPT-4 with timeout
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, request_timeout=60)

# Initialize session state for history
if "idea_history" not in st.session_state:
    st.session_state.idea_history = []

# Streamlit UI
st.set_page_config(page_title="Startup Idea Generator", layout="centered")
st.title("\U0001F680 Startup Idea Generator")
st.markdown("Generate startup ideas based on your interests, skills, or industry trends.")

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
            # DuckDuckGo search API endpoint
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract results from both the RelatedTopics and Results sections
                results = []
                
                # Get related topics (these are usually more informative)
                for topic in data.get("RelatedTopics", [])[:3]:
                    if "Text" in topic and "FirstURL" in topic:
                        results.append(f"- {topic['Text']}: {topic['FirstURL']}")
                    elif "Topics" in topic:  # Handle nested topics
                        for subtopic in topic.get("Topics", [])[:2]:
                            if "Text" in subtopic and "FirstURL" in subtopic:
                                results.append(f"- {subtopic['Text']}: {subtopic['FirstURL']}")
                
                # If we don't have enough results, add some from the Results section
                if len(results) < 3 and "Results" in data:
                    for result in data.get("Results", [])[:3 - len(results)]:
                        if "Text" in result and "FirstURL" in result:
                            results.append(f"- {result['Text']}: {result['FirstURL']}")
                
                # Add the abstract if available
                if data.get("AbstractText") and data.get("AbstractURL"):
                    results.insert(0, f"**Overview**: {data['AbstractText']} [Source]({data['AbstractURL']})")
                
                if results:
                    return "\n\n".join(results)
                else:
                    return "No relevant search results found for the query."
            else:
                return f"Search failed with status code {response.status_code}. Please try again later."
        except requests.exceptions.Timeout:
            return "Search request timed out. Please try again later."
        except requests.exceptions.RequestException as e:
            return f"HTTP Error: {str(e)}"
        except ValueError as e:  # JSON parsing error
            return "Invalid response from search API."
        except Exception as e:
            return f"Search failed due to an error: {str(e)}"

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

# Generate idea with proper error handling
if st.button("Generate Idea") and user_input:
    with st.spinner("Thinking hard and brainstorming ideas... \U0001F4A1"):
        try:
            # Generate core idea
            idea = idea_chain.run(user_input)
            
            # Get market data with error handling
            try:
                trends = agent.run(f"startup trends related to {user_input}")
            except Exception as e:
                st.warning(f"Could not fetch market trends: {str(e)}")
                trends = "Unable to fetch trends at this time."
                
            try:
                competitors = agent.run(f"top startups or products related to {user_input}")
            except Exception as e:
                st.warning(f"Could not fetch competitor data: {str(e)}")
                competitors = "Unable to fetch competitor data at this time."
            
            # Generate additional content with error handling
            try:
                persona = persona_chain.run(user_input)
            except Exception as e:
                st.warning(f"Could not generate persona: {str(e)}")
                persona = "Unable to generate persona at this time."
                
            try:
                pitch = pitch_chain.run(idea)
            except Exception as e:
                st.warning(f"Could not generate pitch: {str(e)}")
                pitch = "Unable to generate pitch at this time."
                
            try:
                swot = swot_chain.run(idea)
            except Exception as e:
                st.warning(f"Could not generate SWOT analysis: {str(e)}")
                swot = "Unable to generate SWOT analysis at this time."
                
            try:
                mvp = mvp_chain.run(idea)
            except Exception as e:
                st.warning(f"Could not generate MVP scope: {str(e)}")
                mvp = "Unable to generate MVP scope at this time."
                
            try:
                landing_copy = landing_chain.run(idea)
            except Exception as e:
                st.warning(f"Could not generate landing page copy: {str(e)}")
                landing_copy = "Unable to generate landing page copy at this time."

            # Add to history
            session_id = str(uuid4())[:8]
            st.session_state.idea_history.append({
                "id": session_id,
                "input": user_input,
                "idea": idea,
                "pitch": pitch
            })

            # Display results
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

            # Feedback
            st.markdown("### \U0001F44D Was this idea helpful?")
            col1, col2 = st.columns(2)
            if col1.button("\U0001F44D Yes, I like it"):
                st.success("Awesome! Glad it helped.")
            if col2.button("\U0001F44E Needs improvement"):
                try:
                    refined = idea_chain.run(user_input + " but make it more innovative and unique")
                    st.markdown("### \U0001F501 Refined Idea")
                    st.markdown(refined)
                except Exception as e:
                    st.error(f"Could not refine idea: {str(e)}")

            # Download as PDF
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
                except Exception as e:
                    st.error(f"Could not generate PDF: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred during idea generation: {str(e)}")
            st.markdown("Please try again with a different input or refresh the page.")

# Session history
if st.session_state.idea_history:
    with st.expander("\U0001F552 Session History"):
        for entry in reversed(st.session_state.idea_history):
            st.markdown(f"**{entry['id']}** — _{entry['input']}_")
            st.markdown(entry['idea'])
            st.markdown("---")

st.markdown("---")
st.markdown("Made with \u2764\ufe0f using GPT-4 + LangChain")
