import os
import numpy as np
import matplotlib.pyplot as plt
from langchain.tools import StructuredTool
from langchain_community.llms import MistralAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
import streamlit as st

# --- Set up Streamlit UI ---
st.set_page_config(
    layout="wide",
    page_title="F1 Telemetry Agent",
)

# Custom CSS for Crimson Paddock theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0D0E10;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #FF1801;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #1E2024;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Define Tools ---
def get_track_metadata(track_name: str, year: int = None):
    """
    Fetches track metadata (e.g., lap records, location, layout).
    """
    mock_data = {
        "Spa": {"location": "Belgium", "lap_record": "1:46.286", "year": 2024},
        "Monza": {"location": "Italy", "lap_record": "1:21.046", "year": 2024},
        "Monaco": {"location": "Monaco", "lap_record": "1:12.909", "year": 2024},
    }
    return mock_data.get(track_name, "Track not found.")

def generate_track_layout_map(track_name: str):
    """
    Generates a 2D track layout map using Matplotlib.
    """
    track_coords = {
        "Spa": np.array([[0, 0], [1, 2], [3, 1], [2, -1], [0, 0]]),
        "Monza": np.array([[0, 0], [1, 1], [2, 0], [1, -1], [0, 0]]),
        "Monaco": np.array([[0, 0], [0.5, 1], [1, 0.5], [0.5, 0], [0, 0]]),
    }

    coords = track_coords.get(track_name)
    if coords is None:
        return "Track not found."

    plt.figure(facecolor='#0D0E10')
    plt.plot(coords[:, 0], coords[:, 1], color='cyan', linewidth=2)
    plt.title(f"{track_name} Track Layout", color='white')
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(f"{track_name}_layout.png", facecolor='#0D0E10')
    return f"{track_name}_layout.png"

# Wrap tools for LangChain
tools = [
    StructuredTool.from_function(
        func=get_track_metadata,
        name="get_track_metadata",
        description="Fetches track metadata (location, lap records, etc.)."
    ),
    StructuredTool.from_function(
        func=generate_track_layout_map,
        name="generate_track_layout_map",
        description="Generates a 2D track layout map."
    )
]

# --- Initialize Mistral Large Model ---
mistral = MistralAI(model="mistral-large-latest", api_key=os.environ["MISTRAL_API_KEY"])

# --- Create LangChain Agent ---
prompt = PromptTemplate.from_template(
    """
    You are a seasoned F1 telemetry analyst.
    Answer the user's query using the tools below.
    If the query is broad (e.g., "Tell me about Spa"), use `get_track_metadata`.
    If the query is about track layouts, use `generate_track_layout_map`.

    Tools: {tools}
    """
)

agent = create_react_agent(
    llm=mistral,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# --- Streamlit UI ---
st.title("🏎️ F1 Telemetry Agent")
query = st.text_input("Ask about a track (e.g., 'Spa', 'Monza', 'Monaco'):")

if query:
    with st.spinner("Processing..."):
        response = agent_executor.invoke({"input": query})
        st.write(response["output"])
        if "layout.png" in response["output"]:
            st.image(response["output"], use_column_width=True)