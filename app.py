
import os
import streamlit as st
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent Research and Automation System")
st.write(
    "A supervisor coordinates research, analysis, writing, "
    "and review agents to prepare a research report."
)

with st.sidebar:
    st.header("Research team")
    st.write("🔎 Research Agent")
    st.write("📊 Analysis Agent")
    st.write("✍️ Writer Agent")
    st.write("✅ Reviewer Agent")
    st.caption("Powered by LangChain subagents")

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

if not api_key:
    st.warning(
        "The OpenAI API key has not been configured yet. "
        "Add it in Streamlit Secrets after deploying."
    )
    st.stop()

model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0,
)

search_tool = DuckDuckGoSearchRun()

@tool
def research_agent(task: str) -> str:
    """Research a topic using web search and summarize useful findings."""
    search_agent = create_agent(
        model,
        tools=[search_tool],
        system_prompt=(
            "You are a research specialist. Search the web for the "
            "requested topic. Summarize relevant findings and include "
            "source URLs returned by the search tool. Do not invent sources."
        ),
    )
    result = search_agent.invoke({
        "messages": [{"role": "user", "content": task}]
    })
    return str(result["messages"][-1].content)

@tool
def analysis_agent(task: str) -> str:
    """Analyze research findings and identify insights and limitations."""
    agent = create_agent(
        model,
        tools=[],
        system_prompt=(
            "You are an analytical specialist. Identify key findings, "
            "patterns, risks, limitations, and practical recommendations. "
            "Distinguish evidence from assumptions."
        ),
    )
    result = agent.invoke({
        "messages": [{"role": "user", "content": task}]
    })
    return str(result["messages"][-1].content)

@tool
def writer_agent(task: str) -> str:
    """Turn research and analysis into a clear, structured report."""
    agent = create_agent(
        model,
        tools=[],
        system_prompt=(
            "You are a professional report writer. Create a clear report "
            "with a title, executive summary, findings, recommendations, "
            "and a conclusion. Preserve source URLs supplied to you."
        ),
    )
    result = agent.invoke({
        "messages": [{"role": "user", "content": task}]
    })
    return str(result["messages"][-1].content)

@tool
def reviewer_agent(report: str) -> str:
    """Review a report for clarity, unsupported claims, and missing details."""
    agent = create_agent(
        model,
        tools=[],
        system_prompt=(
            "You are a careful report reviewer. Check clarity, consistency, "
            "unsupported claims, missing context, and whether sources are "
            "provided. Suggest corrections. Never claim you verified a source "
            "unless you actually did."
        ),
    )
    result = agent.invoke({
        "messages": [{"role": "user", "content": report}]
    })
    return str(result["messages"][-1].content)

supervisor = create_agent(
    model,
    tools=[
        research_agent,
        analysis_agent,
        writer_agent,
        reviewer_agent,
    ],
    system_prompt=(
        "You are the supervisor of a multi-agent research team. "
        "For every research request, delegate to the research_agent first. "
        "Pass its findings to analysis_agent. Use writer_agent to draft "
        "the report, then reviewer_agent to review it. Revise the report "
        "if needed. Return the final report and preserve source URLs. "
        "Do not claim a tool was used unless you called it."
    ),
)

topic = st.text_area(
    "What should the AI team research?",
    placeholder=(
        "Example: Research how small businesses use AI, "
        "including benefits, risks, and practical recommendations."
    ),
    height=120,
)

if st.button("🚀 Start Research", type="primary"):
    if not topic.strip():
        st.error("Please enter a research topic first.")
    else:
        try:
            with st.spinner(
                "The supervisor is coordinating the research team..."
            ):
                result = supervisor.invoke({
                    "messages": [{"role": "user", "content": topic}]
                })

            messages = result.get("messages", [])
            final_answer = (
                messages[-1].content if messages else "No report was returned."
            )

            st.session_state["research_report"] = str(final_answer)
            st.success("Research process completed!")

        except Exception as exc:
            st.error(
                "The research process failed. Check your API key, "
                "dependencies, and app logs."
            )
            st.exception(exc)

if "research_report" in st.session_state:
    st.divider()
    st.subheader("📄 Final Research Report")
    st.markdown(st.session_state["research_report"])
    st.download_button(
        "Download report",
        data=st.session_state["research_report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
