import streamlit as st
from phi.tools.tavily import TavilyTools
from assistant import get_research_assistant, get_planning_assistant, get_dp_assistant, get_followup_assistant
import markdown
from streamlit.components.v1 import html
from streamlit_pills import pills
import os
import re
import time

os.environ["GROQ_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

st.set_page_config(
    page_title="JZ NewBizBot #2",
    page_icon="💰",
)
st.title("JZ NewBizBot #2")

def main() -> None:
    # Get model
    llm_model = "llama3-70b-8192"

    # Set assistant_type in session state
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        st.rerun()

    query_params = st.query_params

    # Default to "Bill Gates" if 'input' is not provided
    input_value = query_params.get('input', 'Bill Gates')

    # Get topic for report
    input_topic = st.text_input(
        "Enter the name of a prospect or intermediary, can be a person, company or non-profit",
        value=input_value,
    )
    # Button to generate report
    generate_report = st.button("Generate Report")
    if generate_report:
        st.session_state["topic"] = input_topic

    if "topic" in st.session_state:
        report_topic = st.session_state["topic"]
        research_assistant = get_research_assistant(model=llm_model)
        planning_assistant = get_planning_assistant(model=llm_model)
        followup_assistant = get_followup_assistant(model=llm_model)
        dp_assistant = get_dp_assistant(model=llm_model)

        tavily_search_results = None
        spacing = "\n---\n"  # Adjust the number of new lines or use a horizontal rule for separation

        with st.status(f"🔍 {report_topic} - Initial Search", expanded=True) as status:
            with st.container():
                tavily_container = st.empty()
                tavily_search_results1 = ""
                tavily_search_results1 += TavilyTools().web_search_using_tavily(report_topic)
                if tavily_search_results1:
                    tavily_container.markdown(tavily_search_results1)
            status.update(label= f"🔍 {report_topic} - Initial Search Results", state="complete", expanded=False)

        with st.status(f"📝 {report_topic} - Generating First Draft", expanded=True) as status:
            with st.container():
                first_report = ""
                first_report_container = st.empty()
                for delta in research_assistant.run(tavily_search_results1):
                    first_report += delta  # type: ignore
                    first_report_container.markdown(first_report)
            status.update(label= f"📝 {report_topic} - First Draft Finished", state="complete", expanded=True)

        with st.status(f"🔍 {report_topic} - Follow-up Search", expanded=True) as status:
            with st.container():
                search_planning = ""
                for delta in followup_assistant.run(first_report):
                    search_planning += delta  # type: ignore
                    matches = re.search(r"\[(.*?)\]", search_planning, re.DOTALL)
                    if matches:
                        # Split the found string by commas to get individual items
                        search_list = matches.group(1).strip().split(",\n")
                        # Strip extra spaces and quotes
                        search_list = [item.strip().strip('"') for item in search_list]

                tavily_container = st.empty()
                tavily_search_results = ""
                for search_queries in search_list:
                    search_display = ""
                    search_display += f"**Searching for:** {search_queries}"
                    tavily_container.markdown(search_display)
                    tavily_container = st.empty()

                    tavily_search_results += TavilyTools().web_search_using_tavily(search_queries)
                if tavily_search_results:
                    tavily_container.markdown(tavily_search_results)
            status.update(label= f"🔍 {report_topic} - Follow-up Search Results", state="complete", expanded=False)
        
        if not tavily_search_results:
            st.write("Sorry report generation failed. Please try again.")
            return
        


        with st.status(f"📝 {report_topic} - Generating Follow-up Report", expanded=True) as status:
            with st.container():
                final_report = ""
                final_report_container = st.empty()
                for delta in research_assistant.run(tavily_search_results):
                    final_report += delta  # type: ignore
                    final_report_container.markdown(final_report)

                time.sleep(5)

                dp_report = ""
                for delta in dp_assistant.run(first_report + final_report):
                    dp_report += delta  # type: ignore
                    final_report_container.markdown(final_report + spacing + dp_report)                
            status.update(label= f"📝 {report_topic} - Follow-up Report", state="complete", expanded=True)


main()
