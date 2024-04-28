import streamlit as st
from phi.tools.tavily import TavilyTools
from assistant import get_research_assistant, get_planning_assistant, get_dp_assistant
import markdown
from streamlit.components.v1 import html
import os
import re



st.set_page_config(
    page_title="JZ NewBizBot",
    page_icon=":dollar_banknote:",
)
st.title("JZ NewBizBot")
st.markdown("##### Dossier Builder")

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

    # Get topic for report
    input_topic = st.text_input(
        "Enter the name of a prospect or intermediary, can be a person, company or non-profit",
        value="Bill Gates",
    )
    # Button to generate report
    generate_report = st.button("Generate Report")
    if generate_report:
        st.session_state["topic"] = input_topic

    if "topic" in st.session_state:
        report_topic = st.session_state["topic"]
        research_assistant = get_research_assistant(model=llm_model)
        planning_assistant = get_planning_assistant(model=llm_model)
        dp_assistant = get_dp_assistant(model=llm_model)

        tavily_search_results = None

        with st.status(f"{report_topic} - Searching Web", expanded=True) as status:
            with st.container():

                search_planning = ""
                search_planning_container = st.empty()
                for delta in planning_assistant.run(report_topic):
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
            status.update(label= f"{report_topic} - Web Search Results", state="complete", expanded=False)
        
        if not tavily_search_results:
            st.write("Sorry report generation failed. Please try again.")
            return

        with st.status(f"{report_topic} - Generating Report", expanded=True) as status:
            with st.container():
                final_report = ""
                final_report_container = st.empty()
                for delta in research_assistant.run(tavily_search_results):
                    final_report += delta  # type: ignore
                    final_report_container.markdown(final_report)
                        
                spacing = "\n---\n"  # Adjust the number of new lines or use a horizontal rule for separation

                dp_report = ""
                for delta in dp_assistant.run(final_report):
                    dp_report += delta  # type: ignore
                    final_report_container.markdown(final_report + spacing + dp_report)
            status.update(label= f"{report_topic} - Report Finished", state="complete", expanded=True)


main()
