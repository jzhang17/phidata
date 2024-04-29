import streamlit as st
from phi.tools.tavily import TavilyTools
from assistant import get_research_assistant, get_planning_assistant, get_dp_assistant, get_suggestion_assistant
import markdown
from streamlit.components.v1 import html
from streamlit_pills import pills
import os
import re

os.environ["TAVILY_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"

st.set_page_config(
    page_title="JZ NewBizBot",
    page_icon="💰",
)
st.title("JZ NewBizBot")

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
    
    # Initialize loop control in session state
    if "loop_count" not in st.session_state:
        st.session_state["loop_count"] = 0
        st.session_state["topic"] = {}

    # Get topic for report
    input_topic = st.text_input(
        "Enter the name of a prospect or intermediary, can be a person, company or non-profit",
        value="Bill Gates",
    )
    # Button to generate report
    generate_report = st.button("Generate Report")



    # Generate multiple reports based on the loop_count
    for i in range(st.session_state["loop_count"]):
        if generate_report:
            current_count = st.session_state["loop_count"]
            st.session_state["topic"][current_count] = input_topic
            st.session_state["loop_count"] += 1  # Increase the loop counter to generate new report container
            
        report_topic = st.session_state["topic"][i]
        research_assistant = get_research_assistant(model=llm_model)
        planning_assistant = get_planning_assistant(model=llm_model)
        dp_assistant = get_dp_assistant(model=llm_model)
        suggestion_assistant = get_suggestion_assistant(model=llm_model)

        tavily_search_results = None

        with st.status(f"🔍 {report_topic} - Searching Web", expanded=True) as status:
            with st.container():

                search_planning = ""
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
            status.update(label= f"🔍 {report_topic} - Web Search Results", state="complete", expanded=False)
        
        if not tavily_search_results:
            st.write("Sorry report generation failed. Please try again.")
            return

        with st.status(f"📝 {report_topic} - Generating Report", expanded=True) as status:
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
                spacing
                
                suggested_options = ""
                for delta in suggestion_assistant.run(dp_report):
                    suggested_options += delta  # type: ignore
                    suggested_result = re.search(r"\[(.*?)\]", suggested_options, re.DOTALL)
                    if suggested_result:
                        # Split the found string by commas to get individual items
                        options = suggested_result.group(1).strip().split(",\n")
                        # Strip extra spaces and quotes
                        options = [item.strip().strip('"') for item in options]
                
                st.markdown("#### Generate another report:")
                # Calculate the number of options and how to distribute them across three columns
                # Determine the number of columns
                num_columns = 4

                # Calculate the number of options per column
                num_options = len(options)
                options_per_col = (num_options + num_columns - 1) // num_columns

                # Create columns
                cols = st.columns(4)

                # A dictionary to hold the state of each button
                button_states = {}

                # Place each button in the appropriate column
                for index, option in enumerate(options):
                    col_index = index // options_per_col
                    with cols[col_index]:
                        # Create a button and store its state in the dictionary
                        button_states[option] = st.button(option, key=f"button_{option}", use_container_width=True)

                # Handle button actions
                for option, was_clicked in button_states.items():
                    if was_clicked:
                        st.session_state["loop_count"] += 1  # Trigger to generate another report with the new topic
                        new_count = st.session_state['loop_count']
                        st.session_state["topic"][new_count] = option
                        st.write(f"You clicked: {option}, setting new topic for the next report generation.")
            status.update(label= f"📝 {report_topic} - Report Finished", state="complete", expanded=True)


main()
