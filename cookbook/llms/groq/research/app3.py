import streamlit as st
from phi.tools.tavily import TavilyTools
from assistant import get_research_assistant, get_planning_assistant, get_dp_assistant, get_followup_assistant, get_consolidate_assistant
import markdown
from streamlit.components.v1 import html
from streamlit_pills import pills
import os
import re
import time
import dropbox
from tenacity import retry, stop_after_attempt, wait_exponential


st.set_page_config(
    page_title="JZ NewBizBot #3",
    page_icon="💰",
)
st.title("JZ NewBizBot v3")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))  # Retry configuration
def search_with_retry(input):
    return TavilyTools().web_search_using_tavily(input)

def upload_file_to_dropbox(file_path, dropbox_path):
    """Upload a file to Dropbox, checking for duplicates and modifying the path if necessary."""
    # Access token from your Dropbox app
    access_token = os.getenv('DROPBOX_ACCESS_TOKEN')

    # Create a Dropbox client
    dbx = dropbox.Dropbox(access_token)

    # Extract the file name from the file path
    file_name = os.path.basename(file_path)
    folder_path = os.path.dirname(dropbox_path)

    try:
        # Define a method to generate a new file name with a number appended
        def generate_new_filename(index):
            base_name, extension = os.path.splitext(file_name)
            return f"{base_name}_{index}{extension}"

        # Start checking for existing files and generate new file name if necessary
        index = 1
        new_file_name = file_name
        while True:
            search_result = dbx.files_search(folder_path, new_file_name)
            if not search_result.matches:
                break
            new_file_name = generate_new_filename(index)
            index += 1

        # Update the Dropbox path with the potentially new file name
        dropbox_path = os.path.join(folder_path, new_file_name)

        with open(file_path, "rb") as file:
            # Read the contents of the file
            file_contents = file.read()
            # Upload the file to Dropbox
            dbx.files_upload(file_contents, dropbox_path, mode=dropbox.files.WriteMode.add)
            print("File uploaded successfully")

    except Exception as e:
        print(f"Error: {e}")

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
        followup_assistant = get_followup_assistant(model="mixtral-8x7b-32768")
        dp_assistant = get_dp_assistant(model=llm_model)
        consolidate_assistant = get_consolidate_assistant(model="mixtral-8x7b-32768")

        tavily_search_results = None
        spacing = "\n---\n"  # Adjust the number of new lines or use a horizontal rule for separation

        with st.status(f"🔍 {report_topic} - Initial Search", expanded=True) as status:
            with st.container():
                tavily_container = st.empty()
                tavily_search_results1 = ""
                try:
                    tavily_search_results1 += search_with_retry(report_topic)
                except Exception as e:  # Catch any exceptions during retries
                    print(f"Error after retries: {e}") 
                
                if tavily_search_results1:
                    tavily_container.markdown(tavily_search_results1)
                    file_path = f'/tmp/{report_topic}_first_search.txt'
                    with open(file_path, 'w') as file:
                        file.write(tavily_search_results1)
                    with open(file_path, 'rb') as file:
                        dropbox_path = f'/Apps/NewBizBot/{report_topic}_first_search.txt'
                        upload_file_to_dropbox(file_path, dropbox_path)
            status.update(label= f"🔍 {report_topic} - Initial Search Results", state="complete", expanded=False)



        with st.status(f"📝 {report_topic} - Generating First Draft", expanded=True) as first_draft_status:
            with st.container():
                first_report = ""
                first_report_container = st.empty()
                for delta in research_assistant.run(tavily_search_results1):
                    first_report += delta  # type: ignore
                    first_report_container.markdown(first_report)
                file_path = f'/tmp/{report_topic}_first_report.txt'
                with open(file_path, 'w') as file:
                    file.write(first_report)
                with open(file_path, 'rb') as file:
                    dropbox_path = f'/Apps/NewBizBot/{report_topic}_first_report.txt'
                    upload_file_to_dropbox(file_path, dropbox_path)
            first_draft_status.update(label= f"📝 {report_topic} - First Draft Finished", state="complete", expanded=True)

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
                tavily_search_results2 = ""
                for search_queries in search_list:
                    search_display = ""
                    search_display += f"**Searching for:** {search_queries}"
                    tavily_container.markdown(search_display)
                    tavily_container = st.empty()
                    try:
                        tavily_search_results2 += search_with_retry(search_queries)
                    except Exception as e:  # Catch any exceptions during retries
                        print(f"Error after retries: {e}") 
                if tavily_search_results2:
                    tavily_container.markdown(tavily_search_results2)
                    file_path = f'/tmp/{report_topic}_followup_search_result.txt'
                    with open(file_path, 'w') as file:
                        file.write(tavily_search_results2)
                    with open(file_path, 'rb') as file:
                        dropbox_path = f'/Apps/NewBizBot/{report_topic}_followup_search_result.txt'
                        upload_file_to_dropbox(file_path, dropbox_path)
            status.update(label= f"🔍 {report_topic} - Follow-up Search Results", state="complete", expanded=False)
        
        if not tavily_search_results2:
            st.write("Sorry report generation failed. Please try again.")
            return
        


        with st.status(f"📝 {report_topic} - Generating Follow-up Report", expanded=True) as followup_status:
            with st.container():
                followup_report = ""
                followup_report_container = st.empty()
                for delta in research_assistant.run(tavily_search_results2):
                    followup_report += delta  # type: ignore
                    followup_report_container.markdown(followup_report)
                file_path = f'/tmp/{report_topic}_followup_report.txt'
                with open(file_path, 'w') as file:
                    file.write(followup_report)
                with open(file_path, 'rb') as file:
                    dropbox_path = f'/Apps/NewBizBot/{report_topic}_followup_report.txt'
                    upload_file_to_dropbox(file_path, dropbox_path)

         
            followup_status.update(label= f"📝 {report_topic} - Follow-up Report", state="complete", expanded=True)

        with st.status(f"📝 {report_topic} - Generating Consolidated Report", expanded=True) as status:
            with st.container():
                consolidate_report_container = st.empty()
                consolidate_report = ""
                for delta in consolidate_assistant.run(first_report + spacing + followup_report):
                    consolidate_report += delta  # type: ignore
                    consolidate_report_container.markdown(consolidate_report)

                dp_report = ""
                for delta in dp_assistant.run(first_report + followup_report):
                    dp_report += delta  # type: ignore
                    consolidate_report_container.markdown(consolidate_report + spacing + dp_report)

                file_path = f'/tmp/{report_topic}_final_report.txt'
                with open(file_path, 'w') as file:
                    file.write(consolidate_report + spacing + dp_report)
                with open(file_path, 'rb') as file:
                    dropbox_path = f'/Apps/NewBizBot/{report_topic}_final_report.txt'
                    upload_file_to_dropbox(file_path, dropbox_path)

                       
            status.update(label= f"📝 {report_topic} - Consolidated Report", state="complete", expanded=True)

        first_draft_status.update(expanded=False)
        followup_status.update(expanded=False)





main()
