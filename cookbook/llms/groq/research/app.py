import streamlit as st
from phi.tools.tavily import TavilyTools
from assistant import get_research_assistant  # type: ignore
import markdown
from streamlit.components.v1 import html

st.set_page_config(
    page_title="JZ NewBizBot",
    page_icon=":dollar_banknote:",
)
st.title("JZ NewBizBot")
st.markdown("##### Dossier Builder")

def copy_to_clipboard(report):
    # Custom HTML and CSS for the clipboard copy button
    button_html = f"""
    <html>
    <body>
    <style>
        .copy-button {{
            background-color: #FFFFFF; /* White background */
            border: 1px solid #D1D1D1; /* Light gray border */
            color: #000000; /* Black text */
            padding: 5px 2px; /* Padding to match the button size */
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            line-height: 20px; /* Line height to vertically center the text */
            border-radius: 4px; /* Rounded corners */
            width: 135px; /* Width of the button */
            height: 38px; /* Height of the button */
            box-sizing: border-box; /* Includes padding and border in the width and height */
            cursor: pointer; /* Changes the cursor to a pointer when hovering over the button */
            outline: none; /* Removes the outline */
        }}
        .copy-button:hover {{
            background-color: #F8F8F8; /* Slightly darker white when hovering over the button */
        }}
        .copy-button:active {{
            background-color: #EAEAEA; /* Even darker white when pressing the button */
        }}
    </style>
    <!-- Button and container for the text to be copied -->
    <button id="copyButton" class="copy-button" title="Click to copy the report to your clipboard">
        Copy to Clipboard
    </button>
    <div class="output" style="display:none;">{report}</div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const copyButton = document.getElementById('copyButton');
            const output = document.querySelector('.output');

            copyButton.addEventListener('click', function() {{
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = output.innerHTML; // Preserve formatting including HTML structure
                tempDiv.style.position = 'absolute';
                tempDiv.style.left = '-9999px';
                tempDiv.style.whiteSpace = 'pre'; // Maintain whitespace
                document.body.appendChild(tempDiv);

                const range = document.createRange();
                range.selectNodeContents(tempDiv);
                const sel = window.getSelection();
                sel.removeAllRanges(); 
                sel.addRange(range);

                try {{
                    document.execCommand('copy');
                    alert('Output copied successfully.');
                }} catch (err) {{
                    console.error('Unable to copy', err);
                    alert('Oops, unable to copy!');
                }} finally {{
                    document.body.removeChild(tempDiv); // Cleanup
                }}
            }});
        }});
    </script>
    </body>
    </html>
    """
    html(button_html)






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
        value="Bill Gate",
    )
    # Button to generate report
    generate_report = st.button("Generate Report")
    if generate_report:
        st.session_state["topic"] = input_topic

    if "topic" in st.session_state:
        report_topic = st.session_state["topic"]
        research_assistant = get_research_assistant(model=llm_model)
        tavily_search_results = None

        with st.status("Searching Web", expanded=True) as status:
            with st.container():
                tavily_container = st.empty()
                tavily_search_results = TavilyTools().web_search_using_tavily(report_topic)
                if tavily_search_results:
                    tavily_container.markdown(tavily_search_results)
            status.update(label="Web Search Complete", state="complete", expanded=False)

        if not tavily_search_results:
            st.write("Sorry report generation failed. Please try again.")
            return

        with st.spinner("Generating Report"):
            final_report = ""
            final_report_container = st.empty()
            for delta in research_assistant.run(tavily_search_results):
                final_report += delta  # type: ignore
                final_report_container.markdown(final_report)
            copy_to_clipboard(markdown.markdown(final_report, extensions=['mdx_truly_sane_lists'], tab_length=2))

main()
