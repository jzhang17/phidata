import streamlit as st
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
import os

# Initialize ChatAnthropic
chat_model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

# Create the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """Extract the following information from each line of the given text:
    - First name
    - Last name
    - Employer
    - City
    - State

    Text:
    {input_text}

    For each line, provide the output in the following format:
    First name: [first name]
    Last name: [last name]
    Employer: [employer]
    City: [city]
    State: [state]

    Separate each person's information with a blank line.
    """
)

def parse_output(output):
    entries = output.strip().split('\n\n')
    parsed_data = []
    for entry in entries:
        person = {}
        for line in entry.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                person[key.strip().lower().replace(' ', '_')] = value.strip()
        if person:
            parsed_data.append(person)
    return parsed_data

# Streamlit app
st.title("Information Extractor")

# Text input
input_text = st.text_area("Enter the text to extract information from (one entry per line):")

if st.button("Extract Information"):
    if input_text:
        # Format the prompt
        prompt = prompt_template.format_messages(input_text=input_text)

        # Get the response from Claude
        response = chat_model.invoke(prompt)

        # Parse the response
        parsed_output = parse_output(response.content)

        if not parsed_output:
            st.error("Failed to extract information. Please try again with different input.")
            st.stop()

        # Create a DataFrame
        df = pd.DataFrame(parsed_output)

        # Ensure all expected columns are present
        expected_columns = ['first_name', 'last_name', 'employer', 'city', 'state']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ''

        # Reorder columns
        df = df[expected_columns]

        # Remove any completely empty rows
        df = df.dropna(how='all').reset_index(drop=True)

        # Display the extracted information
        st.subheader("Extracted Information:")
        st.write(df)

        # Save to Excel
        excel_file = "extracted_information.xlsx"
        df.to_excel(excel_file, index=False, engine='openpyxl')
        st.success(f"Information saved to {excel_file}")

        # Provide download link
        with open(excel_file, "rb") as file:
            st.download_button(
                label="Download Excel File",
                data=file,
                file_name=excel_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("Please enter some text to extract information from.")