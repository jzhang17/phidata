import streamlit as st
import os
import tempfile
import requests
import base64
from pdf2image import convert_from_bytes
from io import BytesIO
from PIL import Image
from anthropic import Anthropic
from datetime import datetime
import time

anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

def pdf_reader(pdf_url):
    '''
    Extracts text from a PDF at the given URL and returns it in Markdown format. Input a valid PDF URL to receive the full text content of the document formatted as Markdown. Always use this tool for form 990 Filings.
    '''
    custom_headers = None
    content_placeholder = st.empty()
    full_text = "# PDF Content\n\n"
    error_occurred = False

    try:
        # Set up headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        if custom_headers:
            headers.update(custom_headers)

        # Download PDF with redirect handling
        session = requests.Session()
        response = session.get(pdf_url, headers=headers, allow_redirects=True)
        response.raise_for_status()
        pdf_content = response.content
        
        # Add a sleep after getting the PDF
        time.sleep(2)
        
        # Convert PDF to images
        images = convert_from_bytes(pdf_content, dpi=72)
        
        # Convert images to text
        client = Anthropic(api_key=anthropic_api_key)
        
        content_placeholder.markdown(full_text)
        
        for i, image in enumerate(images):
            try:
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                stream = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0,
                    stream=True,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Transcribe the content of this image in detail, providing a comprehensive textual representation of what you see. Format your response in Markdown."
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_base64
                                    }
                                }
                            ]
                        }
                    ]
                )
                
                full_text += f"## Page {i+1}\n\n"
                content_placeholder.markdown(full_text)
                
                for chunk in stream:
                    if hasattr(chunk, 'delta'):
                        if hasattr(chunk.delta, 'text'):
                            full_text += chunk.delta.text
                            content_placeholder.markdown(full_text)
                        elif hasattr(chunk.delta, 'content'):
                            for content in chunk.delta.content:
                                if hasattr(content, 'text'):
                                    full_text += content.text
                                    content_placeholder.markdown(full_text)
                
                full_text += "\n\n"
                content_placeholder.markdown(full_text)
            except Exception as e:
                error_occurred = True
                full_text += f"Error processing page {i+1}: {str(e)}\n\n"
                content_placeholder.markdown(full_text)
    
    except Exception as e:
        error_occurred = True
        full_text += f"An error occurred while processing the PDF: {str(e)}\n\n"
        content_placeholder.markdown(full_text)
    
    finally:
        if error_occurred:
            full_text += "\nNote: Some errors occurred during processing. The content may be incomplete.\n"
        else:
            full_text += "\nPDF processing completed successfully.\n"
        content_placeholder.markdown(full_text)
        return full_text

# Streamlit app
st.title("PDF Reader App")

# Input field for PDF URL
pdf_url = st.text_input("Enter the URL of the PDF:")

if st.button("Process PDF"):
    if pdf_url:
        with st.spinner("Processing PDF..."):
            # Create a placeholder for the streaming output
            result = pdf_reader(pdf_url)
    else:
        st.error("Please enter a valid PDF URL.")

# Add some information about the app
st.sidebar.header("About")
st.sidebar.info(
    "This app extracts text from a PDF at the given URL and returns it in Markdown format. "
    "Enter a valid PDF URL to receive the full text content of the document formatted as Markdown."
)

# Add a footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with Streamlit and Anthropic's Claude API")
