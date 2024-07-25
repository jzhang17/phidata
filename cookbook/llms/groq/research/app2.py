import streamlit as st
from streamlit.components.v1 import html
import os
import boto3
from botocore.client import Config
from pathlib import Path
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, SQLDatabase
from langchain_core.runnables import RunnableConfig
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import OpenAI
from typing import Annotated, List, Tuple, Union
from sqlalchemy import create_engine
from capturing_callback_handler import playback_callbacks
from clear_results import with_clear_container
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from typing import Callable, TypeVar
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
import inspect
from langchain_community.document_loaders import PyPDFLoader
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from crewai import Crew, Process, Agent, Task
from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, Optional
from langchain_openai import ChatOpenAI
import re
import sys
from assistant import get_dp_assistant
import json  # Make sure to import the json module
from phi.tools.tavily import TavilyTools
from typing import Optional, Literal, Dict, Any
from phi.tools import Toolkit
from phi.utils.log import logger
try:
    from tavily import TavilyClient
except ImportError:
    raise ImportError("`tavily-python` not installed. Please install using `pip install tavily-python`")
from os import getenv
from langchain_anthropic import ChatAnthropic
from PyPDF2 import PdfFileReader
import base64
from pdf2image import convert_from_bytes
from io import BytesIO
from PIL import Image
from anthropic import Anthropic
import time
import fitz
import tiktoken
import random

os.environ['PATH'] += os.pathsep
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY') # replace with your API key from anthropic website or use environment variable if available in the codebase


st.set_page_config(
    page_title="JZ NewBizBot",
    page_icon="💰"
    )

st.title("JZ NewBizBot XL")
# Add a dropdown to select the model
model_option = 'Claude-3.5-Sonnet'

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

class TavilyTools(Toolkit):
    def __init__(
        self,
        api_key: Optional[str] = None,
        search: bool = True,
        max_tokens: int = 5000,
        include_answer: bool = True,
        search_depth: Literal["basic", "advanced"] = "advanced",
        format: Literal["json", "markdown"] = "markdown",
        use_search_context: bool = False,
    ):
        super().__init__(name="tavily_tools")

        self.api_key = api_key or getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.error("TAVILY_API_KEY not provided")

        self.client: TavilyClient = TavilyClient(api_key=self.api_key)
        self.search_depth: Literal["basic", "advanced"] = search_depth
        self.max_tokens: int = max_tokens
        self.include_answer: bool = include_answer
        self.format: Literal["json", "markdown"] = format

        if search:
            if use_search_context:
                self.register(self.web_search_with_tavily)
            else:
                self.register(self.web_search_using_tavily)

    def web_search_using_tavily(self, query: str, max_results: int = 7) -> str:
        """Use this function to search the web for a given query.
        This function uses the Tavily API to provide realtime online information about the query.

        Args:
            query (str): Query to search for.
            max_results (int): Maximum number of results to return. Defaults to 5.

        Returns:
            str: JSON string of results related to the query.
        """

        response = self.client.search(
            query=query, search_depth=self.search_depth, include_answer=self.include_answer, max_results=max_results
        )

        clean_response: Dict[str, Any] = {"query": query}
        if "answer" in response:
            clean_response["answer"] = response["answer"]

        clean_results = []
        current_token_count = len(json.dumps(clean_response))
        for result in response.get("results", []):
            _result = {
                "title": result["title"],
                "url": result["url"],
                "content": result["content"],
                "score": result["score"],
            }
            current_token_count += len(json.dumps(_result))
            if current_token_count > self.max_tokens:
                break
            clean_results.append(_result)
        clean_response["results"] = clean_results

        if self.format == "json":
            return json.dumps(clean_response) if clean_response else "No results found."
        elif self.format == "markdown":
            _markdown = ""
            _markdown += f"#### {query}\n\n"
            if "answer" in clean_response:
                _markdown += "#### Summary\n"
                _markdown += f"{clean_response.get('answer')}\n\n"
            for result in clean_response["results"]:
                _markdown += f"#### [{result['title']}]({result['url']})\n"
                _markdown += f"{result['content']}\n\n"
            _markdown = _markdown.replace("$","\$")
            pdf_links = []
            webpage_links = []
            for result in clean_response["results"]:
                url = result['url']
                if "https://www.linkedin.com/in" in url:
                    continue
                if "pdf" in url.lower():
                    pdf_links.append(url)
                else:
                    webpage_links.append(url)

            if pdf_links:
                _markdown += f"Links for load_pdf tool: {pdf_links}\n\n"
            if webpage_links:
                _markdown += f"Links for scrape_webpages tool: {webpage_links}\n"
            return _markdown

    def web_search_with_tavily(self, query: str) -> str:
        """Use this function to search the web for a given query.
        This function uses the Tavily API to provide realtime online information about the query.

        Args:
            query (str): Query to search for.

        Returns:
            str: JSON string of results related to the query.
        """

        return self.client.get_search_context(
            query=query, search_depth=self.search_depth, max_tokens=self.max_tokens, include_answer=self.include_answer
        )

def count_tokens_and_estimate_cost(input_text, output_text, model="GPT-4o"):
    if model == "GPT-4o":
        encoding = tiktoken.encoding_for_model("gpt-4")
    else:
        # For Claude or other models, use a generic tokenizer
        encoding = tiktoken.get_encoding("cl100k_base")
    
    input_token_count = len(encoding.encode(input_text))
    output_token_count = len(encoding.encode(output_text))
    
    # Cost estimates (update these based on current pricing)
    cost_per_1k_tokens = {
        "GPT-4o": {"input": 0.005, "output": 0.015},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015}
    }
    
    model_costs = cost_per_1k_tokens.get(model, cost_per_1k_tokens["GPT-4o"])
    input_cost = (input_token_count / 1000) * model_costs["input"]
    output_cost = (output_token_count / 1000) * model_costs["output"]
    
    return input_token_count, output_token_count, input_cost, output_cost

@tool
def tavily_tool(query):
    """A search engine optimized for comprehensive, accurate, and trusted results. 
    Useful for when you need to answer questions about current events. Input should be a search query.
    """
    tavily_search_results = TavilyTools().web_search_using_tavily(query)
    return tavily_search_results

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests to scrape the provided web pages for detailed information."""
    combined_content = ""
    
    # Function to resize images in markdown to max 300px wide using HTML
    def resize_images(content: str) -> str:
        # Regex to find markdown image syntax
        pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        
        def replacement(match):
            alt_text = match.group(1)
            image_url = match.group(2)
            return f'<img src="{image_url}" alt="{alt_text}" style="max-width:300px; width:100%;" />'
        
        return re.sub(pattern, replacement, content)
    
    for url in urls:
        response = requests.get("https://r.jina.ai/" + url)
        content = response.text.replace("$","\$")
        # Resize images in the current content
        resized_content = resize_images(content)
        combined_content += resized_content
        if len(combined_content) > 100000:
            break
            
    return combined_content[:100000]  # Limit the output to the first 100,000 characters
   

@tool
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
        time.sleep(2)

        # Download PDF with redirect handling
        session = requests.Session()
        response = session.get(pdf_url, headers=headers, allow_redirects=True)
        response.raise_for_status()
        pdf_content = response.content
        

        
        # Open the PDF using PyMuPDF
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Convert PDF pages to images
        client = Anthropic(api_key=anthropic_api_key)
        
        content_placeholder.markdown(full_text)
        
        for page_num in range(len(pdf_document)):
            try:
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
                img_bytes = pix.tobytes("png")
                image_base64 = base64.b64encode(img_bytes).decode()
                
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
                
                full_text += f"## Page {page_num+1}\n\n"
                content_placeholder.markdown(full_text)
                
                for chunk in stream:
                    if hasattr(chunk, 'delta'):
                        if hasattr(chunk.delta, 'text'):
                            chunk_text = chunk.delta.text.replace("$", "\$").replace("```", "\n")
                            full_text += chunk_text
                            content_placeholder.markdown(full_text)
                        elif hasattr(chunk.delta, 'content'):
                            for content in chunk.delta.content:
                                if hasattr(content, 'text'):
                                    content_text = content.text.replace("$", "\$").replace("```", "\n")
                                    full_text += content_text
                                    content_placeholder.markdown(full_text)
                
                full_text += "\n\n"
                content_placeholder.markdown(full_text)
            except Exception as e:
                error_occurred = True
                full_text += f"Error processing page {page_num+1}: {str(e)}\n\n"
                content_placeholder.markdown(full_text)

        pdf_document.close()

    except Exception as e:
        error_occurred = True
        full_text += f"An error occurred while processing the PDF: {str(e)}\n\n"
        content_placeholder.markdown(full_text)

    return full_text



@tool
def nonprofit_financials(nonprofit_name):
    """Get structured financial data and latest filing PDF content for nonprofits. This tool prefers EIN as input."""
    
    base_url = "https://projects.propublica.org/nonprofits/api/v2"
    
    # Step 1: Search for the nonprofit by name to get the EIN
    search_url = f"{base_url}/search.json"
    params = {'q': nonprofit_name}
    search_response = requests.get(search_url, params=params)
    
    if search_response.status_code != 200:
        return f"**Error searching for nonprofit:** {search_response.status_code}"
    
    search_results = search_response.json()
    if not search_results['organizations']:
        return f"**No nonprofit found with the name {nonprofit_name}**"
    
    # Get the EIN of the first matching organization
    ein = search_results['organizations'][0]['ein']
    
    # Step 2: Use the EIN to get the organization overview
    organization_url = f"{base_url}/organizations/{ein}.json"
    organization_response = requests.get(organization_url)
    
    if organization_response.status_code != 200:
        return f"**Error retrieving organization data:** {organization_response.status_code}"
    
    organization_data = organization_response.json().get('organization', {})
    
    # Find the most recent filing with data
    filings = organization_response.json().get('filings_with_data', [])
    
    # Check if there are filings
    latest_filing = filings[0] if filings else None
    
    def format_dollar(amount):
        return f"${amount:,.0f}" if amount is not None else "None"
    
    # Create an overview of the organization
    overview = {
        'organization': {
            'ID': organization_data.get('id'),
            'EIN': organization_data.get('ein'),
            'Name': organization_data.get('name'),
            'Address': organization_data.get('address'),
            'City': organization_data.get('city'),
            'State': organization_data.get('state'),
            'Zipcode': organization_data.get('zipcode'),
            'Tax Period': organization_data.get('tax_period'),
            'Total Assets': format_dollar(organization_data.get('asset_amount')),
            'Total Income': format_dollar(organization_data.get('income_amount')),
            'Total Revenue': format_dollar(organization_data.get('revenue_amount')),
            'Updated At': organization_data.get('updated_at'),
            'Data Source': organization_data.get('data_source')
        },
        'latest_filing': {
            'Tax Period': latest_filing.get('tax_prd') if latest_filing else None,
            'Tax Period Year': latest_filing.get('tax_prd_yr') if latest_filing else None,
            'PDF URL': latest_filing.get('pdf_url') if latest_filing else None,
            'Updated': latest_filing.get('updated') if latest_filing else None,
            'Total Revenue': format_dollar(latest_filing.get('totrevenue')) if latest_filing else None,
            'Total Functional Expenses': format_dollar(latest_filing.get('totfuncexpns')) if latest_filing else None,
            'Total Assets at End of Year': format_dollar(latest_filing.get('totassetsend')) if latest_filing else None,
            'Total Liabilities at End of Year': format_dollar(latest_filing.get('totliabend')) if latest_filing else None,
            'Compensation of Current Officers': format_dollar(latest_filing.get('compnsatncurrofcr')) if latest_filing else None,
            'Other Salaries and Wages': format_dollar(latest_filing.get('othrsalwages')) if latest_filing else None,
            'Total Net Assets at End of Year': format_dollar(latest_filing.get('totnetassetend')) if latest_filing else None,
            'Investment Income': format_dollar(latest_filing.get('invstmntinc')) if latest_filing else None
        }
    }
    
    # Generate markdown output
    markdown = f"#### Nonprofit Financials: {overview['organization']['Name']}\n\n"
    
    markdown += "#### Organization Details\n"
    for key, value in overview['organization'].items():
        markdown += f"- **{key.replace('_', ' ')}:** {value}\n"
    
    markdown += "\n#### Latest Filing Details\n"
    for key, value in overview['latest_filing'].items():
        markdown += f"- **{key.replace('_', ' ')}:** {value}\n"

    return markdown


Researcher = Agent(
    role='Researcher',
    backstory='''
        ### Prompt Instruction:
        You are an expert in wealth and investment management, specializing in developing comprehensive client profiles across various sectors. Your task is to create detailed financial profiles of potential clients without strategizing. Utilize your expertise to produce informative profiles that will aid in crafting personalized financial management plans later. Include hyperlinks to essential financial data sources like Bloomberg, Forbes, and specific financial databases for additional context.

        #### Context:
        Your service offerings include investment management, Outsourced Chief Investment Officer (OCIO) services, private banking, single-stock risk handling, and trust & estate planning. Leverage your expertise to provide analytical insights suitable for a diverse client base. Adopt a methodical and detail-oriented approach to ensure all pertinent financial details are covered comprehensively.
        ''',
    goal='''#### Objectives:
        1. **For an Individual**: Gather and document information about the individual’s employment history, age, personal net worth, diverse income sources, family circumstances, and involvement in boards or charities. Hyperlinks to relevant professional pages should be included for verification of employment history.

        2. **For a Nonprofit**: Compile the nonprofit’s asset details, highlight key board members/Tustees/Executives/Leadership, enumerate major donors, and review their financial transparency using links to platforms like [Cause IQ](https://www.causeiq.com/) and [ProPublica](https://www.propublica.org/) for access to recent Form 990s. Always use nonprofit_financials tool to get detailed and reliable information of the financial data after you have searched the web.

        3. **For a Company**: Create thorough profiles for top executives, pinpoint primary investors, record significant financial milestones, and evaluate the company's financial health using metrics like valuation, revenue, and profitability. Link to resources such as [Yahoo Finance](https://finance.yahoo.com/) or the company website for financial reports and analyses.

        ''',
    tools=[tavily_tool,scrape_webpages,pdf_reader,nonprofit_financials],  # This can be optionally specified; defaults to an empty list
    llm=llm,
    verbose=True
    )

Followup_Agent = Agent(
    role='Followup Agent',
    backstory='''### Expert Instruction:
You are an expert in wealth and investment management, specializing in developing comprehensive client profiles across various sectors. Your task is to read the current report, identify missing information, and perform further research to find out more information. Make sure to include sources and corresponding hyperlinks for any new data you add.

### Context:
Your service offerings include investment management, Outsourced Chief Investment Officer (OCIO) services, private banking, single-stock risk handling, and trust & estate planning. Leverage your expertise to provide analytical insights suitable for a diverse client base. Adopt a methodical and detail-oriented approach to ensure all pertinent financial details are covered comprehensively.
additional context.
                ''',
    goal='''
### Objectives:
1. **For an Individual**: Gather and add information about the individual's employment history, age, personal net worth, diverse income sources, family circumstances, and involvement in boards or charities if this information is missing. Make sure to include sources and hyperlinks.

2. **For a Nonprofit**: Make sure to include all members of the board of trustees and detailed financials if these details are missing. Provide sources and hyperlinks for all information gathered.

3. **For a Company**: Ensure to include information about top executives, primary investors, valuation, revenue, and profitability if absent. Include sources and hyperlinks for each piece of added information
    ''',
    tools=[tavily_tool,scrape_webpages,pdf_reader,nonprofit_financials],  # Optionally specify tools; defaults to an empty list
    llm=llm,
    verbose=True
)


Factcheck_agent = Agent(
    role='Fact-Checking Agent',
    backstory='''
    You are an diligent fact checking expert, skilled in critical thinking and identifying erronous information.
    ''',
    goal='''
    Make sure that the information is not compiled from two people with the same name.
    ''',
    tools=[tavily_tool,scrape_webpages,pdf_reader,nonprofit_financials],  # Optionally specify tools; defaults to an empty list
    llm=llm
    )


class StreamToExpander:
    def __init__(self):
        self.expanders = []
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']
        self.color_index = 0
        self.current_expander = None
        self.finished_chain_detected = False
        self.input_text = ""
        self.output_text = ""
        self.current_type = "input"
        self.thought_count = 0
        self.last_thought = ""
        self.last_action = ""

    def random_emoji(self):
        emojis = ["💡", "🔍", "🧠", "💭", "🤔", "💬", "🗯️", "🚀", "✨"]
        return random.choice(emojis)

    def process_content(self, content):
        if "Thought:" in content:
            thought_match = re.search(r'Thought: (.*?)(?=\n|$)', content, re.DOTALL)
            if thought_match and thought_match.group(1) != self.last_thought:
                self.thought_count += 1
                self.last_thought = thought_match.group(1)
                st.toast(f"Thought #{self.thought_count}: {self.last_thought[:100]}...", icon=f"{self.random_emoji()}")

        if "Action:" in content:
            action_match = re.search(r'Action: (.*?)\nAction Input: (.*?)(?=\n|$)', content, re.DOTALL)
            if action_match:
                action, action_input = action_match.groups()
                if f"{action}:{action_input}" != self.last_action:
                    self.last_action = f"{action}:{action_input}"
                    st.toast(f"Action: {action}\nInput: {action_input[:100]}...", icon="🛠️")

    def write(self, data):
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        if "Human:" in cleaned_data:
            self.current_type = "input"
        elif "AI:" in cleaned_data or "Thought:" in cleaned_data:
            self.current_type = "output"

        if self.current_type == "input":
            self.input_text += cleaned_data
        else:
            self.output_text += cleaned_data

        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.split("Finished chain.")[0]
            self.finished_chain_detected = True

        if self.finished_chain_detected:
            return

        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            self.color_index = (self.color_index + 1) % len(self.colors)
            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", f"<span style='color:{self.colors[self.color_index]}'>Entering new CrewAgentExecutor chain</span>")

        if "Researcher" in cleaned_data:
            cleaned_data = cleaned_data.replace("Researcher", f"<span style='color:{self.colors[self.color_index]}'>Researcher</span>")
        
        if "Fact-Checking Agent" in cleaned_data:
            cleaned_data = cleaned_data.replace("Fact-Checking Agent", f"<span style='color:{self.colors[self.color_index]}'>Fact-Checking Agent</span>")

        if "[DEBUG]:" in cleaned_data or "Thought:" in cleaned_data:
            self.current_expander = st.expander(f"Executing Intermediate Step", expanded=True)
            self.expanders.append(self.current_expander)
            self.buffer = []

        if "Task output:" in cleaned_data:
            cleaned_data = cleaned_data.replace("Task output:", "Task output:\n\n")

        if self.current_expander is None:
            self.current_expander = st.expander(f"Starting Search", expanded=True)
            self.expanders.append(self.current_expander)

        self.buffer.append(cleaned_data)

        if "\n" in data:
            content = ''.join(self.buffer)
            self.current_expander.markdown(content, unsafe_allow_html=True)
            self.process_content(content)
            self.buffer = []

    def flush(self):
        pass



query_params = st.query_params
input_value = query_params.get('input', 'Bill Gates')

with st.form(key="form"):
    user_input = ""

    if not user_input:
        user_input = st.text_input("Enter the name of a prospect or intermediary, can be a person, company or non-profit:", value=input_value)
    submit_clicked = st.form_submit_button("Generate Report")

output_container = st.empty()
# Check if submit button was clicked and clear container if needed
if with_clear_container(submit_clicked):
    if prompt := user_input:
        st.session_state["messages"] = [{"role": "user", "content": prompt}]
        task1 = Task(
        description=f"""Produce detailed, structured profiles that meticulously capture the financial complexities of {prompt}. These profiles should be rich in data and neatly organized.""",
        agent=Researcher,
        expected_output='''
        #### Individual Prospect Profile: John Doe
            - **Name:** John Doe
            - **Summary:** John Doe is a seasoned tech entrepreneur with a demonstrated history of success in the tech industry and a strong commitment to philanthropy. His current focus is on innovative solutions that address key societal challenges.
            - **Age:** 45
            - **Location:** New York
            - **Net Worth:** Approximately `$2 million, verified by [WealthX](https://www.wealthx.com/)
            - **Occupation:** Tech Entrepreneur with a focus on innovative software solutions. 
            - **Family Dynamics:** Married with two children, emphasizing a balanced work-life integration
            - **Board Affiliations:** Active in philanthropic ventures; serves on the boards of:
                - XYZ Nonprofit: Promoting educational initiatives. More details [here](https://www.xyznonprofit.org)
                - ABC Foundation: Supporting environmental sustainability. Learn more [here](https://www.abcfoundation.org/about-us)
            - **Interests:** Known for interests in renewable energy and education technology
            - **Recent News:** John Doe was recently featured in a TechCrunch article for his significant contribution to developing a new educational app that aims to improve accessibility for students with disabilities. Read the full article [here](https://techcrunch.com).
            - **Additional Information:** 
                - Advocates for technology-driven solutions to social issues
                - Actively participates in conferences and workshops related to tech innovation and social responsibility.

        #### Nonprofit Organization Profile: Help the World Grow
            - **Organization Name:** Help the World Grow
            - **Location:** Los Angeles, CA
            - **Summary:** Help the World Grow is a robust nonprofit organization with a global reach, actively working to enhance educational outcomes and reduce inequalities through strategic partnerships and impactful initiatives.
            - **Asset Size:** Estimated at `$5 million
            - **Key Members:** 
                - Jane Smith, notable for her expertise in financial strategy; profile available on the organization’s [team page](https://www.helptheworldgrow.org/team)
                - John Doe, notable for her expertise in financial strategy; profile available on the organization’s [team page](https://www.helptheworldgrow.org/team)
            - **Major Donors:**
                - XYZ Corp: Engaged in various corporate philanthropy efforts
                - ABC Foundation: Long-term supporter, focusing on impactful projects
            - **Financials:** Recent Form 990 indicates a surplus of `$200,000 in the last fiscal year. The report is accessible at [CauseIQ](https://www.causeiq.com/)
            - **Impact Highlights:** Recent projects have notably increased literacy rates in underserved regions
            - **Recent News:** The organization has launched a new initiative in partnership with local governments in South America to enhance educational infrastructure, reported last week by CNN. Full story [here](https://www.cnn.com).
            - **Additional Information:** 
                - Collaborates with educational experts and local communities to tailor programs
                - Addresses specific educational challenges in various regions

        #### Company Profile: Innovative Tech Solutions
            - **Company Name:** Innovative Tech Solutions
            - **Location:** San Diego
            - **Summary:** Innovative Tech Solutions is a leading tech company that stands at the forefront of AI and machine learning technology, with strong financial performance and strategic plans for continued growth and innovation in the industry.
            - **Industry:** Technology, specializing in AI and machine learning applications
            - **CEO:** Robert Johnson, a visionary leader with over 20 years in the tech industry. Full bio available on [Bloomberg Executives](https://www.bloomberg.com/profile/person/xxxxx)
            - **Founder:** Emily White, an entrepreneur recognized for her innovative approaches to technology development
            - **Major Investors:** Includes prominent venture capital firms such as [VentureXYZ](https://www.venturexyz.com) and [CapitalABC](https://www.capitalabc.com)
            - **Financial Performance Metrics:**
                - Current Valuation: `$50 million
                - Annual Revenue: `$10 million, demonstrating robust growth in the tech sector
                - Annual Profit: `$1 million, highlighting effective cost management and business operations
            - **Recent News:** Innovative Tech Solutions has been awarded a patent for a groundbreaking AI algorithm that optimizes energy usage in large-scale manufacturing, as reported last month by Forbes. More details [here](https://www.forbes.com).
            - **Additional Information:** 
                - Committed to sustainability, investing in green technologies
                - Aiming to reduce its carbon footprint over the next decade
                '''
        )

        task2 = Task(
        description=f"""Identify any relates entities or missing information regarding {prompt}. Perform additional research on up to 3 relates entities that is most closely related to {prompt}. These entities can be a company, related individual, or a instritution.""",
        agent=Followup_Agent,
        expected_output='''
        #### Individual Prospect Profile: John Doe
            - **Name:** John Doe
            - **Summary:** John Doe is a seasoned tech entrepreneur with a demonstrated history of success in the tech industry and a strong commitment to philanthropy. His current focus is on innovative solutions that address key societal challenges.
            - **Age:** 45
            - **Location:** New York
            - **Occupation:** Tech Entrepreneur with a focus on innovative software solutions. 
            - **Family Dynamics:** Married with two children, emphasizing a balanced work-life integration
            - **Board Affiliations:** Active in philanthropic ventures; serves on the boards of:
                - XYZ Nonprofit: Promoting educational initiatives. More details [here](https://www.xyznonprofit.org)
                - ABC Foundation: Supporting environmental sustainability. Learn more [here](https://www.abcfoundation.org/about-us)
            - **Interests:** Known for interests in renewable energy and education technology
            - **Recent News:** John Doe was recently featured in a TechCrunch article for his significant contribution to developing a new educational app that aims to improve accessibility for students with disabilities. Read the full article [here](https://techcrunch.com).
            - **Additional Information:** 
                - Advocates for technology-driven solutions to social issues
                - Actively participates in conferences and workshops related to tech innovation and social responsibility.

        #### Nonprofit Organization Profile: Help the World Grow
            - **Organization Name:** Help the World Grow
            - **Location:** Los Angeles, CA
            - **Summary:** Help the World Grow is a robust nonprofit organization with a global reach, actively working to enhance educational outcomes and reduce inequalities through strategic partnerships and impactful initiatives.
            - **Asset Size:** Estimated at `$5 million
            - **Key Members:** 
                - Jane Smith, notable for her expertise in financial strategy; profile available on the organization’s [team page](https://www.helptheworldgrow.org/team)
                - John Doe, notable for her expertise in financial strategy; profile available on the organization’s [team page](https://www.helptheworldgrow.org/team)
            - **Major Donors:**
                - XYZ Corp: Engaged in various corporate philanthropy efforts, details [here](https://www.xyzcorp.com/philanthropy)
                - ABC Foundation: Long-term supporter, focusing on impactful projects
            - **Financials:** Recent Form 990 indicates a surplus of `$200,000 in the last fiscal year. The report is accessible at [CauseIQ](https://www.causeiq.com/)
            - **Impact Highlights:** Recent projects have notably increased literacy rates in underserved regions
            - **Recent News:** The organization has launched a new initiative in partnership with local governments in South America to enhance educational infrastructure, reported last week by CNN. Full story [here](https://www.cnn.com).
            - **Additional Information:** 
                - Collaborates with educational experts and local communities to tailor programs
                - Addresses specific educational challenges in various regions

        #### Company Profile: Innovative Tech Solutions
            - **Company Name:** Innovative Tech Solutions
            - **Location:** San Diego
            - **Summary:** Innovative Tech Solutions is a leading tech company that stands at the forefront of AI and machine learning technology, with strong financial performance and strategic plans for continued growth and innovation in the industry.
            - **Industry:** Technology, specializing in AI and machine learning applications
            - **CEO:** Robert Johnson, a visionary leader with over 20 years in the tech industry. Full bio available on [Bloomberg Executives](https://www.bloomberg.com/profile/person/xxxxx)
            - **Founder:** Emily White, an entrepreneur recognized for her innovative approaches to technology development
            - **Major Investors:** Includes prominent venture capital firms such as [VentureXYZ](https://www.venturexyz.com) and [CapitalABC](https://www.capitalabc.com)
            - **Financial Metrics:**
                - Current Valuation: `$50 million
                - Annual Revenue: `$10 million, demonstrating robust growth in the tech sector
                - Annual Profit: `$1 million, highlighting effective cost management and business operations
            - **Recent News:** Innovative Tech Solutions has been awarded a patent for a groundbreaking AI algorithm that optimizes energy usage in large-scale manufacturing, as reported last month by Forbes. More details [here](https://www.forbes.com).
            - **Additional Information:** 
                - Committed to sustainability, investing in green technologies
                - Aiming to reduce its carbon footprint over the next decade
                    '''
        )

        task3 = Task(
            description="""
            Verify the factual accuracy of the provided report. Ensure that the information is correct and not compiled from two different individuals with the same name.
            """,
            agent=Factcheck_agent,
            expected_output='''
            Identification and correction of any instances where information from different individuals with the same name has been conflated.
            '''
        )


    process = Process.sequential
    agents = [Researcher]
    tasks = [task1]

    # Crew definition
    project_crew = Crew(
        tasks=tasks,
        agents=agents,
        manager_llm=llm,
        process=process,
        verbose=True
    )

    stream_to_expander = StreamToExpander()
    sys.stdout = stream_to_expander
    with st.spinner("Generating Results"):
        crew_result = project_crew.kickoff()

    input_tokens, output_tokens, input_cost, output_cost = count_tokens_and_estimate_cost(
        stream_to_expander.input_text, 
        stream_to_expander.output_text, 
        model_option
    )

    total_cost = input_cost + output_cost
    st.toast(f"Estimated Cost: ${total_cost:.4f}", icon="💸")


    dp_assistant = get_dp_assistant(model="llama3-70b-8192")
    spacing = "\n\n---\n\n"
    dp_report = ""
    for delta in dp_assistant.run(crew_result):
        dp_report += delta  # type: ignore)

    st.header("Results:")
    st.markdown(crew_result.replace("$","\$") + spacing + dp_report)
