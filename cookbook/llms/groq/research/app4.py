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

st.set_page_config(
    page_title="JZ NewBizBot #4",
    page_icon="💰"
)
st.title("JZ NewBizBot v4")

tavily_tool = TavilySearchResults(max_results=5)

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

llm = ChatOpenAI(model="gpt-4o")
tools = [tavily_tool, scrape_webpages]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
        """
        ### Prompt Instruction:
        You are an expert in wealth and investment management, specializing in developing comprehensive client profiles across various sectors. Your task is to create detailed financial profiles of potential clients without strategizing. Utilize your expertise to produce informative profiles that will aid in crafting personalized financial management plans later. Include hyperlinks to essential financial data sources like Bloomberg, Forbes, and specific financial databases for additional context.

        #### Context:
        Your service offerings include investment management, Outsourced Chief Investment Officer (OCIO) services, private banking, single-stock risk handling, and trust & estate planning. Leverage your expertise to provide analytical insights suitable for a diverse client base. Adopt a methodical and detail-oriented approach to ensure all pertinent financial details are covered comprehensively.

        #### Objectives:
        1. **For an Individual**: Gather and document information about the individual’s employment history, age, personal net worth, diverse income sources, family circumstances, and involvement in boards or charities. Hyperlinks to LinkedIn or other relevant professional pages should be included for verification of employment history.

        2. **For a Nonprofit**: Compile the nonprofit’s asset details, highlight key Investment Committee members, enumerate major donors, and review their financial transparency using links to platforms like [Cause IQ](https://www.causeiq.com/) and [ProPublica](https://www.propublica.org/) for access to recent Form 990s.

        3. **For a Company**: Create thorough profiles for top executives, pinpoint primary investors, record significant financial milestones, and evaluate the company's financial health using metrics like valuation, revenue, and profitability. Link to resources such as [Yahoo Finance](https://finance.yahoo.com/) or the company website for financial reports and analyses.

        #### Desired Output:
        Produce detailed, structured profiles that meticulously capture the financial and personal complexities of potential clients. These profiles should be rich in data and neatly organized to serve as a foundational tool for subsequent personalized financial planning and advisory sessions. Ensure each profile incorporates relevant hyperlinks to substantiate the data collected or to offer further insights.

        #### Individual Prospect Profile: John Doe
            - **Name:** John Doe
            - **Summary:** John Doe is a seasoned tech entrepreneur with a demonstrated history of success in the tech industry and a strong commitment to philanthropy. His current focus is on innovative solutions that address key societal challenges.
            - **Age:** 45
            - **Location:** New York
            - **Net Worth:** Approximately `$2 million, verified by [WealthX](https://www.wealthx.com/)
            - **Occupation:** Tech Entrepreneur with a focus on innovative software solutions. View the LinkedIn Profile: [John Doe](https://www.linkedin.com/in/johndoe/)
            - **Family Dynamics:** Married with two children, emphasizing a balanced work-life integration
            - **Board Affiliations:** Active in philanthropic ventures; serves on the boards of:
                - XYZ Nonprofit: Promoting educational initiatives. More details [here](https://www.xyznonprofit.org)
                - ABC Foundation: Supporting environmental sustainability. Learn more [here](https://www.abcfoundation.org/about-us)
            - **Interests:** Known for interests in renewable energy and education technology
            - **Recent News:** John Doe was recently featured in a TechCrunch article for his significant contribution to developing a new educational app that aims to improve accessibility for students with disabilities. Read the full article [here](https://techcrunch.com).
            - **Additional Information:** 
                - Advocates for technology-driven solutions to social issues
                - Actively participates in conferences and workshops related to tech innovation and social responsibility, with schedules and events listed on [EventBrite](https://www.eventbrite.com).

            #### Nonprofit Organization Profile: Help the World Grow
            - **Organization Name:** Help the World Grow
            - **Location:** Los Angeles, CA
            - **Summary:** Help the World Grow is a robust nonprofit organization with a global reach, actively working to enhance educational outcomes and reduce inequalities through strategic partnerships and impactful initiatives.
            - **Mission:** Dedicated to fostering educational opportunities and reducing inequality worldwide
            - **Asset Size:** Estimated at `$5 million
            - **Investment Committee Key Member:** Jane Smith, notable for her expertise in financial strategy; profile available on the organization’s [team page](https://www.helptheworldgrow.org/team)
            - **Major Donors:**
                - XYZ Corp: Engaged in various corporate philanthropy efforts, details [here](https://www.xyzcorp.com/philanthropy)
                - ABC Foundation: Long-term supporter, focusing on impactful projects
            - **Financial Disclosures:** Recent Form 990 indicates a surplus of `$200,000 in the last fiscal year. The report is accessible at [CauseIQ](https://www.causeiq.com/)
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
        """,
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def upload_file_to_cloudflare_r2(file_path, object_name):
    """Upload a file to Cloudflare R2 Storage in the 'newbizbot' bucket, checking for duplicates and modifying the object name if necessary."""
    # Use environment variables for Cloudflare API keys
    access_key = os.getenv('CLOUDFLARE_ACCESS_KEY')
    secret_key = os.getenv('CLOUDFLARE_SECRET_KEY')

    # Endpoint for Cloudflare R2 storage
    endpoint_url = 'https://44ae5977e790e0a48e71df40637d166a.r2.cloudflarestorage.com/'

    # Hardcoded bucket name
    bucket_name = 'newbizbot'

    # Create a client for the Cloudflare R2 storage
    session = boto3.session.Session()
    s3_client = session.client('s3',
                               region_name='auto',
                               endpoint_url=endpoint_url,
                               aws_access_key_id=access_key,
                               aws_secret_access_key=secret_key,
                               config=Config(signature_version='s3v4'))

    # Extract the file name and folder path from the object name
    file_name = os.path.basename(object_name)
    folder_path = os.path.dirname(object_name)

    # Define a method to generate a new object name with a number appended
    def generate_new_object_name(index):
        base_name, extension = os.path.splitext(file_name)
        return f"{base_name}_{index}{extension}"

    # Start checking for existing files and generate new object name if necessary
    index = 1
    new_object_name = file_name
    while True:
        try:
            s3_client.head_object(Bucket=bucket_name, Key=os.path.join(folder_path, new_object_name))
            new_object_name = generate_new_object_name(index)
            index += 1
        except s3_client.exceptions.ClientError as e:
            if int(e.response['Error']['Code']) == 404:
                break  # Object does not exist, and we can use this name
            else:
                raise  # Some other error, raise it

    # Update the object name with the potentially new file name
    object_name = os.path.join(folder_path, new_object_name)

    # Upload the file
    with open(file_path, "rb") as file:
        s3_client.upload_fileobj(file, bucket_name, object_name)
        print(f"File uploaded successfully to {object_name}")

query_params = st.query_params
input_value = query_params.get('input', 'Bill Gates')

with st.form(key="form"):
    user_input = ""

    if not user_input:
        user_input = st.text_input("Enter the name of a prospect or intermediary, can be a person, company or non-profit:", value=input_value)
    submit_clicked = st.form_submit_button("Generate Report")

output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="💰")
    st_callback = StreamlitCallbackHandler(answer_container)
    cfg = RunnableConfig()
    cfg["callbacks"] = [st_callback]

    # If we've saved this question, play it back instead of actually running LangChain
    # (so that we don't exhaust our API calls unnecessarily)

    answer = agent_executor.invoke({"input": user_input}, cfg)
    

    answer_container.write(answer["output"])

    file_path = f'/tmp/{user_input}_first_report.txt'
    with open(file_path, 'w') as file:
        file.write(answer["output"])
    with open(file_path, 'rb') as file:
        cloud_path = f'/Apps/NewBizBot/{user_input}_first_report.txt'
        upload_file_to_cloudflare_r2(file_path, cloud_path)