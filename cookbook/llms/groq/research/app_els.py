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


try:
    from tavily import TavilyClient
except ImportError:
    raise ImportError("`tavily-python` not installed. Please install using `pip install tavily-python`")

from os import getenv

st.set_page_config(
    page_title="ELS Networking Bot",
    page_icon="🎬"
)

st.title("ELS Networking Bot")

# Initialize the Claude model
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

            webpage_links = []
            for result in clean_response["results"]:
                url = result['url']
                if "https://www.linkedin.com/in" in url:
                    continue
                else:
                    webpage_links.append(url)
            if webpage_links:
                _markdown += f"Links for scrape_webpages tool: {webpage_links}\n"
            return _markdown


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
    
    def resize_images(content: str) -> str:
        pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        
        def replacement(match):
            alt_text = match.group(1)
            image_url = match.group(2)
            return f'<img src="{image_url}" alt="{alt_text}" style="max-width:300px; width:100%;" />'
        
        return re.sub(pattern, replacement, content)
    
    for url in urls:
        response = requests.get("https://r.jina.ai/" + url)
        content = response.text.replace("$","\$")
        resized_content = resize_images(content)
        combined_content += resized_content
        if len(combined_content) > 100000:
            break
            
    return combined_content[:100000]

Researcher = Agent(
    role='Entertainment Industry Researcher',
    backstory='''You are an expert in the entertainment industry, specializing in creating comprehensive profiles for projects, teams, and companies. Your task is to assemble detailed information on movies, shows, team members, and company backgrounds without engaging in production strategy. Utilize your industry knowledge to produce informative profiles that will aid in project development and team management.''',
    goal='''
    Create detailed profiles based on the type of entity:
    1. For a Project (Movie/Show): Gather and document information about the project's genre, key personnel (like directors, producers, writers and lead actors), production stages, notable achievements, budget, box office returns, and distribution details. Include hyperlinks to IMDb or other relevant entertainment pages for verification.
    2. For a Person: Compile detailed career histories, list recent projects, highlight notable achievements, and include additional information such as education, awards, affiliations, influence in the industry, significant collaborations, agent or representation details, upcoming projects, and media appearances. Include links to professional profiles on IMDb or LinkedIn.
    3. For a Company: Document the company's involvement in the entertainment industry, including production capabilities, content library, distribution networks, genre specialization, media partnerships, and notable awards. Detail key executives and creative talents, illustrating how their roles and backgrounds contribute to the company's success. Include links to the company's website or professional entertainment industry resources.
    ''',
    tools=[tavily_tool, scrape_webpages],
    llm=llm,
    verbose=True
)

class StreamToExpander:
    def __init__(self):
        self.expanders = []
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']
        self.color_index = 0
        self.current_expander = None
        self.finished_chain_detected = False

    def write(self, data):
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

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

        if "Entertainment Industry Researcher" in cleaned_data:
            cleaned_data = cleaned_data.replace("Entertainment Industry Researcher", f"<span style='color:{self.colors[self.color_index]}'>Entertainment Industry Researcher</span>")

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
            self.current_expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

    def flush(self):
        pass

query_params = st.query_params
input_value = query_params.get('input', 'Steven Spielberg')

with st.form(key="form"):
    user_input = st.text_input("Enter the name of a person, project, or company in the entertainment industry:", value=input_value)
    submit_clicked = st.form_submit_button("Generate Report")

output_container = st.empty()
if with_clear_container(submit_clicked):
    if prompt := user_input:
        st.session_state["messages"] = [{"role": "user", "content": prompt}]
        task = Task(
            description=f"""Produce a detailed, structured profile that meticulously captures the professional and creative dynamics of {prompt} in the entertainment industry. This profile should be rich in data and neatly organized.""",
            agent=Researcher,
            expected_output='''
            #### Project Profile: ABC Film
            - **Project Name:** ABC Film
            - **Genre:** Adventure/Comedy
            - **Director:** [Jane Doe](https://www.imdb.com/name/nmXXXXX)
            - **Lead Actors:** [John Smith](https://www.imdb.com/name/nmXXXXX), [Emily White](https://www.imdb.com/name/nmXXXXX)
            - **Production Status:** Pre-production
            - **Budget:** $120 million
            - **Box Office:** Projected $300 million
            - **Distribution Details:** Released in theaters worldwide and available on [XYZ Streaming Service](https://www.xyzstreaming.com)
            - **Notable Achievements:** Selected for the [upcoming international film festival](https://www.filmfestival.com). More details [here](https://www.imdb.com/title/tt1234567/)
            - **Additional Information:**
                - Expected to significantly impact the adventure genre
                - High anticipation from the audience, evidenced by [social media trends](https://twitter.com/hashtag/TheGreatAdventure)
                #### Key Personnel
                ##### Director: Jane Doe
                - **Role:** Director
                - **Age:** 45
                - **Notable Past Works:** [Film A](link), [Film B](link)
                - **Awards:** [Award 1](link), [Award 2](link)
                - **Unique Style/Contribution:** Known for innovative use of CGI
                - **IMDb Profile:** [Jane Doe](link)
                ##### Lead Actor: John Smith
                - **Role:** Lead Actor
                - **Age:** 38
                - **Notable Past Roles:** [Role A](link), [Role B](link)
                - **Awards:** [Award 1](link), [Award 2](link)
                - **Special Skills:** Martial arts, fluent in 3 languages
                - **IMDb Profile:** [John Smith](link)
                ##### Lead Actress: Emily White
                - **Role:** Lead Actress
                - **Age:** 32
                - **Notable Past Roles:** [Role X](link), [Role Y](link)
                - **Awards:** [Award 1](link), [Award 2](link)
                - **Special Skills:** Opera singing, horseback riding
                - **IMDb Profile:** [Emily White](link)
                ##### Screenwriter: Sarah Johnson
                - **Role:** Screenwriter
                - **Age:** 41
                - **Notable Past Works:** [Script A](link), [Script B](link)
                - **Awards:** [Award 1](link), [Award 2](link)
                - **Unique Perspective:** Known for complex, multi-layered narratives
                - **IMDb Profile:** [Sarah Johnson](link)
                ##### Producer: David Lee
                - **Role:** Producer
                - **Age:** 55
                - **Notable Past Productions:** [Film P](link), [Film Q](link)
                - **Industry Recognition:** [Achievement 1](link), [Achievement 2](link)
                - **Specialization:** Expertise in international co-productions
                - **IMDb Profile:** [David Lee](link)

            #### Individual Profile: John Smith
            - **Name:** John Smith
            - **Role:** Director
            - **Recent Projects:** [The Last Stand](https://www.imdb.com/title/ttXXXXX), [Night in the Jungle](https://www.imdb.com/title/ttXXXXX)
            - **Notable Achievements:** Winner of the [Best Director at the 2022 Global Film Awards](https://globalfilmawards.com/past-winners)
            - **Education and Training:** Graduated from the [American Film Institute](https://www.afi.com)
            - **Awards and Nominations:** Multiple [Academy Award nominations](https://www.oscars.org/oscars/ceremonies/2022)
            - **Professional Affiliations:** Member of the Directors Guild of America
            - **Influence and Impact in the Industry:** Known for pioneering the use of virtual reality in film
            - **Significant Collaborations:** Long-time collaboration with screenwriter [Emily White](https://www.imdb.com/name/nmXXXXX)
            - **Agent or Representation:** Represented by Creative Artists Agency
            - **Upcoming Projects:** Directing the upcoming film "[Future's Edge](https://www.imdb.com/title/ttXXXXX)"
            - **Media Appearances:** Recently featured on "[The Tonight Show](https://www.nbc.com/the-tonight-show)"
            - **IMDb Profile:** [John Smith](https://www.imdb.com/name/nm1234567/)
            - **Additional Information:**
                - Known for a unique storytelling style
                - Actively involved in mentoring young filmmakers

            #### Company Profile: XYZ Production
            - **Company Name:** XYZ Production
            - **Industry:** Entertainment
            - **Key Projects:** ABC Film, Tomorrow's Dawn
            - **Content Library:** Over 500 titles including award-winning films and series
                - [Movie 1](https://www.imdb.com/title/ttXXXXX)
                - [Movie 2](https://www.imdb.com/title/ttXXXXX)
                - [Show 1](https://www.imdb.com/title/ttXXXXX)
                - [Show 2](https://www.imdb.com/title/ttXXXXX)
            - **Distribution Networks:** Global distribution through major streaming services and theatrical releases
            - **Awards and Accolades:** Multiple [Golden Globe](https://www.goldenglobes.com) and [Emmy Awards](https://www.emmys.com)
            - **Key Members:**
                - John Doe - Oversees all company operations and strategic direction. [LinkedIn Profile](https://www.linkedin.com/in/johndoe)
                - Jane Smith - Manages financial strategies and budget planning. [LinkedIn Profile](https://www.linkedin.com/in/janesmith)
                - Emily White - Leads creative teams and directs the artistic vision of projects. [LinkedIn Profile](https://www.linkedin.com/in/emilywhite)
                - Michael Brown - Coordinates production activities, ensuring efficiency and quality control. [LinkedIn Profile](https://www.linkedin.com/in/michaelbrown)
                - Lisa Green - Directs marketing strategies and brand partnerships. [LinkedIn Profile](https://www.linkedin.com/in/lisagreen)
            - **Website:** [XYZ Production](https://www.dynamicproductions.com)
            - **Additional Information:**
                - Recognized for innovation in visual effects and narrative techniques
                - Committed to fostering new talent in the entertainment industry
                #### Additional Key Personnel at XYZ Production
                ##### Chief Operating Officer: Amanda Clark
                - **Role:** COO
                - **Age:** 48
                - **Background:** MBA from Harvard Business School
                - **Previous Experience:** Former VP at [Major Studio](link)
                - **Specialization:** Strategic planning and operational efficiency
                - **LinkedIn Profile:** [Amanda Clark](link)
                ##### Head of Development: Thomas Wright
                - **Role:** Head of Development
                - **Age:** 43
                - **Notable Projects Developed:** [Project C](link), [Project D](link)
                - **Industry Recognition:** [Achievement 1](link), [Achievement 2](link)
                - **Specialization:** Identifying and nurturing new talent and scripts
                - **LinkedIn Profile:** [Thomas Wright](link)
                ##### Chief Technology Officer: Jennifer Lee
                - **Role:** CTO
                - **Age:** 37
                - **Background:** Ph.D. in Computer Science from MIT
                - **Previous Experience:** Led R&D at [Tech Company](link)
                - **Specialization:** Implementation of AI in production processes
                - **LinkedIn Profile:** [Jennifer Lee](link)
            '''
        )

        # Crew definition
        project_crew = Crew(
            tasks=[task],
            agents=[Researcher],
            process = Process.sequential,
            verbose=True
        )

        stream_to_expander = StreamToExpander()
        sys.stdout = stream_to_expander
        with st.spinner("Generating Results"):
            crew_result = project_crew.kickoff()

        st.header("Results:")
        st.markdown(crew_result.replace("$","\$"))
