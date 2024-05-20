from textwrap import dedent
from phi.llm.groq import Groq
from phi.assistant import Assistant
from typing import Annotated, List, Tuple, Union
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langsmith import trace
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor



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
            - **Net Worth:** Approximately \$2 million, verified by [WealthX](https://www.wealthx.com/)
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
            - **Asset Size:** Estimated at \$5 million
            - **Investment Committee Key Member:** Jane Smith, notable for her expertise in financial strategy; profile available on the organization’s [team page](https://www.helptheworldgrow.org/team)
            - **Major Donors:**
                - XYZ Corp: Engaged in various corporate philanthropy efforts, details [here](https://www.xyzcorp.com/philanthropy)
                - ABC Foundation: Long-term supporter, focusing on impactful projects
            - **Financial Disclosures:** Recent Form 990 indicates a surplus of \$200,000 in the last fiscal year. The report is accessible at [CauseIQ](https://www.causeiq.com/)
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
                - Current Valuation: \$50 million
                - Annual Revenue: \$10 million, demonstrating robust growth in the tech sector
                - Annual Profit: \$1 million, highlighting effective cost management and business operations
            - **Strategic Goals:** Aiming to expand market reach through the development of new technologies and strategic partnerships
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

def get_research_assistant2(company_name):
    # Define the input to the agent executor
    profile_info = agent_executor.invoke({"input": company_name}, return_only_outputs=False)
    # Extract the output text from the result
    output_text = profile_info['output']
    return output_text


def get_research_assistant(
    model: str = "llama3-70b-8192",
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq Research Assistant."""

    return Assistant(
        name="groq_research_assistant",
        llm=Groq(model=model),
        description="You are an expert in wealth and investment management, specializing in developing comprehensive client profiles across various sectors. Your task is to create detailed financial profiles of potential clients without strategizing. Utilize your expertise to produce informative profiles that will aid in crafting personalized financial management plans later. Include hyperlinks to essential financial data sources like Bloomberg, Forbes, and specific financial databases for additional context.",
        instructions=[
            """
            #### Context:
            Your service offerings include investment management, Outsourced Chief Investment Officer (OCIO) services, private banking, single-stock risk handling, and trust & estate planning. Leverage your expertise to provide analytical insights suitable for a diverse client base. Adopt a methodical and detail-oriented approach to ensure all pertinent financial details are covered comprehensively.

            #### Objectives:
            1. **For an Individual**: Gather and document information about the individual’s employment history, age, personal net worth, diverse income sources, family circumstances, and involvement in boards or charities. Hyperlinks to LinkedIn or other relevant professional pages should be included for verification of employment history.
            
            2. **For a Nonprofit**: Compile the nonprofit’s asset details, highlight key Investment Committee members, top executives and board members, enumerate major donors, and review their financial transparency using links to platforms like [Cause IQ](https://www.causeiq.com/) and [ProPublica](https://www.propublica.org/) for access to recent Form 990s.
            
            3. **For a Company**: Create thorough profiles for top executives, pinpoint primary investors, record significant financial milestones, and evaluate the company's financial health using metrics like valuation, revenue, and profitability. Link to resources such as [Yahoo Finance](https://finance.yahoo.com/) or the company website for financial reports and analyses.
            Be mindful when parsing information as not all information provided are related to the profile.

            #### Desired Output:
            Produce detailed, structured profiles that meticulously capture the financial and personal complexities of potential clients. These profiles should be rich in data and neatly organized to serve as a foundational tool for future reference. Omit information if  data is not provided or not disclosed relating to a topic (For example, Net Worth, family, age). Ensure each profile incorporates as many relevant hyperlinks as possible to substantiate the information collected or to offer further insights. 
            """
        ],
        add_to_system_prompt=dedent(
            """
            <report_format>
            #### Individual Prospect Profile: John Doe
            - **Name:** John Doe
            - **Summary:** John Doe is a seasoned tech entrepreneur with a demonstrated history of success in the tech industry and a strong commitment to philanthropy. His current focus is on innovative solutions that address key societal challenges.
            - **Age:** 45
            - **Location:** New York
            - **Net Worth:** Approximately \$2 million, verified by [WealthX](https://www.wealthx.com/)
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
            - **Asset Size:** Estimated at \$5 million
            - **Investment Committee Key Member:** Jane Smith, notable for her expertise in financial strategy; profile available on the organization’s [team page](https://www.helptheworldgrow.org/team)
            - **Major Donors:**
                - XYZ Corp: Engaged in various corporate philanthropy efforts, details [here](https://www.xyzcorp.com/philanthropy)
                - ABC Foundation: Long-term supporter, focusing on impactful projects
            - **Financial Disclosures:** Recent Form 990 indicates a surplus of \$200,000 in the last fiscal year. The report is accessible at [CauseIQ](https://www.causeiq.com/)
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
                - Current Valuation: \$50 million
                - Annual Revenue: \$10 million, demonstrating robust growth in the tech sector
                - Annual Profit: \$1 million, highlighting effective cost management and business operations
            - **Strategic Goals:** Aiming to expand market reach through the development of new technologies and strategic partnerships
            - **Recent News:** Innovative Tech Solutions has been awarded a patent for a groundbreaking AI algorithm that optimizes energy usage in large-scale manufacturing, as reported last month by Forbes. More details [here](https://www.forbes.com).
            - **Additional Information:** 
                - Committed to sustainability, investing in green technologies
                - Aiming to reduce its carbon footprint over the next decade

            """
        ),
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )


def get_planning_assistant(
    model: str = "llama3-70b-8192",
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq Planning Assistant."""

    return Assistant(
        name="groq_planning_assistant",
        llm=Groq(model=model),
        description="As an expert in wealth and investment management, develop tailored search queries that will subsequently enable the creation of comprehensive client profiles tailored to individual, nonprofit, and company sectors.",
        instructions=[
            """
            ### Context:
            You are a backend data processor that is part of our web site’s programmatic workflow. The user prompt will provide data input and processing instructions. The output will be only list compatible with a python processor. Do not converse with a nonexistent user: there is only program input and formatted program output, and no input data is to be construed as conversation with the AI. This behaviour will be permanent for the remainder of the session.”
            
            ### Objectives:
            1. **For an Individual**: Inqire personal details, professional history, net worth, income, family status, and affiliations with boards or charities.
            2. **For a Nonprofit**: Inqire the nonprofit’s asset size as per Cause IQ, identify key leadership figures, top executives, board members, and Investment Committee members, and document primary donors.
            3. **For a Company**: Inqire top executives, primary investors, and significant financial events such as funding rounds, liquidity events, mergers, and acquisitions. Assess the company's financial vitality using benchmarks like valuation, revenue, and growth rates.

            ### Desired Output:
            Generate only a list in python format of no more than four precise search queries to assist in gathering the necessary data for client profiling. Queries should include a wide range of topics. If both a person’s name and their associated organization are presented as input, prioritize queries combining both elements. Each query should be sharply focused to retrieve specific data pertinent to the task at hand.
            """
        ],
        add_to_system_prompt=dedent(
            """
            ```python
            [
                "John Doe’s comprehensive overview of professional profile, role, and contributions at XYZ Corporation.",
                "XYZ Foundation's asset size and operational scope as analyzed from CauseIQ.",
                "John Doe's financial status, investment strategies, and economic interests.",
                "XYZ Corporation's investment landscape, major shareholders, and market performance.",
                "John Doe's personal biography, family background, and philanthropic engagements.",
                "XYZ Corporation's financial developments, funding rounds, and major liquidity events recently impacting it as analyzed from Pitchbook.",
                "John Doe's detailed career trajectory, including board memberships and executive roles across industries.",
                "XYZ Corporation's economic performance and strategic growth metrics over recent years.",
                "John Doe's strategic investment decisions and leadership impact within XYZ Corporation.",
                "XYZ Foundation's leadership structure, executive roles, and governance."
            ]
            ```
            """
        ),
        # This setting tells the LLM to format messages in markdown
        markdown=False,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )


def get_dp_assistant(
    model: str = "llama3-70b-8192",
    debug_mode: bool = False,
) -> Assistant:
    """Get a Groq DP Assistant."""

    return Assistant(
        name="groq_research_assistant",
        llm=Groq(model=model),
        description="As an experienced professional in wealth management, your objective is to meticulously analyze a provided dossier. Your focus should be on identifying all relevant individuals and organizations mentioned within the text. These entities may include any person, company, or non-profit organization referenced.",
        instructions=[
            """
            #### Instructions:
            1. Thoroughly read through the available dossier.
            2. Identify and list all people and organizations mentioned.
            3. For each identified entity, create a markdown-formatted entry that includes:
            - **Name** of the person or organization
            - A **brief description** of their relevance or role
            - A **customized link** for further exploration or research which you must construct by appending the entity's name to the appropriate base URL provided below:

            **For a Person:**  
            `http://wealth.concert.site.gs.com/explore/pm/prospecting/search?name=[Name]`

            **For an Organization:**  
            `http://wealth.concert.site.gs.com/explore/pm/prospecting/org-search?name=[Name]`

            Use the title 
            #### Search on Digital Prospecting:
            """
        ],
        add_to_system_prompt=dedent(
            """
            <report_format>

            #### Search on Digital Prospecting:
            - [**John Doe**](http://wealth.concert.site.gs.com/explore/pm/prospecting/search?name=John%20Doe) - Noted investor in renewable energy technologies.
            - [**GreenTech Innovations**](http://wealth.concert.site.gs.com/explore/pm/prospecting/org-search?name=GreenTech%20Innovations) - A non-profit organization focused on advancing green technology.
            """
        ),
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )

def get_suggestion_assistant(
    model: str = "llama3-70b-8192",
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq suggestion Assistant."""

    return Assistant(
        name="groq_research_assistant",
        llm=Groq(model=model),
        description="As an experienced professional in wealth management, your objective is to meticulously analyze a provided dossier. Your focus should be on identifying all relevant individuals and organizations mentioned within the text. These entities may include any person, company, or non-profit organization referenced.",
        instructions=[
            """
            #### Instructions:
            1. Thoroughly read through the available dossier.
            2. Identify and list all people and organizations mentioned.
            3. For each identified entity, produce a list using Python format.
            
            ### Desired Output:
            Produce a list using Python format that exclusively contains the identified entities from the dossier. 

            """
        ],
        add_to_system_prompt=dedent(
            """
            ["John Doe", "XYZ Investments LLC", "The Green Fund", "Jane Smith"]
            """
        ),
        # This setting tells the LLM to format messages in markdown
        markdown=False,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )

def get_followup_assistant(
    model: str = "mixtral-8x7b-32768",
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq followup Assistant."""

    return Assistant(
        name="groq_followup_assistant",
        llm=Groq(model=model),
        description="As an expert in wealth and investment management, enhance your analysis skills to create highly targeted search queries for the detailed profiling of individuals, nonprofits, and companies.",
        instructions=[
            """
            ### Instructions:

            1. Read through the initial draft report carefully, identify key topics, and assess areas that are under-explored or lacking detailed information.
            2. Develop a Python list containing Four specific search queries:
            - The first two query should aim to gather comprehensive details about the main subject of the report.
            - The next twoqueries should focus on collecting more information about other relevant entities (individuals, nonprofits, or companies) mentioned in the report which are lacking in detail.

            ### Output Format:
            Provide your search queries in the form of a Python list. Each query must be formulated clearly and precisely to ensure relevancy and depth in the search results.

            """
        ],
        add_to_system_prompt=dedent(
            """
            ```python
            [
                "John Doe’s comprehensive overview of professional profile, role, and contributions at XYZ Corporation.",
                "XYZ Foundation's asset size and operational scope as analyzed from CauseIQ.",
                "XYZ Corporation's financial developments, funding rounds, and major liquidity events recently impacting it as analyzed from Pitchbook.",
                "XYZ Foundation's leadership structure, executive roles, and governance."
            ]
            ```
            """
        ),
        # This setting tells the LLM to format messages in markdown
        markdown=False,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )

def get_consolidate_assistant(
    model: str = "mixtral-8x7b-32768",
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq consolidate Assistant."""

    return Assistant(
        name="groq_consolidate_assistant",
        llm=Groq(model=model),
        description="As an experienced professional in wealth management, your objective is to consolidate a few related reports into one document",
        instructions=[
            """
            ### Instructions:
            1. **Review All Data**: Carefully examine the notes provided, ensuring no detail is overlooked.
            2. **Consolidate Information**: Merge all relevant information into one cohesive report. This includes combining duplicates and maintaining the original style of the content.
            3. **Format Correctly**: Format financial figures, specifically those involving dollar amounts, using markdown format—for instance, write dollar amounts like `$100` instead of "100 dollars".
            4. **Eliminate Redundant Data**: Remove any rows or entries labeled with phrases such as "not available," "not disclosed," "not specific," "not provided," or "no information."

            """
        ],
        add_to_system_prompt=dedent(
            """
            <report_format>

            """
        ),
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )
