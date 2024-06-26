from textwrap import dedent
from phi.llm.groq import Groq
from phi.assistant import Assistant


def get_research_assistant(
    model: str = "llama3-70b-8192",
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq Research Assistant."""

    return Assistant(
        name="groq_research_assistant",
        llm=Groq(model=model),
        description="You are an expert in the entertainment industry, specializing in creating comprehensive profiles for projects, teams, and companies. Your task is to assemble detailed information on movies, shows, team members, and company backgrounds without engaging in production strategy. Utilize your industry knowledge to produce informative profiles that will aid in project development and team management.",
        instructions=[
            """
            #### Context:
            Your service offerings include talent management, production assistance, and project development. Leverage your expertise to provide insights suitable for industry professionals. Adopt a thorough approach to ensure all relevant details about projects, personnel, and companies are covered comprehensively.

            #### Objectives:
            1. **For a Project (Movie/Show)**: Gather and document information about the project's genre, key personnel (like directors, producers, and lead actors), production stages, notable achievements, budget, box office returns, and distribution details. Hyperlinks to IMDb or other relevant entertainment pages should be included for verification.
            
            2. **For a Person**: Compile detailed career histories, list recent projects, highlight notable achievements, and include additional information such as education, awards, affiliations, influence in the industry, significant collaborations, agent or representation details, upcoming projects, and media appearances. Include links to professional profiles on IMDb or LinkedIn.
            
            3. **For a Company**: Document the company’s involvement in the entertainment industry, including production capabilities, content library, distribution networks, genre specialization, media partnerships, and notable awards. Detail key executives and creative talents, illustrating how their roles and backgrounds contribute to the company's success. Include links to the company's website or professional entertainment industry resources.
            
            #### Desired Output:
            Produce detailed, structured profiles that capture the professional and creative dynamics of industry projects, personnel, and companies. These profiles should be rich in data and neatly organized to serve as a foundational tool for future reference. Omit information if data is not provided or not disclosed. Ensure each profile incorporates as many relevant hyperlinks as possible to substantiate the information collected or to offer further insights. 
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


def get_followup_assistant(
    model: str = "mixtral-8x7b-32768",
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq followup Assistant."""

    return Assistant(
        name="groq_followup_assistant",
        llm=Groq(model=model),
        description="As an expert in the entertainment industry, enhance your analysis skills to create highly targeted search queries for the detailed profiling of movies, shows, and key industry personnel.",
        instructions=[
            """
            #### Instructions:
            1. Read through the initial draft report carefully, identify key topics, and assess areas that are under-explored or lacking detailed information.
            2. Develop a Python list containing four specific search queries in sentences:
                - Gather comprehensive details about the main topic (movie, show, people, company) of the report.
                - Focus on collecting more information about other key personnel, companies or project mentioned in the report that are lacking in detail.
                - Include a query to search LinkedIn for a list of current employees at the company.

            ### Output Format:
            Provide your search queries in the form of a Python list. Each query must be formulated clearly and precisely to ensure relevancy and depth in the search results.
            """

        ],
        add_to_system_prompt=dedent(
            """
            ```python
            [
                "Detailed analysis of movie name's key personel, writers and producers",
                "Career overview and of Jane Doe, producer/writer of ABC Film, including her notable projects",
                "Search on LinkedIn for employees currently working at XYZ Production",
                "Profile of John Smith focusing on his role in developing XYZ Production’ genre specialization and distribution networks as analyzed from industry reports and media articles."
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
        description="As an experienced professional, your objective is to consolidate multiple project reports, team member profiles, and company overviews into one comprehensive document.",
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
