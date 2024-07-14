import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Initialize the ChatAnthropic model
@st.cache_resource
def get_llm():
    return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

claude_model = get_llm()

# Create translation prompt template
translation_prompt = ChatPromptTemplate.from_template(
    "You are an expert translator. Translate the following English text to Chinese:\n\n{text}"
)

# Create revision prompt template
revision_prompt = ChatPromptTemplate.from_template(
    "You are a Chinese language expert. Revise the following translated Chinese text to make it sound native, concise, and professional:\n\n{text}"
)

# Create feedback implementation prompt template
feedback_prompt = ChatPromptTemplate.from_template(
    "You are an expert in refining translations. Improve the following Chinese translation based on this feedback:\n\nCurrent translation: {translation}\n\nFeedback: {feedback}\n\nProvide the improved translation."
)

# Create LangChain chains
translation_chain = LLMChain(llm=claude_model, prompt=translation_prompt)
revision_chain = LLMChain(llm=claude_model, prompt=revision_prompt)
feedback_chain = LLMChain(llm=claude_model, prompt=feedback_prompt)

# Initialize session state
if 'current_best' not in st.session_state:
    st.session_state.current_best = ""
if 'translation_done' not in st.session_state:
    st.session_state.translation_done = False
if 'revision_done' not in st.session_state:
    st.session_state.revision_done = False

# Streamlit app
st.title("English to Chinese Translation App")

# Input text area
input_text = st.text_area("Enter English text to translate:", height=150)

def translate_and_revise():
    st.session_state.translation_done = False
    st.session_state.revision_done = False

if st.button("Translate and Revise"):
    if input_text:
        translate_and_revise()

if input_text and not st.session_state.translation_done:
    with st.spinner("Translating..."):
        # Translate
        translation_result = translation_chain.run(text=input_text)
        st.session_state.current_best = translation_result
        st.session_state.translation_done = True

if st.session_state.translation_done and not st.session_state.revision_done:
    with st.spinner("Revising..."):
        # Revise
        revision_result = revision_chain.run(text=st.session_state.current_best)
        st.session_state.current_best = revision_result
        st.session_state.revision_done = True

if st.session_state.revision_done:
    st.subheader("Current Best Translation:")
    st.write(st.session_state.current_best)

    suggestion = st.text_area("Enter your suggestions for improvement (or leave blank to finish):", key="suggestion")
    if suggestion:
        if st.button("Implement Suggestion"):
            with st.spinner("Implementing suggestion..."):
                # Implement feedback
                improved_result = feedback_chain.run(translation=st.session_state.current_best, feedback=suggestion)
                st.session_state.current_best = improved_result
                st.experimental_rerun()

    st.subheader("Final Translation:")
    st.write(st.session_state.current_best)
else:
    st.info("Enter text and click 'Translate and Revise' to start.")
