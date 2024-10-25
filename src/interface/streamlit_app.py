import argparse
import re
from typing import Dict, List, Tuple
import streamlit as st

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from pathlib import Path

from src.retrieval.retriever import initialize_retriever
from src.utils.config import load_config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file'
    )
    args, _ = parser.parse_known_args()
    return args

def format_answer_with_links(result: Dict) -> Tuple[str, List[Dict[str, str]]]:
    """Format the answer with proper citations and context snippets."""
    answer = result["answer"]
    citations = {}
    formatted_citations = []
    found_citations = set()
    
    # Build citations dictionary from context
    for doc in result.get('context', []):
        page_number = str(doc.metadata.get('page', 'Unknown'))
        context = doc.page_content.strip()
        if len(context) > 800:
            context = context[:800].rsplit(' ', 1)[0] + '...'
        citations[page_number] = context

    # Normalize double brackets to single
    answer = re.sub(r'\[\[([^\]]+)\]\]', r'[\1]', answer)
    
    # Process citations from right to left to maintain string positions
    for match in sorted(
        re.finditer(r'\[(?:Page\s*)?(\d+)\]', answer), 
        key=lambda x: x.start(), 
        reverse=True
    ):
        start, end = match.span()
        page_number = match.group(1)

        # Skip invalid citations
        if page_number not in citations:
            answer = answer[:start] + answer[end:]
            continue

        # Track unique citations for footnotes
        if page_number not in found_citations:
            found_citations.add(page_number)
            formatted_citations.append({
                'page': page_number,
                'text': citations[page_number]
            })

        # Replace citation with styled version
        styled_citation = (
            f'<span class="citation" onclick="showFootnote(\'{page_number}\')">'
            f'[Page {page_number}]</span>'
        )
        answer = answer[:start] + styled_citation + answer[end:]

    return answer, formatted_citations

# Parse arguments
args = parse_args()
config_path = args.config

# Initialize session state before anything else
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.user_question = ""
    st.session_state.bot_answer = ""
    st.session_state.footnotes = []

# Set page config
st.set_page_config(page_title="USCIS Chatbot", layout="wide")

# Load configuration and initialize resources only once
if not st.session_state.initialized:
    config = load_config(config_path)
    
    # Initialize model and retriever
    model_name = config.get('model', {}).get('name', "llama3.1:8b")
    llm = OllamaLLM(
        model=model_name,
        temperature=config.get('model', {}).get('temperature', 0.7)
    )
    retriever = initialize_retriever(config, llm)
    
    # Create prompt template
    system_prompt = """
    Answer the user's question based on the provided context with clear and concise responses. 
    
    Step 1: Answer the question directly in no more than 128 words or about 4-6 sentences. Focus on clarity and accuracy.
    
    Step 2: Include page citations in single square brackets [Page #] after each piece of information referenced from the context. Ensure every claim is properly cited. If no reference is available, say "I don't know."
    
    If unsure or ambiguous, prioritize precision and stop generating after the answer. Do not generate uncited claims. Keep your answer concise and within 256 words/tokens.
    
    Do not repeat yourself or be unnecessarily verbose.
    
    Your answer will be parsed for citations and will be processed by Streamlit's markdown renderer, so make sure to use the correct format.
    
    Example of citation format:
    Canines are mammals [Page 21]. They are also known as doggies [Page 35]. Cats are also mammals [Page 2].
    
    CITATIONS ARE MANDATORY. UNCITED CLAIMS WILL BE PENALIZED.
    IF USING BULLETS, HAVE A CITATION FOR EACH BULLET POINT.
    
    Context:
    {context}
    
    Question:
    {input}
    
    Answer: 
    """
    
    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["context", "question"]
    )

    document_prompt = PromptTemplate.from_template("[Page {page}]: {page_content}")

    # Create document chain with custom formatting
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=document_prompt,
        document_separator="\n\n",
        document_variable_name="context"
    )

    # Create retrieval chain with the correct parameter name
    qa = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    # Store in session state
    st.session_state.config = config
    st.session_state.llm = llm
    st.session_state.retriever = retriever
    st.session_state.qa = qa
    st.session_state.initialized = True

# Access resources from session state
qa = st.session_state.qa

st.title('🛂  USCIS Chatbot')
st.markdown('''
Ask any question regarding the USCIS manual, and receive detailed, cited answers to help you navigate the complexities of immigration procedures.
''')

# Sidebar for advanced options
num_documents = st.sidebar.slider(
    '🗂 Number of Documents to Search',
    min_value=1,
    max_value=10,
    value=3,
    help="Select the number of documents to include in the search for generating responses."
)
temperature = st.sidebar.slider(
    '🌡️ Model Temperature',
    min_value=0.0,
    max_value=1.0,
    step=0.1,
    value=0.7,
    help="Adjust the randomness of the model's responses. Higher values lead to more creative answers."
)
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.user_question = ""
    st.session_state.bot_answer = ""
    st.session_state.footnotes = []

# Input area
st.markdown("### 💬 Ask a Question")
with st.form(key='input_form', clear_on_submit=True):
    user_question = st.text_input(
        '',
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button(label='Send 🚀')

    if submit_button and user_question:
        # Update session state
        st.session_state.user_question = user_question
        st.session_state.bot_answer = ""
        st.session_state.footnotes = []
        with st.spinner("🔍 Searching for answers..."):
            try:
                result = qa.invoke({
                    "input": user_question,
                    "num_documents": num_documents,
                    "temperature": temperature
                })
                formatted_answer, footnotes = format_answer_with_links(result)
                st.session_state.bot_answer = formatted_answer
                st.session_state.footnotes = footnotes
            except Exception as e:
                st.session_state.bot_answer = f"⚠️ An error occurred: {e}"

# Display the answer and citations
if st.session_state.user_question and st.session_state.bot_answer:
    # Question display
    st.markdown(f"""
    <div class="question-container">
        <div class="question-label">Your Question:</div>
        <div class="question-text">{st.session_state.user_question}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Answer display
    st.markdown(f"""
    <div class="answer-container">
        <div class="answer-label">Answer:</div>
        <div class="answer-text">{st.session_state.bot_answer}</div>
    </div>
    """, unsafe_allow_html=True)

    # Display citations in an expander with better formatting
    if st.session_state.footnotes:
        with st.expander("📚 Source References"):
            for citation in st.session_state.footnotes:
                st.markdown(f"""
                <div class="citation-container" id="footnote-{citation['page']}">
                    <div class="citation-page">Page {citation['page']}</div>
                    <div class="citation-text">{citation['text']}</div>
                </div>
                """, unsafe_allow_html=True)

# Add custom CSS for better styling
st.markdown("""
<style>
/* Container styles */
.question-container, .answer-container {
    background-color: white;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Label styles */
.question-label, .answer-label {
    font-weight: bold;
    color: #333;
    margin-bottom: 8px;
    font-size: 1.1em;
}

/* Text styles */
.question-text, .answer-text {
    color: #2c3e50;
    font-size: 1.05em;
    line-height: 1.5;
}

/* Citation styles */
.citation {
    display: inline-block;
    background-color: #e3f2fd;
    color: #1976d2;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
    cursor: pointer;
    margin: 0 2px;
    transition: background-color 0.2s;
}

.citation:hover {
    background-color: #bbdefb;
}

/* Citation container in expander */
.citation-container {
    border-left: 3px solid #1976d2;
    padding: 10px 15px;
    margin: 10px 0;
    background-color: #f8f9fa;
}

.citation-page {
    font-weight: bold;
    color: #1976d2;
    margin-bottom: 5px;
}

.citation-text {
    color: #2c3e50;
    font-size: 0.95em;
    line-height: 1.4;
}

/* Make the main container full width */
.block-container {
    max-width: 100% !important;
    padding: 2rem 5rem !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
<script>
function showFootnote(page) {
    const footnote = document.getElementById('footnote-' + page);
    if (footnote) {
        footnote.scrollIntoView({ behavior: 'smooth' });
        footnote.style.backgroundColor = '#e3f2fd';
        setTimeout(() => {
            footnote.style.backgroundColor = '#f8f9fa';
        }, 1000);
    }
}
</script>
""", unsafe_allow_html=True)
