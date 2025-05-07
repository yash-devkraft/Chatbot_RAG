import os
import sys
import re
import PyPDF2
import docx
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import traceback
# Updated RAG-related imports
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Using community vectorstores
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    return txt_file.read().decode('utf-8')


def extract_text(file):
    """Extract text from uploaded file based on file type"""
    file_name = file.name.lower()
    if file_name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file_name.endswith('.docx'):
        return extract_text_from_docx(file)
    elif file_name.endswith('.txt'):
        return extract_text_from_txt(file)
    else:
        st.error("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
        return None


def chunk_text(text, max_chunk_size=8000):
    """Split text into chunks of maximum size to fit within token limits"""
    chunks = []
    current_chunk = ""
    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = paragraph + "\n"
        else:
            current_chunk += paragraph + "\n"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# RAG-specific functions
def create_vector_store(text):
    """Create a vector store from the document text using FAISS"""
    try:
        # Create a text splitter for more effective chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # Split text into chunks
        chunks = text_splitter.split_text(text)

        # Create metadata for each chunk to trace them back to the document
        texts = chunks
        metadatas = [{"source": f"chunk_{i}"} for i in range(len(chunks))]

        # Initialize the embedding model (using a free, local model from HuggingFace)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create the vector store
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

        return vector_store

    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        st.error(traceback.format_exc())
        return None


def setup_rag_chain(vector_store):
    """Set up the RAG chain with the vector store"""
    try:
        # Create a memory object to store conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Specify which output to store in memory
        )

        # Create the LLM using Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            top_p=0.95,
            top_k=40,  # Use a positive value
            max_output_tokens=1024,
        )

        # Create the conversational retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=True  # Add this for debugging
        )

        return chain

    except Exception as e:
        st.error(f"Error setting up RAG chain: {str(e)}")
        st.error(traceback.format_exc())
        return None


def rag_answer_question(rag_chain, question):
    """Answer a question using the RAG chain"""
    try:
        # Query the RAG chain
        response = rag_chain({"question": question})

        # Extract the answer and source documents
        answer = response.get("answer", "")
        source_docs = response.get("source_documents", [])

        # Format sources for display
        sources = []
        for i, doc in enumerate(source_docs):
            source = doc.metadata.get("source", f"Source {i + 1}")
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            sources.append(f"**{source}**: {content_preview}")

        return answer, sources

    except Exception as e:
        st.error(f"Error answering question with RAG: {str(e)}")
        st.error(traceback.format_exc())
        return "Error processing your question. Please try again.", []


def generate_summary(text):
    """Generate a summary of the document using Gemini API"""
    try:
        chunks = chunk_text(text, max_chunk_size=8000)
        summary = ""

        generation_config = {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 1024,
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
        )

        for i, chunk in enumerate(chunks):
            prompt = f"Please summarize the following text concisely:\n\n{chunk}"

            response = model.generate_content(prompt)
            chunk_summary = response.text.strip()
            summary += chunk_summary + " "

        return summary.strip()

    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        st.error(traceback.format_exc())
        return None


def generate_bullet_points(text):
    """Generate bullet points from the document using Gemini API"""
    try:
        chunks = chunk_text(text)
        all_bullet_points = []

        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 1024,
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
        )

        for i, chunk in enumerate(chunks):
            try:
                prompt = f"""
                Extract 5-10 key points from the following text as bullet points.
                Format each point with a bullet (•) at the beginning.
                Make sure each point is clear and concise.

                IMPORTANT: Do not include any introductory text like "Here are key points" or any headers.
                Start directly with the bullet points themselves.

                Text:
                {chunk}
                """

                response = model.generate_content(prompt)
                chunk_bullets = response.text.strip()

                st.session_state.debug_output = st.session_state.get('debug_output', []) + [
                    f"API response for chunk {i + 1}: {chunk_bullets}"]

                all_bullet_points.append(chunk_bullets)

            except Exception as chunk_error:
                st.error(f"Error processing chunk {i + 1}: {str(chunk_error)}")
                st.error(traceback.format_exc())

        # Join all bullet points and clean up
        combined_bullets = "\n".join(all_bullet_points)

        # Process the combined bullets
        processed_bullets = []

        # Remove common introductory lines and duplicates
        seen_lines = set()

        for line in combined_bullets.split('\n'):
            line = line.strip()
            if not line:
                continue

            if any(line.lower().startswith(prefix) for prefix in [
                "here are", "key points", "points extracted", "following are"
            ]):
                continue

            if not line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                line = '• ' + line

            if line in seen_lines:
                continue

            seen_lines.add(line)
            processed_bullets.append(line)

        final_output = "Key Points:\n\n" + "\n".join(processed_bullets)

        if not processed_bullets:
            return combined_bullets

        return final_output

    except Exception as e:
        st.error(f"Error generating bullet points: {str(e)}")
        st.error(traceback.format_exc())
        return None


def main():
    # Set page configuration
    st.set_page_config(page_title="Document Q&A Bot", page_icon="📄", layout="wide")

    # Apply custom CSS styling
    st.markdown("""
    <style>
    /* Main Style Overrides */
    .main {
        background-color: #f8f9fa;
    }

    /* Custom Containers */
    .css-1kyxreq.e115fcil2, .css-12oz5g7.e1tzin5v2 {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Headers */
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }

    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }

    /* Button styling */
    .stButton button {
        background-color: #4285F4;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: #3367d6;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* File uploader */
    .uploadedFile {
        border: 2px dashed #4285F4;
        border-radius: 5px;
        padding: 10px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4285F4;
        color: white;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #4285F4;
    }

    /* Response containers */
    .response-container {
        background-color: #e8f0fe;
        border-left: 5px solid #4285F4;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* Source containers */
    .source-container {
        background-color: #f0f2f6;
        border-left: 5px solid #80868b;
        padding: 10px;
        border-radius: 5px;
        margin-top: 5px;
        margin-bottom: 10px;
        font-size: 0.9em;
    }

    /* Text input */
    .stTextInput input {
        border-radius: 20px;
        padding: 10px 15px;
        border: 1px solid #dadce0;
    }

    /* Message boxes */
    .info-box {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #2196f3;
    }

    .success-box {
        background-color: #e8f5e9;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #4caf50;
    }

    .warning-box {
        background-color: #fff3e0;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #ff9800;
    }

    .error-box {
        background-color: #ffebee;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #f44336;
    }
    </style>
    """, unsafe_allow_html=True)

    # Add a sidebar for debug information
    st.sidebar.title("Debug Information")

    # Initialize session state variables if they don't exist
    if 'document_text' not in st.session_state:
        st.session_state.document_text = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None

    # App Header with custom styling
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 30px 0;">
        <h1 style="font-size: 42px;">📄 Document Q&A Bot</h1>
        <p style="font-size: 18px; color: #5f6368; margin-top: -10px;">Upload a document, get a summary, bullet points, and ask questions about its content.</p>
    </div>
    """, unsafe_allow_html=True)

    if not GEMINI_API_KEY:
        st.error("API key not found. Please set the GEMINI_API_KEY environment variable.")
        return

    # File uploader with custom styling
    st.markdown('<div class="uploadedFile">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Extract text from the document
        with st.spinner("Extracting text from document..."):
            document_text = extract_text(uploaded_file)

        if document_text:
            st.session_state.document_text = document_text
            st.sidebar.write(f"Document length: {len(document_text)} characters")

            # Create RAG components if they don't exist
            if st.session_state.vector_store is None:
                with st.spinner("Setting up smart search system..."):
                    st.session_state.vector_store = create_vector_store(document_text)

                if st.session_state.vector_store:
                    st.session_state.rag_chain = setup_rag_chain(st.session_state.vector_store)
                    st.sidebar.write("✅ Vector store created successfully")
                    st.sidebar.write("✅ Q&A system initialized successfully")

                    st.markdown(f"""
                    <div class="success-box">
                        <h3 style="color: #4caf50; margin-top: 0;">✅ System Ready!</h3>
                        <p>Successfully extracted text from <b>{uploaded_file.name}</b> and prepared for smart search.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.sidebar.write("❌ Failed to create vector store")
                    st.error("Failed to create search index. Please try uploading the document again.")

            st.markdown(f"""
            <div class="success-box">
                <h3 style="color: #4caf50; margin-top: 0;">✅ Success!</h3>
                <p>Successfully extracted text from <b>{uploaded_file.name}</b></p>
            </div>
            """, unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["Q&A", "Summary", "Bullet Points"])

            # Tab 1: Q&A with RAG
            with tab1:
                st.markdown(
                    '<h2 style="text-align: center; color: #4285F4; margin-bottom: 20px;">Ask questions about the document</h2>',
                    unsafe_allow_html=True)

                # Add info message without mentioning RAG
                st.markdown("""
                <div class="info-box">
                    <p style="margin: 0;"><b>Smart Search Active:</b> Using semantic search for accurate answers.</p>
                </div>
                """, unsafe_allow_html=True)

                question = st.text_input("Enter your question:", key="question_input")

                if question:
                    if st.session_state.rag_chain:
                        with st.spinner("Processing your question..."):
                            answer, sources = rag_answer_question(st.session_state.rag_chain, question)

                        st.markdown("""
                        <div class="response-container">
                            <h3 style="color: #4285F4; margin-top: 0;">Answer:</h3>
                            <p>{}</p>
                        </div>
                        """.format(answer.replace("\n", "<br>")), unsafe_allow_html=True)

                        # Only show relevant document sections if they exist and are meaningful
                        if sources and any(not s.endswith("...") for s in sources):
                            st.markdown("<h4>Relevant Document Sections:</h4>", unsafe_allow_html=True)
                            for source in sources:
                                st.markdown(f"""
                                <div class="source-container">
                                    {source}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("Search system is not initialized. Please try uploading the document again.")

            # Tab 2: Summary
            with tab2:
                st.markdown(
                    '<h2 style="text-align: center; color: #4285F4; margin-bottom: 20px;">Document Summary</h2>',
                    unsafe_allow_html=True)

                if st.button("Generate Summary", key="gen_summary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(document_text)

                    if summary:
                        st.markdown("""
                           <div class="response-container">
                               <h3 style="color: #4285F4; margin-top: 0;">Summary:</h3>
                               <p>{}</p>
                           </div>
                           """.format(summary.replace("\n", "<br>")), unsafe_allow_html=True)

            # Tab 3: Bullet Points
            with tab3:
                st.markdown('<h2 style="text-align: center; color: #4285F4; margin-bottom: 20px;">Key Points</h2>',
                            unsafe_allow_html=True)

                if st.button("Extract Key Points", key="extract_key_points"):
                    with st.spinner("Extracting key points..."):
                        bullet_points = generate_bullet_points(document_text)

                    if bullet_points:
                        st.markdown("""
                           <div class="response-container">
                               <h3 style="color: #4285F4; margin-top: 0;">Key Points:</h3>
                               <div>{}</div>
                           </div>
                           """.format(bullet_points.replace("\n", "<br>").replace("• ", "• <b>").replace("•", "•</b>")),
                                    unsafe_allow_html=True)
                    else:
                        st.markdown("""
                           <div class="error-box">
                               <h3 style="color: #f44336; margin-top: 0;">Error</h3>
                               <p>Failed to extract key points. Please try again.</p>
                           </div>
                           """, unsafe_allow_html=True)
    else:
        # Display welcome message and instructions when no file is uploaded
        st.markdown("""
        <div style="background-color: #e8f0fe; padding: 20px; border-radius: 10px; text-align: center; margin: 30px 0;">
            <h2 style="color: #4285F4; margin-top: 0;">Welcome to Document Q&A Bot!</h2>
            <p style="font-size: 18px;">Upload a document using the file uploader above to get started.</p>
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <div style="text-align: center; padding: 15px; margin: 0 10px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 200px;">
                    <h3 style="color: #4285F4;">Ask Questions</h3>
                    <p>Get accurate answers about your document content</p>
                </div>
                <div style="text-align: center; padding: 15px; margin: 0 10px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 200px;">
                    <h3 style="color: #4285F4;">Generate Summary</h3>
                    <p>Get a concise summary of your document content</p>
                </div>
                <div style="text-align: center; padding: 15px; margin: 0 10px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 200px;">
                    <h3 style="color: #4285F4;">Extract Key Points</h3>
                    <p>Get the most important points as bullet points</p>
                </div>
            </div>
        </div>

        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 30px 0;">
            <h3 style="color: #4285F4;">How This App Works:</h3>
            <ol>
                <li><b>Document Processing:</b> Your document is converted into text and analyzed for content</li>
                <li><b>Smart Search:</b> When you ask a question, the system finds the most relevant information</li>
                <li><b>AI-Powered Answers:</b> The system generates accurate, context-aware responses to your questions</li>
                <li><b>Helpful Context:</b> When relevant, you'll see document sections that support the answers</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()