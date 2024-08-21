import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import hashlib
from fpdf import FPDF
from urllib.parse import quote, unquote 
import streamlit as st
import extra_streamlit_components as stx
from datetime import datetime, timedelta

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]

PDF_UPLOAD_COUNT_FILE = "pdf_upload_count.json"


def load_pdf_upload_count():
    if os.path.exists(PDF_UPLOAD_COUNT_FILE):
        with open(PDF_UPLOAD_COUNT_FILE, "r") as f:
            data = json.load(f)
            return data.get("count", 0)
    else:
        # Initialize the file if it does not exist
        save_pdf_upload_count(0)
        return 0

def save_pdf_upload_count(count):
    with open(PDF_UPLOAD_COUNT_FILE, "w") as f:
        json.dump({"count": count}, f)

def increment_pdf_upload_count():
    count = load_pdf_upload_count()
    count += 1
    save_pdf_upload_count(count)

def get_manager():
    return stx.CookieManager()

def get_cookie_manager():
    if "cookie_manager" not in st.session_state:
        st.session_state.cookie_manager = get_manager()
    return st.session_state.cookie_manager

def set_auth_cookie(email):
    cookie_manager = get_cookie_manager()
    cookie_manager.set('auth_email', email, expires_at=datetime.now() + timedelta(days=30))

def clear_auth_cookie():
    cookie_manager = get_cookie_manager()
    cookie_manager.delete('auth_email')

def get_auth_cookie():
    cookie_manager = get_cookie_manager()
    return cookie_manager.get('auth_email')



genai.configure(api_key=GOOGLE_API_KEY)

# Paths for user data and processed data
USER_DATA_FILE = "user_data.json"
PROCESSED_DATA_FILE = "processed_data.json"

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_data(user_data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(user_data, f)

def load_processed_data():
    if os.path.exists(PROCESSED_DATA_FILE):
        with open(PROCESSED_DATA_FILE, "r") as f:
            return json.load(f)
    return {"explanations": [], "summaries": [], "tips": []}

def save_processed_data(data):
    with open(PROCESSED_DATA_FILE, "w") as f:
        json.dump(data, f)

def check_password(email, password):
    user_data = load_user_data()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    stored_password = user_data.get(email)
    return stored_password and stored_password == hashed_password

def add_user(email, password):
    user_data = load_user_data()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user_data[email] = hashed_password
    save_user_data(user_data)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    try:
        vector_store.save_local("faiss_index")
        
    except Exception as e:
        st.write(f"Failed to save FAISS index: {e}")

def get_extraction_chain(language):
    if language == "Hindi":
        prompt_template = """
नीचे दिए गए पाठ से सबसे महत्वपूर्ण शब्दों की पहचान करें और उन्हें निम्नलिखित प्रारूप में समझाएं: 'महत्वपूर्ण शब्द(रिपोर्ट भाषा) : स्पष्टीकरण(चयनित भाषा)'।'

सामग्री:\n{context}\n
महत्वपूर्ण शब्द और स्पष्टीकरण:
"""

    elif language == "Gujarati":
        prompt_template = """
આ નીચે આપેલ લખાણમાંથી સૌથી મહત્વપૂર્ણ શબ્દોની ઓળખ કરો અને તેમને નીચેના ફોર્મેટમાં સમજાવો: 'મહત્વપૂર્ણ શબ્દ(અહેવાલની ભાષા) : સમજાણું(પસંદ કરેલી ભાષા)'। '

સામગ્રી:\n{context}\n
મહત્વપૂર્ણ શબ્દો અને સમજણું:
"""

    else:  # Default to English
        prompt_template = """
Identify the most important terms from the following text and provide explanations in the simple words.'

Content:\n{context}\n
Important Terms and Explanations:
"""

        
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def extract_and_explain(text_chunks, language):
    chain = get_extraction_chain(language)
    explanations = []
    for chunk in text_chunks:
        response = chain.run({"context": chunk})
        explanations.append(response)
    return explanations

def get_summarization_chain(language):
    if language == "Hindi":
        prompt_template = """
        निम्नलिखित पाठ को संक्षेप में और हिंदी में संक्षेपित करें।
        
        पाठ:\n{context}\n
        संक्षेप:
        """
    elif language == "Gujarati":
        prompt_template = """
        નીચેના લખાણને સંક્ષેપમાં અને ગુજરાતી ભાષામાં સંક્ષિપ્ત કરો.
        લખાણ:\n{context}\n
        સંક્ષેપ:
        """
    else:  # Default to English
        prompt_template = """
        Summarize the following text in a concise manner and in English.
        Summary:
        """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def summarize_text_chunks(text_chunks, language):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    chain = get_summarization_chain(language)
    summaries = []
    for chunk in text_chunks:
        response = chain.run({"context": chunk})
        summaries.append(response)
    return summaries

def get_tips_chain(language):
    if language == "Hindi":
        prompt_template = """
        निम्नलिखित संक्षेप के आधार पर व्यावहारिक सुझाव या सलाह प्रदान करें। 
        संक्षेप:\n{context}\n
        सुझाव:
        """
    elif language == "Gujarati":
        prompt_template = """
        નીચેના સંક્ષેપના આધારે વ્યવહારૂ સલાહ અથવા સૂચનો આપો. 
        
        સંક્ષેપ:\n{context}\n
        સૂચનો:
        """
    else:  # Default to English
        prompt_template = """
        Provide practical tips or advice based on the following summary.
        
        Summary:\n{context}\n
        Tips:
        """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def get_chatbot_chain(language):
    if language == "Hindi":
        prompt_template = """
        निम्नलिखित दस्तावेज़ के आधार पर दिए गए प्रश्न का उत्तर दें।
        
        दस्तावेज़:\n{context}\n
        प्रश्न:\n{question}\n
        उत्तर:
        """
    elif language == "Gujarati":
        prompt_template = """
        નીચેના દસ્તાવેજના આધારે આપવામાં આવેલા પ્રશ્નનો જવાબ આપો.
        
        દસ્તાવેજ:\n{context}\n
        પ્રશ્ન:\n{question}\n
        જવાબ:
        """
    else:  # Default to English
        prompt_template = """
        Answer the following question based on the provided document.
        
        Document:\n{context}\n
        Question:\n{question}\n
        Answer:
        """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def answer_question(context, question, language):
    chain = get_chatbot_chain(language)
    response = chain.run({"context": context, "question": question})
    return response
def get_tips(summary, language):
    chain = get_tips_chain(language)
    response = chain.run({"context": summary})
    return response




def load_user_emails():
    user_data = load_user_data()
    return list(user_data.keys())


def login():
    st.title("Login")
    email = st.text_input("Email", type="default")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if not email or not password:
            st.error("Please enter both email and password.")
        elif check_password(email, password):
            st.session_state["authenticated"] = True
            st.session_state["email"] = email
            set_auth_cookie(email)  # Set the authentication cookie
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid credentials")


def sign_up():
    st.title("Sign Up")
    
    email = st.text_input("Email", type="default")
    password = st.text_input("Password", type="password")
    
    if st.button("Sign Up"):
        if not email or not password:
            st.error("Please enter both email and password.")
        elif email in load_user_emails():
            st.error("Email already exists. Please use a different email.")
        else:
            add_user(email, password)
            st.session_state["authenticated"] = True
            st.session_state["email"] = email
            set_auth_cookie(email)  # Set the authentication cookie
            st.success("Signed up successfully!")
            st.rerun()  # Redirect to app on successful sign-up


def view_users():
    st.title("User Management")
    admin_password = st.text_input("Password", type="password")
    if admin_password == ADMIN_PASSWORD:
        st.write(f"### Total PDF Uploads: {load_pdf_upload_count()}")
        st.write("### Registered Users")
        user_data = load_user_data()
        
        for i in user_data.keys():
            st.write(i)
        

def main():
    st.set_page_config(
        page_title="Wellness",
        page_icon="📜",
        layout="wide"
    )
    st.image("logo.png", use_column_width=True)
    
    # Check for authentication cookie
    auth_email = get_auth_cookie()
    if auth_email:
        st.session_state["authenticated"] = True
        st.session_state["email"] = auth_email
    
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        choice = st.sidebar.radio("Select an Option", ["Login", "Sign Up", "View Users"], key="auth_choice")
        
        if choice == "Login":
            login()
        elif choice == "Sign Up":
            sign_up()
        elif choice == "View Users":
            view_users()
    else:
        st.sidebar.title("User Options")
        st.sidebar.write(f"Logged in as: {st.session_state['email']}")
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state["authenticated"] = False
            st.session_state.pop("email", None)
            clear_auth_cookie()
            st.query_params.clear()
            st.rerun()
        
        # Rest of your app code here...
        st.title("Welcome to Wellness")
        
        # Language selection
        language = st.sidebar.selectbox("Select Language", ["English", "Hindi", "Gujarati"])
        
        # PDF Upload
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if pdf_docs:
            # Increment the PDF upload count
            increment_pdf_upload_count()

            st.sidebar.write("### PDF Upload Status: Files Uploaded")
            st.sidebar.write("### Options")
            selected_option = st.sidebar.radio(
                "Select an Option",
                ["Summary and Tips", "Important Terms", "Chatbot"]
            )

            if selected_option == "Summary and Tips":
                if st.button("Generate Summary and Tips"):
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)  # Save chunks to vector store

                        # Display summaries incrementally
                        st.write("### Summary")
                        summaries_placeholder = st.empty()
                        summaries = []
                        for chunk in text_chunks:
                            chunk_summaries = summarize_text_chunks([chunk], language)
                            for summary in chunk_summaries:
                                summaries_placeholder.write(summary)
                                summaries.extend(chunk_summaries)

                        # Display tips incrementally
                        st.write("### Tips Based on Summary")
                        tips_placeholder = st.empty()
                        summaries_text = " ".join(summaries)
                        tips = get_tips(summaries_text, language)
                        tips_placeholder.write(tips)
                        
                        # Save the processed data for download
                        save_processed_data({"summaries": summaries, "tips": tips})
                        
                        st.success("Processing complete!")

                       
            elif selected_option == "Important Terms":
                if st.button("Generate Important Terms"):
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)  # Save chunks to vector store

                        # Display explanations incrementally
                        st.write("### Important Terms and Explanations")
                        explanations_placeholder = st.empty()
                        for chunk in text_chunks:
                            explanations = extract_and_explain([chunk], language)
                            for explanation in explanations:
                                explanations_placeholder.write(explanation)
                        
                        # Save the processed data for download
                        save_processed_data({"explanations": explanations})
                        
                        st.success("Processing complete!")

                       

            elif selected_option == "Chatbot":
                if pdf_docs:
                    with st.spinner("Processing PDF..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)  # Save chunks to vector store

                    st.write("PDF processed. You can now ask questions.")
                    
                    question = st.text_input("Your Question", key="chatbot_question")
                    if question:
                        if st.button("Ask", key="ask_button"):
                            with st.spinner("Searching for answer..."):
                                answer = answer_question(raw_text, question, language)
                                st.write("### Answer")
                                st.write(answer)
                else:
                    st.warning("Please upload a PDF document first.")
        
        
        

if __name__ == "__main__":
    main()
