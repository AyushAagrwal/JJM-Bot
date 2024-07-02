import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import warnings
from googletrans import Translator
import google.generativeai as genai
from docx import Document
from sentence_transformers import SentenceTransformer
import torch

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Extract FAQ from DOCX
def extract_faq_from_docx(file_path):
    doc = Document(file_path)
    faq_dict = {}
    current_question = None
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if text.endswith('?'):  # Assuming questions end with a question mark
            current_question = text
            faq_dict[current_question] = ""
        elif current_question:
            faq_dict[current_question] += text + " "
    
    # Strip trailing spaces from answers
    for question in faq_dict:
        faq_dict[question] = faq_dict[question].strip()
    
    return faq_dict

# Preprocess FAQ data
def preprocess_faq(faq_data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    questions = list(faq_data.keys())
    answers = [faq_data[q] for q in questions]
    
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    
    return questions, answers, question_embeddings, model

# Load and preprocess FAQ data
faq_file_path = 'OPERATIONAL_GUIDELINES_JJM.docx'
faq_data = extract_faq_from_docx(faq_file_path)
questions, answers, question_embeddings, model = preprocess_faq(faq_data)

# Get suggested questions from FAQ data
def get_faq_suggestions(user_query, questions, question_embeddings, model, top_k=5):
    user_query_embedding = model.encode(user_query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(user_query_embedding, question_embeddings)
    top_k_indices = torch.topk(cos_scores, k=top_k).indices.tolist()
    
    suggestions = []
    for idx in top_k_indices:
        question = questions[idx]
        if question in faq_data:
            suggestions.append((question, cos_scores[idx].item()))
    
    return suggestions

# Get answer from FAQ data
def get_answer(user_query, questions, answers, question_embeddings, model):
    user_query_embedding = model.encode(user_query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(user_query_embedding, question_embeddings)
    best_match_idx = torch.argmax(cos_scores).item()
    
    return answers[best_match_idx]

# Translation function using 'googletrans' library
def translate(text, target_language):
    translator = Translator()
    try:
        translated_text = translator.translate(text, dest=target_language).text
        return translated_text
    except Exception as e:
        return text

# Load text file and return content
def get_text_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return None

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    return splitter.split_text(text)

# Get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create a conversational chain
def get_conversational_chain():
    prompt_template = """
    You are an expert on the JJM Operational Guidelines. Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "The answer is not available in the context." Do not provide a wrong answer.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", client=genai, temperature=0.8, max_tokens=1000, top_p=0.98, top_k=50, stop_sequences=["\n"])
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# Global initialization flag
initialized = False

@app.before_request
def before_request():
    global initialized
    if not initialized:
        file_path = "JJM_Operational_Guidelines.txt"
        faq_file_path = 'OPERATIONAL_GUIDELINES_JJM.docx'
        
        raw_text = get_text_content(file_path)
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
        
        # Load and preprocess FAQ data
        faq_data = extract_faq_from_docx(faq_file_path)
        global questions, answers, question_embeddings, model
        questions, answers, question_embeddings, model = preprocess_faq(faq_data)
        
        initialized = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data['question']
    selected_language = data['language']

    # Attempt to get answer using Gemini API
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response['output_text']
    
    translated_answer = translate(answer, selected_language)
    
    if "The answer is not available in the context." in answer:
        # Get FAQ suggestions
        suggestions = get_faq_suggestions(user_question, questions, question_embeddings, model)
        translated_suggestions = []
        for idx, (q, _) in enumerate(suggestions, start=1):
            translated_q = translate(q, selected_language)
            translated_suggestions.append(f"{idx}. {translated_q}")
        return jsonify({"answer": translated_answer, "available": False, "faq_available": True, "suggestions": translated_suggestions})
    else:
        return jsonify({"answer": translated_answer, "available": True, "faq_available": False})

@app.route('/faq_answer', methods=['POST'])
def faq_answer():
    data = request.get_json()
    user_question = data['question']
    selected_language = data['language']
    
    # Get answer from FAQ data
    answer = get_answer(user_question, questions, answers, question_embeddings, model)
    translated_answer = translate(answer, selected_language)
    
    return jsonify({"answer": translated_answer})

@app.route('/google_search', methods=['POST'])
def google_search():
    data = request.get_json()
    query = data['query']
    selected_language = data['language']

    max_query_length = 500  # Maximum allowed query length
    if len(query) > max_query_length:
        query = query[:max_query_length]
        
    try:
        headers = {
            "User-Agent": "Chrome/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(f"https://www.google.com/search?q={query}", headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd', limit=2)
        
        results_set = set()
        for result in search_results:
            text = result.text.strip()
            if text and text not in results_set:  # Ensure the result is unique
                translated_result = translate(text, selected_language)
                results_set.add(translated_result)
        
        results = list(results_set)
        
        if results:
            return jsonify({"answers": results})
        else:
            return jsonify({"answers": ["No results found."]})
    except Exception as e:
        return jsonify({"answer": f"Error occurred during search: {e}"})

# Error handling routes    
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
