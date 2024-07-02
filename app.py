import os
import requests
from flask import Flask, request, jsonify, render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import warnings
from googletrans import Translator
import google.generativeai as genai
from docx import Document
from sentence_transformers import SentenceTransformer
import torch
from googleapiclient.discovery import build
import re

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime




warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_question = db.Column(db.String(500), nullable=False)
    answer = db.Column(db.String(500), nullable=False)
    feedback = db.Column(db.String(1000), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

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

# Get answer from FAQ data
def get_faq_suggestions(user_query, questions, question_embeddings, model, top_k=5):
    user_query_embedding = model.encode(user_query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(user_query_embedding, question_embeddings)
    top_k_indices = torch.topk(cos_scores, k=top_k).indices.tolist()
    
    return [(questions[idx], cos_scores[idx].item()) for idx in top_k_indices]

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

def load_filters():
    filters = set()
    with open('filter_words.txt', 'r', encoding='utf-8') as f:
        for line in f:
            filters.add(line.strip())
    return filters

filters = load_filters()

def contains_filtered_content(text):
    for filter_word in filters:
        if filter_word in text.lower():
            return True
    return False

# def is_valid_title(title):
#     # Check if title contains only alphanumeric characters and spaces
#     return re.match("^[a-zA-Z0-9\s]*$", title) is not None


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
        # faq_file_path = 'OPERATIONAL_GUIDELINES_JJM.docx'
        faq_file_path = 'test.docx'

        
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
        suggested_questions = [q for q, _ in suggestions]
        translated_suggestions = [translate(q, selected_language) for q in suggested_questions]
        
        # Perform YouTube search for related videos
        youtube_videos = youtube_search(user_question, selected_language)
        
        return jsonify({
            "answer": translated_answer,
            "available": False,
            "faq_available": True,
            "suggestions": translated_suggestions,
            "videos": youtube_videos
        })
    else:
        # Perform YouTube search for related videos
        youtube_videos = youtube_search(user_question, selected_language)
        
        return jsonify({
            "answer": translated_answer,
            "available": True,
            "faq_available": False,
            "videos": youtube_videos
        })

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    user_question = data['user_question']
    answer = data['answer']
    feedback = data['feedback']

    new_feedback = Feedback(user_question=user_question, answer=answer, feedback=feedback)
    db.session.add(new_feedback)
    db.session.commit()

    return jsonify({"status": "success", "message": "Feedback submitted successfully"})


def youtube_search(query: str, language: str):
    api_key = os.getenv('YOUTUBE_API_KEY')
    search_url = "https://www.googleapis.com/youtube/v3/search"
    video_url = "https://www.googleapis.com/youtube/v3/videos"
    translator = Translator()
    
    translated_query = translator.translate(query, dest=language).text + " Jal jeevan mission "
    
    search_params = {
        'part': 'snippet',
        'q': translated_query,
        'key': api_key,
        'maxResults': 5,
        'type': 'video'
    }
    
    search_response = requests.get(search_url, params=search_params)
    search_results = search_response.json()

    video_ids = [item['id']['videoId'] for item in search_results['items']]
    
    video_params = {
        'part': 'snippet,statistics',
        'id': ','.join(video_ids),
        'key': api_key
    }
    
    video_response = requests.get(video_url, params=video_params)
    video_results = video_response.json()
    
    videos = []
    for item in video_results.get('items', []):
        title = translator.translate(item['snippet']['title'], dest=language).text
        # Check if title contains filtered content
        # if is_valid_title(title):
        if not contains_filtered_content(title):
            video_data = {
                'title': title,
                'url': f"https://www.youtube.com/watch?v={item['id']}",
            }
            videos.append(video_data)
    
    # Sort videos by views, likes, and comments
    return videos


@app.route('/faq_answer', methods=['POST'])
def faq_answer():
    data = request.get_json()
    user_question = data['question']
    selected_language = data['language']
    
    # Get answer from FAQ data
    answer = get_answer(user_question, questions, answers, question_embeddings, model)
    translated_answer = translate(answer, selected_language)
    
    return jsonify({"answer": translated_answer})

@app.route('/youtube_search', methods=['POST'])
def youtube_search_route():
    data = request.get_json()
    question = data.get('question', '')
    language = data.get('language', 'en')
    youtube_videos = youtube_search(question, language)
    return jsonify({'youtube_videos': youtube_videos})



# Error handling routes    
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
