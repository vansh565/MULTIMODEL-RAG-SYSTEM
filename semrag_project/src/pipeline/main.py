from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from pypdf import PdfReader
from src.chunking.semantic_chunker import semantic_chunk
from src.graph.graph_builder import build_graph
from src.graph.community_detector import detect_communities
from src.retrieval.local_search import local_search
from src.retrieval.global_search import global_search
from src.llm.answer_generator import generate_answer
import threading
import time
import uuid
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Required for session
CORS(app)

# Global variables to store system state
system_state = {
    'ready': False,
    'graph': None,
    'chunks': None,
    'communities': None,
    'loading': False,
    'status_message': ''
}

# Store chat histories for different sessions
chat_sessions = defaultdict(lambda: {
    'messages': [],  # List of {'role': 'user/assistant', 'content': str, 'timestamp': str}
    'created_at': datetime.now(),
    'last_updated': datetime.now()
})

@app.route('/api/chat/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions"""
    sessions = []
    for session_id, data in chat_sessions.items():
        sessions.append({
            'session_id': session_id,
            'created_at': data['created_at'].isoformat(),
            'last_updated': data['last_updated'].isoformat(),
            'message_count': len(data['messages'])
        })
    return jsonify({'sessions': sorted(sessions, key=lambda x: x['last_updated'], reverse=True)})

@app.route('/api/chat/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get specific chat session"""
    if session_id not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify({
        'session_id': session_id,
        'messages': chat_sessions[session_id]['messages'],
        'created_at': chat_sessions[session_id]['created_at'].isoformat(),
        'last_updated': chat_sessions[session_id]['last_updated'].isoformat()
    })

@app.route('/api/chat/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return jsonify({'success': True, 'message': 'Session deleted'})
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/chat/clear', methods=['POST'])
def clear_current_session():
    """Clear current session's chat history"""
    session_id = request.json.get('session_id', 'default')
    if session_id in chat_sessions:
        chat_sessions[session_id]['messages'] = []
        chat_sessions[session_id]['last_updated'] = datetime.now()
    return jsonify({'success': True, 'message': 'Chat history cleared'})

def get_chat_context(session_id, max_history=10):
    """Get recent chat history as formatted context"""
    if session_id not in chat_sessions:
        return ""
    
    # Get last N messages (max_history)
    recent_messages = chat_sessions[session_id]['messages'][-max_history:]
    
    if not recent_messages:
        return ""
    
    context = "Previous conversation:\n"
    for msg in recent_messages:
        role = "User" if msg['role'] == 'user' else "Assistant"
        context += f"{role}: {msg['content']}\n"
    context += "\nBased on the above conversation, please answer the following question:\n"
    
    return context

def initialize_system():
    """Initialize the system in a background thread"""
    system_state['loading'] = True
    system_state['status_message'] = 'Loading PDF...'
    time.sleep(0.5)
    
    text = load_pdf("data/IPC.pdf")
    
    system_state['status_message'] = 'Chunking...'
    time.sleep(0.5)
    chunks = semantic_chunk(text)
    system_state['chunks'] = chunks
    
    system_state['status_message'] = 'Building graph...'
    time.sleep(0.5)
    graph = build_graph(chunks)
    system_state['graph'] = graph
    
    system_state['status_message'] = 'Detecting communities...'
    time.sleep(0.5)
    communities = detect_communities(graph)
    system_state['communities'] = communities
    
    system_state['ready'] = True
    system_state['loading'] = False
    system_state['status_message'] = 'System Ready!'

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() + "\n"
        except:
            pass
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify({
        'ready': system_state['ready'],
        'loading': system_state['loading'],
        'message': system_state['status_message']
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    if not system_state['ready']:
        return jsonify({'error': 'System not ready yet'}), 503
    
    data = request.json
    question = data.get('question', '')
    session_id = data.get('session_id', 'default')
    use_history = data.get('use_history', True)  # Option to use history or not
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Get chat context if history is enabled
    chat_context = ""
    if use_history:
        chat_context = get_chat_context(session_id, max_history=10)
    
    # Combine question with context if available
    enhanced_question = question
    if chat_context:
        enhanced_question = f"{chat_context}\nCurrent Question: {question}"
    
    # Perform searches
    local_res = local_search(enhanced_question, system_state['graph'], system_state['chunks'])
    global_res = global_search(enhanced_question, system_state['communities'], system_state['chunks'])
    
    # Generate answer with context
    answer = generate_answer(enhanced_question, local_res, global_res)
    
    # Store the conversation
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            'messages': [],
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
    
    # Add user question and assistant answer to history
    chat_sessions[session_id]['messages'].append({
        'role': 'user',
        'content': question,
        'timestamp': datetime.now().isoformat()
    })
    
    chat_sessions[session_id]['messages'].append({
        'role': 'assistant',
        'content': answer,
        'timestamp': datetime.now().isoformat()
    })
    
    chat_sessions[session_id]['last_updated'] = datetime.now()
    
    return jsonify({
        'question': question,
        'answer': answer,
        'local_results': local_res,
        'global_results': global_res,
        'session_id': session_id,
        'context_used': bool(chat_context)
    })

@app.route('/api/chat/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Get chat history for a specific session"""
    if session_id not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify({
        'session_id': session_id,
        'messages': chat_sessions[session_id]['messages']
    })

if __name__ == "__main__":
    # Start system initialization in background
    init_thread = threading.Thread(target=initialize_system)
    init_thread.start()
    
    # Run Flask app
    app.run(debug=True, port=5000)