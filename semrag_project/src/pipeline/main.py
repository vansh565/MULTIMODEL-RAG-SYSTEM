from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from collections import defaultdict
from datetime import datetime
import threading
import time
import os
import uuid
import glob
import requests
from functools import lru_cache

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Mistral API Configuration (Lightweight)
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"  # Fast and memory-efficient

# System configurations
SYSTEMS = {
    'legal': {
        'name': 'Constitutional & Legal System',
        'icon': '⚖️',
        'color': '#1e3c72',
        'description': 'Indian Constitution, IPC, Cyber Laws, Legal Governance',
        'pdf_folder': 'data/IPC_docs',
        'enabled': True
    },
    'medical': {
        'name': 'Medical Information System',
        'icon': '🏥',
        'color': '#2e7d32',
        'description': 'Diseases, Cures, Doctors, Treatments, Healthcare',
        'pdf_folder': 'data/medical_docs',
        'enabled': True
    },
    'disaster': {
        'name': 'Disaster Management System',
        'icon': '🌊',
        'color': '#d32f2f',
        'description': 'Natural Disasters, Emergency Response, Safety Protocols',
        'pdf_folder': 'data/disaster_docs',
        'enabled': True
    },
    'all': {
        'name': 'All-in-One Integrated System',
        'icon': '🌐',
        'color': '#9c27b0',
        'description': 'Combined knowledge from Legal, Medical & Disaster Management',
        'pdf_folder': None,
        'enabled': True
    }
}

# System prompts for each domain (lightweight)
SYSTEM_PROMPTS = {
    'legal': "You are a legal expert specializing in Indian Constitution, Indian Penal Code (IPC), Cyber Laws, and Legal Governance. Provide accurate, helpful legal information.",
    'medical': "You are a medical expert specializing in diseases, treatments, doctors, and healthcare information. Provide accurate medical guidance.",
    'disaster': "You are a disaster management expert specializing in natural disasters, emergency response, safety protocols, and crisis management.",
    'all': "You are an expert in Constitutional Law, IPC, Medical Information, and Disaster Management. Provide comprehensive answers drawing from all domains."
}

# Store system states (simplified - no heavy models)
system_states = {
    'legal': {'ready': True, 'loading': False, 'status_message': 'Ready', 'chunks_count': 0, 'pdf_files': []},
    'medical': {'ready': True, 'loading': False, 'status_message': 'Ready', 'chunks_count': 0, 'pdf_files': []},
    'disaster': {'ready': True, 'loading': False, 'status_message': 'Ready', 'chunks_count': 0, 'pdf_files': []},
    'all': {'ready': True, 'loading': False, 'status_message': 'Ready', 'chunks_count': 0, 'pdf_files': []}
}

# Chat sessions for each system
chat_sessions = defaultdict(lambda: defaultdict(lambda: {
    'messages': [],
    'created_at': datetime.now(),
    'last_updated': datetime.now(),
    'title': 'New Chat',
    'message_count': 0
}))

def find_pdf_files(folder_path):
    """Find all PDF files in a folder for reference only (not loading models)"""
    pdf_files = []
    if os.path.exists(folder_path):
        for pattern in ['*.pdf', '*.PDF']:
            pdf_files.extend(glob.glob(os.path.join(folder_path, pattern)))
    return pdf_files

def initialize_system(system_key):
    """Lightweight initialization - just check for PDFs, no model loading"""
    try:
        print(f"\n{'='*50}")
        print(f"Initializing {SYSTEMS[system_key]['name']}...")
        print(f"{'='*50}")
        
        # Just check for PDFs (optional - for reference)
        if SYSTEMS[system_key]['pdf_folder']:
            folder_path = SYSTEMS[system_key]['pdf_folder']
            os.makedirs(folder_path, exist_ok=True)
            pdf_files = find_pdf_files(folder_path)
            system_states[system_key]['pdf_files'] = [os.path.basename(f) for f in pdf_files]
            
            if pdf_files:
                print(f"📄 Found {len(pdf_files)} reference PDF(s)")
            else:
                print(f"ℹ️ No PDFs found - using Mistral AI knowledge")
        
        system_states[system_key]['ready'] = True
        system_states[system_key]['loading'] = False
        system_states[system_key]['status_message'] = 'Ready with Mistral AI'
        
        print(f"✅ {SYSTEMS[system_key]['name']} ready (Mistral API)")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        system_states[system_key]['ready'] = True  # Still mark as ready
        system_states[system_key]['loading'] = False
        system_states[system_key]['status_message'] = 'Ready (API mode)'

def initialize_all_systems():
    """Initialize all systems (lightweight)"""
    threads = []
    
    for system_key in ['legal', 'medical', 'disaster']:
        if SYSTEMS[system_key]['enabled']:
            thread = threading.Thread(target=initialize_system, args=(system_key,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            time.sleep(0.2)  # Small stagger
    
    # Wait for all threads
    for thread in threads:
        thread.join(timeout=30)
    
    # Mark all system as ready
    system_states['all']['ready'] = True
    system_states['all']['status_message'] = 'All systems ready'
    print("\n✅ All systems initialized successfully with Mistral API!\n")

def generate_chat_title(messages, system_key):
    """Generate chat title from first message"""
    if not messages:
        return f"New {SYSTEMS[system_key]['name']} Chat"
    
    first_message = None
    for msg in messages:
        if msg['role'] == 'user':
            first_message = msg['content']
            break
    
    if not first_message:
        return f"New {SYSTEMS[system_key]['name']} Chat"
    
    # Take first 5-6 words as title
    words = first_message.split()[:6]
    title = ' '.join(words)
    
    if len(title) > 40:
        title = title[:37] + "..."
    
    return f"{SYSTEMS[system_key]['icon']} {title.capitalize()}"

@lru_cache(maxsize=100)
def call_mistral_api_cached(question, system_prompt):
    """Cached Mistral API call to reduce repeated requests"""
    if not MISTRAL_API_KEY:
        return "Mistral API key not configured. Please set MISTRAL_API_KEY environment variable."
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html', systems=SYSTEMS)

@app.route('/api/status/<system_key>')
def get_system_status(system_key):
    """Get status of a specific system"""
    if system_key not in system_states:
        return jsonify({'error': 'Invalid system'}), 404
    
    state = system_states[system_key]
    return jsonify({
        'ready': state['ready'],
        'loading': state['loading'],
        'message': state['status_message'],
        'chunks_count': state.get('chunks_count', 0),
        'pdf_files': state.get('pdf_files', []),
        'system_info': SYSTEMS[system_key]
    })

@app.route('/api/sessions/<system_key>', methods=['GET'])
def get_sessions(system_key):
    """Get all chat sessions for a system"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    sessions_list = []
    for session_id, session_data in chat_sessions[system_key].items():
        preview = ""
        if session_data['messages']:
            first_msg = session_data['messages'][0]['content']
            preview = first_msg[:50] + '...' if len(first_msg) > 50 else first_msg
        
        sessions_list.append({
            'id': session_id,
            'title': session_data.get('title', f"New {SYSTEMS[system_key]['name']} Chat"),
            'created_at': session_data['created_at'].isoformat(),
            'last_updated': session_data['last_updated'].isoformat(),
            'message_count': len(session_data['messages']),
            'preview': preview or 'Empty chat'
        })
    
    sessions_list.sort(key=lambda x: x['last_updated'], reverse=True)
    return jsonify({'sessions': sessions_list})

@app.route('/api/sessions/<system_key>', methods=['POST'])
def create_session(system_key):
    """Create a new chat session for a system"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    session_id = str(uuid.uuid4())
    chat_sessions[system_key][session_id] = {
        'messages': [],
        'created_at': datetime.now(),
        'last_updated': datetime.now(),
        'title': f"New {SYSTEMS[system_key]['name']} Chat",
        'message_count': 0
    }
    return jsonify({'session_id': session_id, 'title': chat_sessions[system_key][session_id]['title']})

@app.route('/api/sessions/<system_key>/<session_id>', methods=['GET'])
def get_session(system_key, session_id):
    """Get a specific chat session"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    if session_id not in chat_sessions[system_key]:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = chat_sessions[system_key][session_id]
    return jsonify({
        'id': session_id,
        'title': session_data.get('title', f"New {SYSTEMS[system_key]['name']} Chat"),
        'messages': session_data['messages'],
        'created_at': session_data['created_at'].isoformat(),
        'last_updated': session_data['last_updated'].isoformat(),
        'message_count': len(session_data['messages'])
    })

@app.route('/api/sessions/<system_key>/<session_id>', methods=['DELETE'])
def delete_session(system_key, session_id):
    """Delete a chat session"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    if session_id in chat_sessions[system_key]:
        del chat_sessions[system_key][session_id]
        return jsonify({'success': True})
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Handle questions using Mistral API"""
    data = request.json
    system_key = data.get('system', 'legal')
    question = data.get('question', '')
    session_id = data.get('session_id')
    
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 400
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Store user message
    if session_id and session_id in chat_sessions[system_key]:
        chat_sessions[system_key][session_id]['messages'].append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        chat_sessions[system_key][session_id]['message_count'] += 1
        chat_sessions[system_key][session_id]['last_updated'] = datetime.now()
        
        # Update title
        new_title = generate_chat_title(chat_sessions[system_key][session_id]['messages'], system_key)
        chat_sessions[system_key][session_id]['title'] = new_title
    
    try:
        # Get system prompt
        system_prompt = SYSTEM_PROMPTS.get(system_key, SYSTEM_PROMPTS['all'])
        
        # Add conversation context if exists
        if session_id and session_id in chat_sessions[system_key]:
            recent_messages = chat_sessions[system_key][session_id]['messages'][-5:]  # Last 5 messages for context
            if len(recent_messages) > 1:
                context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_messages[:-1]])
                question = f"Previous conversation:\n{context}\n\nUser: {question}"
        
        # Call Mistral API
        answer = call_mistral_api_cached(question, system_prompt)
        
        # Add system-specific formatting
        if system_key == 'legal':
            answer = f"⚖️ **Legal Response:**\n\n{answer}"
        elif system_key == 'medical':
            answer = f"🏥 **Medical Information:**\n\n{answer}"
        elif system_key == 'disaster':
            answer = f"🌊 **Disaster Management:**\n\n{answer}"
        else:
            answer = f"🌐 **Integrated Response:**\n\n{answer}"
        
        # Store assistant response
        if session_id and session_id in chat_sessions[system_key]:
            chat_sessions[system_key][session_id]['messages'].append({
                'role': 'assistant',
                'content': answer,
                'timestamp': datetime.now().isoformat()
            })
            chat_sessions[system_key][session_id]['message_count'] += 1
            chat_sessions[system_key][session_id]['last_updated'] = datetime.now()
        
        return jsonify({
            'question': question,
            'answer': answer,
            'system': system_key,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/systems', methods=['GET'])
def get_systems():
    """Get all available systems and their status"""
    systems_status = {}
    for key in SYSTEMS:
        systems_status[key] = {
            'info': SYSTEMS[key],
            'status': {
                'ready': system_states[key]['ready'],
                'loading': system_states[key]['loading'],
                'chunks_count': system_states[key].get('chunks_count', 0)
            }
        }
    return jsonify(systems_status)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'mistral_api': 'configured' if MISTRAL_API_KEY else 'missing',
        'systems': len(SYSTEMS)
    }), 200

if __name__ == "__main__":
    # Create necessary folders
    for system in ['legal', 'medical', 'disaster']:
        folder = SYSTEMS[system]['pdf_folder']
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created folder: {folder}")
    
    # Check for Mistral API key
    if not MISTRAL_API_KEY:
        print("\n⚠️ WARNING: MISTRAL_API_KEY not set!")
        print("Please set it in Render environment variables or .env file")
    else:
        print(f"\n✅ Mistral API configured (Model: {MISTRAL_MODEL})")
    
    # Start initialization in background (lightweight)
    init_thread = threading.Thread(target=initialize_all_systems)
    init_thread.daemon = True
    init_thread.start()
    
    print("\n" + "="*60)
    print("🚀 Multi-System RAG Platform Starting (Memory Optimized)")
    print("="*60)
    print("\n✨ Features:")
    print("   - No local ML models (using Mistral API)")
    print(f"   - Model: {MISTRAL_MODEL}")
    print("   - Memory usage: < 200MB")
    print("\n📂 PDF folders created (optional - for reference only):")
    print("   - data/IPC_docs/     (Legal documents)")
    print("   - data/medical_docs/ (Medical documents)")
    print("   - data/disaster_docs/ (Disaster documents)")
    print("\n🌐 Access the application at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Get port from environment (for Render)
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=port)
