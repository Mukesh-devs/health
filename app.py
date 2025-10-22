import os
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pickle
import time
import psutil
from functools import wraps
from collections import defaultdict, deque

from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import process, fuzz

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_app.db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'a-long-random-string-that-is-hard-to-guess-and-is-secure' 

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --- Performance Monitoring Setup ---
class PerformanceMonitor:
    """Centralized performance monitoring system"""
    def __init__(self, max_history=100):
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'min_time': float('inf'),
            'max_time': 0,
            'recent_times': deque(maxlen=max_history),
            'errors': 0
        })
        self.start_time = time.time()
        # Track embedding match accuracy
        self.embedding_matches = deque(maxlen=max_history)
        # Track detailed query metrics
        self.query_details = deque(maxlen=max_history)
        
    def record(self, operation, duration, success=True):
        """Record a performance metric"""
        m = self.metrics[operation]
        m['count'] += 1
        m['total_time'] += duration
        m['min_time'] = min(m['min_time'], duration)
        m['max_time'] = max(m['max_time'], duration)
        m['recent_times'].append(duration)
        if not success:
            m['errors'] += 1
    
    def record_embedding_match(self, entity_name, match_score, matched_cui, method):
        """Record embedding match accuracy"""
        self.embedding_matches.append({
            'entity': entity_name,
            'score': match_score,
            'cui': matched_cui,
            'method': method,  # 'exact', 'semantic', 'fuzzy', or 'not_found'
            'timestamp': time.time()
        })
    
    def record_query_detail(self, query, response_time_seconds, entities_found, triples_count, llm_time, 
                           requested_length=None, response_text=None, entity_matches=None):
        """Record detailed query information including adherence and per-query accuracy"""
        
        # Calculate response adherence
        adherence = None
        if requested_length and response_text:
            adherence = self._check_length_adherence(requested_length, response_text)
        
        # Calculate per-query embedding accuracy
        query_accuracy = None
        if entity_matches:
            query_accuracy = self._calculate_query_accuracy(entity_matches)
        
        self.query_details.append({
            'query': query[:100],  # Truncate long queries
            'response_time_seconds': response_time_seconds,
            'entities_found': entities_found,
            'triples_count': triples_count,
            'llm_time_seconds': llm_time,
            'requested_length': requested_length,
            'adherence': adherence,
            'query_accuracy': query_accuracy,
            'timestamp': time.time()
        })
    
    def _check_length_adherence(self, requested_length, response_text):
        """Check if response adheres to requested length"""
        word_count = len(response_text.split())
        
        # Define expected word ranges for each length
        length_ranges = {
            'short': (20, 80),      # Short: 20-80 words
            'medium': (80, 200),    # Medium: 80-200 words
            'long': (200, 500)      # Long: 200-500 words
        }
        
        if requested_length not in length_ranges:
            return None
        
        min_words, max_words = length_ranges[requested_length]
        is_adherent = min_words <= word_count <= max_words
        
        return {
            'is_adherent': is_adherent,
            'requested': requested_length,
            'word_count': word_count,
            'expected_range': f"{min_words}-{max_words}"
        }
    
    def _calculate_query_accuracy(self, entity_matches):
        """Calculate accuracy metrics for entities in a single query"""
        if not entity_matches:
            return None
        
        total = len(entity_matches)
        exact = sum(1 for m in entity_matches if m['method'] == 'exact')
        semantic = sum(1 for m in entity_matches if m['method'] == 'semantic')
        fuzzy = sum(1 for m in entity_matches if m['method'] == 'fuzzy')
        not_found = sum(1 for m in entity_matches if m['method'] == 'not_found')
        
        # Calculate average scores
        semantic_scores = [m['score'] for m in entity_matches if m['method'] == 'semantic' and m['score']]
        avg_semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
        
        fuzzy_scores = [m['score'] for m in entity_matches if m['method'] == 'fuzzy' and m['score']]
        avg_fuzzy_score = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0
        
        success_count = exact + semantic + fuzzy
        
        return {
            'total_entities': total,
            'exact_matches': exact,
            'semantic_matches': semantic,
            'fuzzy_matches': fuzzy,
            'not_found': not_found,
            'success_rate': (success_count / total * 100) if total > 0 else 0,
            'avg_semantic_score': round(avg_semantic_score, 3),
            'avg_fuzzy_score': round(avg_fuzzy_score, 3),
            'match_details': entity_matches
        }
    
    def get_embedding_accuracy_stats(self):
        """Get embedding match accuracy statistics"""
        if not self.embedding_matches:
            return None
        
        matches = list(self.embedding_matches)
        total = len(matches)
        
        # Count by method
        exact_matches = sum(1 for m in matches if m['method'] == 'exact')
        semantic_matches = sum(1 for m in matches if m['method'] == 'semantic')
        fuzzy_matches = sum(1 for m in matches if m['method'] == 'fuzzy')
        not_found = sum(1 for m in matches if m['method'] == 'not_found')
        
        # Calculate average scores for successful matches
        semantic_scores = [m['score'] for m in matches if m['method'] == 'semantic' and m['score'] is not None]
        fuzzy_scores = [m['score'] for m in matches if m['method'] == 'fuzzy' and m['score'] is not None]
        
        return {
            'total_lookups': total,
            'exact_matches': exact_matches,
            'exact_match_rate': (exact_matches / total * 100) if total > 0 else 0,
            'semantic_matches': semantic_matches,
            'semantic_match_rate': (semantic_matches / total * 100) if total > 0 else 0,
            'semantic_avg_score': (sum(semantic_scores) / len(semantic_scores)) if semantic_scores else 0,
            'fuzzy_matches': fuzzy_matches,
            'fuzzy_match_rate': (fuzzy_matches / total * 100) if total > 0 else 0,
            'fuzzy_avg_score': (sum(fuzzy_scores) / len(fuzzy_scores)) if fuzzy_scores else 0,
            'not_found': not_found,
            'not_found_rate': (not_found / total * 100) if total > 0 else 0,
            'overall_success_rate': ((exact_matches + semantic_matches + fuzzy_matches) / total * 100) if total > 0 else 0
        }
    
    def get_query_details_stats(self):
        """Get detailed query statistics"""
        if not self.query_details:
            return None
        
        details = list(self.query_details)
        
        # Calculate adherence rate
        queries_with_length = [d for d in details if d.get('adherence')]
        adherent_queries = [d for d in queries_with_length if d['adherence']['is_adherent']]
        adherence_rate = (len(adherent_queries) / len(queries_with_length) * 100) if queries_with_length else 0
        
        # Adherence by length type
        adherence_by_length = {}
        for length_type in ['short', 'medium', 'long']:
            type_queries = [d for d in queries_with_length if d['adherence']['requested'] == length_type]
            type_adherent = [d for d in type_queries if d['adherence']['is_adherent']]
            adherence_by_length[length_type] = {
                'total': len(type_queries),
                'adherent': len(type_adherent),
                'rate': (len(type_adherent) / len(type_queries) * 100) if type_queries else 0
            }
        
        return {
            'total_queries': len(details),
            'avg_response_time_seconds': sum(d['response_time_seconds'] for d in details) / len(details),
            'min_response_time_seconds': min(d['response_time_seconds'] for d in details),
            'max_response_time_seconds': max(d['response_time_seconds'] for d in details),
            'avg_entities_per_query': sum(d['entities_found'] for d in details) / len(details),
            'avg_triples_per_query': sum(d['triples_count'] for d in details) / len(details),
            'avg_llm_time_seconds': sum(d['llm_time_seconds'] for d in details) / len(details),
            'adherence_rate': adherence_rate,
            'adherence_by_length': adherence_by_length,
            'queries_with_length_preference': len(queries_with_length),
            'recent_queries': [
                {
                    'query': d['query'],
                    'response_time_seconds': round(d['response_time_seconds'], 3),
                    'entities': d['entities_found'],
                    'triples': d['triples_count'],
                    'adherence': d.get('adherence'),
                    'accuracy': d.get('query_accuracy')
                } for d in list(details)[-10:]  # Last 10 queries
            ]
        }
    
    def get_stats(self, operation):
        """Get statistics for an operation"""
        m = self.metrics[operation]
        if m['count'] == 0:
            return None
        
        recent = list(m['recent_times'])
        return {
            'count': m['count'],
            'avg_time': m['total_time'] / m['count'],
            'min_time': m['min_time'],
            'max_time': m['max_time'],
            'recent_avg': sum(recent) / len(recent) if recent else 0,
            'errors': m['errors'],
            'success_rate': ((m['count'] - m['errors']) / m['count']) * 100
        }
    
    def get_all_stats(self):
        """Get all statistics"""
        return {op: self.get_stats(op) for op in self.metrics.keys()}
    
    def get_system_metrics(self):
        """Get current system resource usage"""
        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'uptime_seconds': time.time() - self.start_time,
            'threads': process.num_threads()
        }

perf_monitor = PerformanceMonitor()

def track_performance(operation_name):
    """Decorator to track function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                duration = time.time() - start_time
                perf_monitor.record(operation_name, duration, success)
        return wrapper
    return decorator

# --- JWT Error Handlers ---
@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({"error": "Invalid token. The token is malformed or has an invalid signature.", "details": str(error)}), 422

@jwt.unauthorized_loader
def missing_token_callback(reason):
    return jsonify({"error": "Authorization token is missing.", "details": reason}), 401

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({"error": "Token has expired. Please log in again."}), 401

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class ChatSession(db.Model):
    """Model to track multiple chat sessions for each user"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False, default="New Chat")
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    user = db.relationship('User', backref=db.backref('chat_sessions', lazy=True))

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=True)  # Nullable for backward compatibility
    query = db.Column(db.Text, nullable=False)
    response_data = db.Column(db.JSON, nullable=False) 
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    user = db.relationship('User', backref=db.backref('chat_histories', lazy=True))
    session = db.relationship('ChatSession', backref=db.backref('messages', lazy=True))

# --- Gemini API Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY","AIzaSyDELglk1DUtDs0F0YZWsh23mP4_DVt584A")
if not API_KEY:
    raise ValueError("Error: The GEMINI_API_KEY environment variable is not set.")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- Initialize Sentence Transformer ---
print("Loading Sentence Transformer model...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# --- Global variables for embeddings ---
entity_embeddings_matrix = None
entity_names_list = []
cui_list = []

# --- Load Knowledge Graphs with Optimized Embedding Strategy ---
def load_embeddings():
    """Load or generate embeddings with caching (Apple Silicon optimized)"""
    global entity_embeddings_matrix, entity_names_list, cui_list
    
    embeddings_file = 'entity_embeddings_cache.npz'
    
    # Try to load cached embeddings
    if os.path.exists(embeddings_file):
        print("Loading cached embeddings...")
        cache = np.load(embeddings_file, allow_pickle=True)
        entity_embeddings_matrix = cache['embeddings']
        entity_names_list = cache['names'].tolist()
        cui_list = cache['cuis'].tolist()
        print(f"Loaded cached embeddings for {len(entity_names_list)} entities.")
        return
    
    # Generate embeddings if cache doesn't exist
    print(f"Generating embeddings for {len(nodes_df)} entities...")
    print("This is a one-time process and may take 10-15 minutes for 160K entities.")
    
    entity_names_list = nodes_df['Name'].tolist()
    cui_list = nodes_df['CUI'].tolist()
    
    # Generate embeddings in batches for memory efficiency
    batch_size = 512
    embeddings_list = []
    
    total_batches = (len(entity_names_list) + batch_size - 1) // batch_size
    
    for i in range(0, len(entity_names_list), batch_size):
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        batch = entity_names_list[i:i+batch_size]
        batch_embeddings = sentence_model.encode(
            batch, 
            show_progress_bar=False,
            batch_size=32,
            convert_to_numpy=True
        )
        embeddings_list.append(batch_embeddings)
    
    entity_embeddings_matrix = np.vstack(embeddings_list).astype('float32')
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(entity_embeddings_matrix, axis=1, keepdims=True)
    entity_embeddings_matrix = entity_embeddings_matrix / norms
    
    # Cache embeddings
    print("Caching embeddings for future use...")
    np.savez_compressed(
        embeddings_file,
        embeddings=entity_embeddings_matrix,
        names=np.array(entity_names_list),
        cuis=np.array(cui_list)
    )
    
    print(f"Embeddings generated and cached for {len(entity_names_list)} entities.")

try:
    kg_df = pd.read_csv('neo4j_rel.csv', dtype=str)
    kg_df.fillna('', inplace=True)
    nodes_df = pd.read_csv('neo4j_node.csv', dtype=str)
    nodes_df.fillna('', inplace=True)
    nodes_df['Name_lower'] = nodes_df['Name'].str.lower().str.strip()
    
    print("Knowledge graph data loaded successfully.")
    
    # Load or generate embeddings
    load_embeddings()
    
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'neo4j_rel.csv' and 'neo4j_node.csv' are in the same directory.")
    exit()

# --- Rule-Based Dialog Manager ---
RULE_BASED_RESPONSES = {
    "hello": "Hello again! How can I assist you with your health questions today?",
    "hi": "Hi there! What health information can I help you explore?",
    "how are you": "I'm a computer program, but I'm functioning perfectly. Thanks for asking! What can I help you with?",
    "thanks": "You're welcome! Let me know if you have more questions.",
    "thank you": "You're most welcome! Is there anything else I can help you explore?"
}

# --- Fast Semantic Search (NumPy optimized for Apple Silicon) ---
@track_performance('semantic_search')
def get_cui_from_name_semantic(entity_name, threshold=0.65, top_k=5):
    """
    Uses optimized NumPy operations for fast semantic similarity search.
    Optimized for Apple Silicon's Accelerate framework.
    """
    if not entity_name or entity_embeddings_matrix is None:
        perf_monitor.record_embedding_match(entity_name, None, None, 'not_found')
        return None
    
    entity_name_lower = entity_name.lower().strip()
    
    # Fast exact match first
    exact_match = nodes_df[nodes_df['Name_lower'] == entity_name_lower]
    if not exact_match.empty:
        cui = exact_match.iloc[0]['CUI']
        perf_monitor.record_embedding_match(entity_name, 1.0, cui, 'exact')
        return cui
    
    # Semantic search
    query_embedding = sentence_model.encode([entity_name], convert_to_numpy=True).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Vectorized cosine similarity (uses Apple Accelerate framework automatically)
    similarities = np.dot(entity_embeddings_matrix, query_embedding.T).flatten()
    
    # Get top_k indices
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]
    
    best_similarity = similarities[top_indices[0]]
    best_idx = top_indices[0]
    
    if best_similarity >= threshold:
        cui = cui_list[best_idx]
        perf_monitor.record_embedding_match(entity_name, float(best_similarity), cui, 'semantic')
        return cui
    
    perf_monitor.record_embedding_match(entity_name, float(best_similarity), None, 'not_found')
    return None

# --- Hybrid Entity Matching ---
@track_performance('entity_matching')
def get_cui_from_name(entity_name, threshold=90):
    """
    Hybrid approach: semantic search -> fuzzy matching fallback
    """
    if not entity_name:
        return None
    
    # Try semantic search first
    semantic_cui = get_cui_from_name_semantic(entity_name, threshold=0.65)
    if semantic_cui:
        return semantic_cui
    
    # Fallback to fuzzy matching for edge cases
    entity_name_lower = entity_name.lower().strip()
    matches = process.extract(entity_name_lower, nodes_df['Name_lower'], scorer=fuzz.token_sort_ratio, limit=1)
    
    if matches and matches[0][1] >= threshold:
        match_idx = nodes_df[nodes_df['Name_lower'] == matches[0][0]].index[0]
        cui = nodes_df.iloc[match_idx]['CUI']
        # Record fuzzy match with normalized score (0-1 range)
        perf_monitor.record_embedding_match(entity_name, matches[0][1] / 100.0, cui, 'fuzzy')
        return cui
    
    # If we got here and semantic search didn't already record not_found, record it
    perf_monitor.record_embedding_match(entity_name, None, None, 'not_found')
    return None

# --- Triple Validation ---
@track_performance('triple_validation')
def validate_triple(subject, relation, obj):
    """Validates a single triple against the loaded knowledge graph."""
    subject_cui = get_cui_from_name(subject)
    object_cui = get_cui_from_name(obj)
    relation_upper = relation.strip().upper()

    if not subject_cui or not object_cui:
        return {'status': 'unsure', 'pubmed_ids': None, 'sentence': None}

    results = kg_df[
        (kg_df['START_ID'] == subject_cui) &
        (kg_df['END_ID'] == object_cui) &
        (kg_df['PREDICATE'] == relation_upper)
    ]
    if not results.empty:
        top_result = results.iloc[0]
        return {'status': 'supported', 'pubmed_ids': top_result['PubMed_ID'], 'sentence': top_result['SENTENCE']}

    relevant_results = kg_df[
        (kg_df['START_ID'] == subject_cui) &
        (kg_df['END_ID'] == object_cui)
    ]
    if not relevant_results.empty:
        top_result = relevant_results.iloc[0]
        return {'status': 'relevant', 'pubmed_ids': top_result['PubMed_ID'], 'sentence': top_result['SENTENCE']}

    return {'status': 'unsure', 'pubmed_ids': None, 'sentence': None}

# --- Recommendation Generator ---
@track_performance('recommendation_generation')
def generate_recommendations(entities_in_graph, max_recommendations=4):
    """Generates follow-up questions based on entities in the current graph."""
    recommendations = set()
    if not entities_in_graph:
        return []

    entity_cuis = [get_cui_from_name(name) for name in entities_in_graph]
    entity_cuis = [cui for cui in entity_cuis if cui]

    for cui in entity_cuis:
        related = kg_df[(kg_df['START_ID'] == cui) | (kg_df['END_ID'] == cui)]
        if related.empty:
            continue

        for _, row in related.sample(min(len(related), 3)).iterrows():
            if len(recommendations) >= max_recommendations:
                break
            
            start_node_name = nodes_df[nodes_df['CUI'] == row['START_ID']].iloc[0]['Name']
            end_node_name = nodes_df[nodes_df['CUI'] == row['END_ID']].iloc[0]['Name']

            if row['START_ID'] == cui:
                recommendations.add(f"What is the relationship between {start_node_name} and {end_node_name}?")
            else:
                recommendations.add(f"How does {end_node_name} affect {start_node_name}?")
    
    return list(recommendations)

# --- Authentication Routes ---
@app.route('/api/register', methods=['POST'])
@track_performance('api_register')
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user_exists = db.session.scalar(db.select(User).filter_by(username=username))
    if user_exists:
        return jsonify({"error": "Username already exists"}), 409

    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": f"User {username} created successfully"}), 201

@app.route('/api/login', methods=['POST'])
@track_performance('api_login')
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = db.session.scalar(db.select(User).filter_by(username=username))

    if user and user.check_password(password):
        access_token = create_access_token(identity=user.id)
        return jsonify(access_token=access_token)
    
    return jsonify({"error": "Invalid username or password"}), 401

@app.route('/api/history', methods=['GET'])
@jwt_required()
@track_performance('api_history')
def get_history():
    """Get chat history for a specific session or all history"""
    current_user_id = get_jwt_identity()
    session_id = request.args.get('session_id', type=int)
    
    if session_id:
        # Get history for specific session
        history_records = db.session.scalars(
            db.select(ChatHistory)
            .filter_by(user_id=current_user_id, session_id=session_id)
            .order_by(ChatHistory.timestamp.asc())
        ).all()
    else:
        # Get all history (backward compatibility)
        history_records = db.session.scalars(
            db.select(ChatHistory)
            .filter_by(user_id=current_user_id)
            .order_by(ChatHistory.timestamp.asc())
        ).all()
    
    formatted_history = []
    for record in history_records:
        formatted_history.append({
            "userQuery": record.query,
            "aiResponse": record.response_data,
            "timestamp": record.timestamp.isoformat(),
            "session_id": record.session_id
        })
        
    return jsonify(formatted_history)

# --- Chat Session Management Endpoints ---
@app.route('/api/sessions', methods=['GET'])
@jwt_required()
@track_performance('api_sessions_list')
def list_sessions():
    """List all chat sessions for the current user"""
    current_user_id = get_jwt_identity()
    
    sessions = db.session.scalars(
        db.select(ChatSession)
        .filter_by(user_id=current_user_id)
        .order_by(ChatSession.updated_at.desc())
    ).all()
    
    session_list = []
    for session in sessions:
        # Get message count for this session
        message_count = db.session.scalar(
            db.select(db.func.count(ChatHistory.id))
            .filter_by(session_id=session.id)
        )
        
        session_list.append({
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": message_count or 0
        })
    
    return jsonify(session_list)

@app.route('/api/sessions', methods=['POST'])
@jwt_required()
@track_performance('api_sessions_create')
def create_session():
    """Create a new chat session"""
    current_user_id = get_jwt_identity()
    data = request.json or {}
    title = data.get('title', 'New Chat')
    
    new_session = ChatSession(
        user_id=current_user_id,
        title=title
    )
    db.session.add(new_session)
    db.session.commit()
    
    return jsonify({
        "id": new_session.id,
        "title": new_session.title,
        "created_at": new_session.created_at.isoformat(),
        "updated_at": new_session.updated_at.isoformat(),
        "message_count": 0
    }), 201

@app.route('/api/sessions/<int:session_id>', methods=['PUT'])
@jwt_required()
@track_performance('api_sessions_update')
def update_session(session_id):
    """Update chat session (e.g., rename)"""
    current_user_id = get_jwt_identity()
    data = request.json or {}
    
    session = db.session.get(ChatSession, session_id)
    
    if not session or session.user_id != current_user_id:
        return jsonify({"error": "Session not found"}), 404
    
    if 'title' in data:
        session.title = data['title']
        db.session.commit()
    
    return jsonify({
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat()
    })

@app.route('/api/sessions/<int:session_id>', methods=['DELETE'])
@jwt_required()
@track_performance('api_sessions_delete')
def delete_session(session_id):
    """Delete a chat session and all its messages"""
    current_user_id = get_jwt_identity()
    
    session = db.session.get(ChatSession, session_id)
    
    if not session or session.user_id != current_user_id:
        return jsonify({"error": "Session not found"}), 404
    
    # Delete all messages in this session
    db.session.execute(
        db.delete(ChatHistory).where(ChatHistory.session_id == session_id)
    )
    
    # Delete the session
    db.session.delete(session)
    db.session.commit()
    
    return jsonify({"message": "Session deleted successfully"}), 200

# --- Main Query Endpoint ---
@app.route('/api/query', methods=['POST'])
@jwt_required()
@track_performance('api_query')
def handle_query():
    current_user_id = get_jwt_identity()
    
    # Track overall query performance
    query_start_time = time.time()
    llm_start_time = None
    llm_duration = 0
    
    try:
        data = request.json
        user_query = data.get("query", "").lower().strip()
        answer_length = data.get("answerLength", "medium")
        session_id = data.get("session_id")  # Get session_id from request

        if user_query in RULE_BASED_RESPONSES:
            response_data = {"predefinedResponse": RULE_BASED_RESPONSES[user_query]}
            history_entry = ChatHistory(user_id=current_user_id, session_id=session_id, query=user_query, response_data=response_data)
            db.session.add(history_entry)
            db.session.commit()
            
            # Update session timestamp
            if session_id:
                session = db.session.get(ChatSession, session_id)
                if session:
                    session.updated_at = datetime.now(timezone.utc)
                    db.session.commit()
            
            return jsonify(response_data)

        # Get recent chats from the same session if provided
        if session_id:
            recent_chats_query = db.select(ChatHistory).filter_by(user_id=current_user_id, session_id=session_id).order_by(ChatHistory.timestamp.desc()).limit(5)
        else:
            recent_chats_query = db.select(ChatHistory).filter_by(user_id=current_user_id).order_by(ChatHistory.timestamp.desc()).limit(5)
        
        recent_chats = db.session.scalars(recent_chats_query).all()
        recent_chats.reverse() 

        history_for_api = []
        for chat in recent_chats:
            history_for_api.append({'source': 'user', 'text': chat.query})
            ai_text = chat.response_data.get('predefinedResponse') or chat.response_data.get('textualResponse')
            if ai_text:
                history_for_api.append({'source': 'model', 'text': ai_text})

        system_instruction = {
            "role": "system",
            "parts": [{
                "text": f"""You are an expert health information system. Your task is to respond to the user's query and extract key facts as a list of subject-relation-object triples.
- Provide a factual, concise response in the 'textualResponse' field.
- **Critically, you must also provide a 'highlightedResponse'. This response should be identical to the textualResponse, but with all extracted subject and object entities wrapped in '*|' and '|*' markers.** For example, if a sentence is "Aspirin treats headaches", the highlighted version must be "*|Aspirin|* treats *|headaches|*".
- Identify and extract key entities and relationships into the 'triples' field.
- Follow the JSON schema strictly. Do not add extra explanations.
- The user's desired answer length is "{answer_length}"."""
            }]
        }
        
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "textualResponse": {"type": "STRING"},
                "highlightedResponse": {"type": "STRING"},
                "triples": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": { "subject": {"type": "STRING"}, "relation": {"type": "STRING"}, "object": {"type": "STRING"}},
                        "required": ["subject", "relation", "object"]
                    }
                }
            },
            "required": ["textualResponse", "highlightedResponse", "triples"]
        }
        
        formatted_history = []
        for entry in history_for_api:
            role = "user" if entry.get('source') == 'user' else "model"
            formatted_history.append({"role": role, "parts": [{"text": entry.get('text', '')}]})
        
        formatted_history.append({"role": "user", "parts": [{"text": user_query}]})

        payload = {
            "contents": formatted_history,
            "system_instruction": system_instruction,
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        }
        
        # Track LLM API call performance
        llm_start_time = time.time()
        response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        llm_duration = time.time() - llm_start_time
        perf_monitor.record('llm_api_call', llm_duration, success=True)
        
        llm_response = response.json()
        
        if not llm_response.get('candidates'):
             raise ValueError("Invalid LLM response: No candidates found.")
        
        llm_text = llm_response['candidates'][0]['content']['parts'][0]['text']
        llm_data = json.loads(llm_text)

        textual_response = llm_data.get('textualResponse', 'I could not generate a textual response.')
        highlighted_response = llm_data.get('highlightedResponse', textual_response)
        triples = llm_data.get('triples', [])

        nodes, edges, node_ids = [], [], set()
        validated_triples = []
        
        # Track entity matches for this specific query
        query_entity_matches = []

        for triple in triples:
            subject = triple.get('subject', '').strip()
            relation = triple.get('relation', '').strip()
            obj = triple.get('object', '').strip()

            if not all([subject, relation, obj]):
                continue

            if subject not in node_ids:
                nodes.append({"id": subject, "label": subject})
                node_ids.add(subject)
            if obj not in node_ids:
                nodes.append({"id": obj, "label": obj})
                node_ids.add(obj)
            edges.append({"from": subject, "to": obj, "label": relation})

            validation = validate_triple(subject, relation, obj)
            validated_triples.append({**triple, **validation})
        
        # Collect entity matches from the recent embedding lookups
        # Get the last N matches where N = number of unique entities in this query
        all_entities = list(node_ids)
        recent_matches = list(perf_monitor.embedding_matches)[-len(all_entities):] if all_entities else []
        for match in recent_matches:
            query_entity_matches.append({
                'entity': match['entity'],
                'score': match['score'],
                'method': match['method']
            })
        
        recommendations = generate_recommendations(list(node_ids))
        
        # Calculate total response time (Latency = TimeEnd - TimeStart)
        total_response_time = time.time() - query_start_time
        
        # Calculate per-query accuracy
        query_accuracy = None
        if query_entity_matches:
            total_entities = len(query_entity_matches)
            exact = sum(1 for m in query_entity_matches if m['method'] == 'exact')
            semantic = sum(1 for m in query_entity_matches if m['method'] == 'semantic')
            fuzzy = sum(1 for m in query_entity_matches if m['method'] == 'fuzzy')
            not_found = sum(1 for m in query_entity_matches if m['method'] == 'not_found')
            
            semantic_scores = [m['score'] for m in query_entity_matches if m['method'] == 'semantic' and m['score']]
            avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
            
            success_count = exact + semantic + fuzzy
            query_accuracy = {
                'total_entities': total_entities,
                'exact_matches': exact,
                'semantic_matches': semantic,
                'fuzzy_matches': fuzzy,
                'not_found': not_found,
                'success_rate': round((success_count / total_entities * 100) if total_entities > 0 else 0, 2),
                'avg_semantic_score': round(avg_semantic, 3),
                'entity_details': query_entity_matches
            }
        
        # Record detailed query metrics
        perf_monitor.record_query_detail(
            query=data.get("query", ""),
            response_time_seconds=total_response_time,
            entities_found=len(node_ids),
            triples_count=len(triples),
            llm_time=llm_duration,
            requested_length=answer_length,
            response_text=textual_response,
            entity_matches=query_entity_matches
        )

        final_response_data = {
            "textualResponse": textual_response,
            "highlightedResponse": highlighted_response,
            "graphData": {"nodes": nodes, "edges": edges},
            "validatedTriples": validated_triples,
            "recommendations": recommendations,
            "performance": {
                "latency_seconds": round(total_response_time, 3),  # Latency = TimeEnd - TimeStart
                "response_time_seconds": round(total_response_time, 3),
                "llm_time_seconds": round(llm_duration, 3),
                "entities_found": len(node_ids),
                "triples_extracted": len(triples),
                "query_accuracy": query_accuracy,
                "requested_length": answer_length,
                "response_word_count": len(textual_response.split())
            }
        }

        new_history_entry = ChatHistory(
            user_id=current_user_id,
            session_id=session_id,
            query=data.get("query", ""),
            response_data=final_response_data
        )
        db.session.add(new_history_entry)
        
        # Update session timestamp and title if this is the first message
        if session_id:
            session = db.session.get(ChatSession, session_id)
            if session:
                session.updated_at = datetime.now(timezone.utc)
                # Auto-generate title from first query if still "New Chat"
                if session.title == "New Chat":
                    # Use first 50 chars of query as title
                    session.title = data.get("query", "")[:50] + ("..." if len(data.get("query", "")) > 50 else "")
        
        db.session.commit()

        return jsonify(final_response_data)

    except requests.exceptions.RequestException as e:
        if llm_start_time:
            perf_monitor.record('llm_api_call', time.time() - llm_start_time, success=False)
        print(f"API call failed: {e}")
        return jsonify({"error": f"Failed to connect to the AI service: {e}"}), 500
    except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
        print(f"Failed to parse or process LLM response: {e}")
        return jsonify({"error": f"Received an invalid response from the AI service: {e}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"An unexpected server error occurred."}), 500

# --- Performance Metrics Endpoint ---
@app.route('/api/metrics', methods=['GET'])
@jwt_required()
def get_metrics():
    """Return performance metrics and system resource usage"""
    try:
        all_stats = perf_monitor.get_all_stats()
        system_metrics = perf_monitor.get_system_metrics()
        embedding_accuracy = perf_monitor.get_embedding_accuracy_stats()
        query_details = perf_monitor.get_query_details_stats()
        
        # Calculate database statistics
        db_stats = {}
        with app.app_context():
            user_count = db.session.query(User).count()
            chat_count = db.session.query(ChatHistory).count()
            db_stats = {
                'total_users': user_count,
                'total_chats': chat_count
            }
        
        # Format response with detailed breakdown
        response = {
            'system': {
                'uptime_hours': round(system_metrics['uptime_seconds'] / 3600, 2),
                'cpu_percent': round(system_metrics['cpu_percent'], 2),
                'memory_mb': round(system_metrics['memory_mb'], 2),
                'memory_percent': round(system_metrics['memory_percent'], 2),
                'threads': system_metrics['threads']
            },
            'database': db_stats,
            'api_endpoints': {},
            'processing': {},
            'accuracy': None,
            'query_details': None
        }
        
        # Add embedding accuracy stats
        if embedding_accuracy:
            response['accuracy'] = {
                'total_entity_lookups': embedding_accuracy['total_lookups'],
                'exact_match_rate': round(embedding_accuracy['exact_match_rate'], 2),
                'semantic_match_rate': round(embedding_accuracy['semantic_match_rate'], 2),
                'semantic_avg_score': round(embedding_accuracy['semantic_avg_score'], 3),
                'fuzzy_match_rate': round(embedding_accuracy['fuzzy_match_rate'], 2),
                'fuzzy_avg_score': round(embedding_accuracy['fuzzy_avg_score'], 3),
                'not_found_rate': round(embedding_accuracy['not_found_rate'], 2),
                'overall_success_rate': round(embedding_accuracy['overall_success_rate'], 2)
            }
        
        # Add query details
        if query_details:
            response['query_details'] = {
                'total_queries': query_details['total_queries'],
                'avg_response_time_seconds': round(query_details['avg_response_time_seconds'], 3),
                'avg_latency_seconds': round(query_details['avg_response_time_seconds'], 3),  # Latency metric
                'min_response_time_seconds': round(query_details['min_response_time_seconds'], 3),
                'max_response_time_seconds': round(query_details['max_response_time_seconds'], 3),
                'avg_entities_per_query': round(query_details['avg_entities_per_query'], 2),
                'avg_triples_per_query': round(query_details['avg_triples_per_query'], 2),
                'avg_llm_time_seconds': round(query_details['avg_llm_time_seconds'], 3),
                'recent_queries': query_details['recent_queries']
            }
            
            # Add adherence metrics
            response['adherence'] = {
                'overall_adherence_rate': round(query_details['adherence_rate'], 2),
                'queries_with_length_preference': query_details['queries_with_length_preference'],
                'adherence_by_length': {
                    'short': {
                        'total_queries': query_details['adherence_by_length']['short']['total'],
                        'adherent_queries': query_details['adherence_by_length']['short']['adherent'],
                        'adherence_rate': round(query_details['adherence_by_length']['short']['rate'], 2)
                    },
                    'medium': {
                        'total_queries': query_details['adherence_by_length']['medium']['total'],
                        'adherent_queries': query_details['adherence_by_length']['medium']['adherent'],
                        'adherence_rate': round(query_details['adherence_by_length']['medium']['rate'], 2)
                    },
                    'long': {
                        'total_queries': query_details['adherence_by_length']['long']['total'],
                        'adherent_queries': query_details['adherence_by_length']['long']['adherent'],
                        'adherence_rate': round(query_details['adherence_by_length']['long']['rate'], 2)
                    }
                }
            }
        
        # Categorize metrics
        api_prefixes = ['api_']
        processing_ops = ['semantic_search', 'entity_matching', 'triple_validation', 
                         'recommendation_generation', 'llm_api_call']
        
        for op, stats in all_stats.items():
            if stats:
                formatted_stats = {
                    'total_calls': stats['count'],
                    'avg_time_ms': round(stats['avg_time'] * 1000, 2),
                    'min_time_ms': round(stats['min_time'] * 1000, 2),
                    'max_time_ms': round(stats['max_time'] * 1000, 2),
                    'recent_avg_ms': round(stats['recent_avg'] * 1000, 2),
                    'success_rate': round(stats['success_rate'], 2),
                    'error_count': stats['errors']
                }
                
                if any(op.startswith(prefix) for prefix in api_prefixes):
                    response['api_endpoints'][op] = formatted_stats
                elif op in processing_ops:
                    response['processing'][op] = formatted_stats
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return jsonify({"error": "Failed to fetch metrics"}), 500

@app.route('/api/metrics/summary', methods=['GET'])
@jwt_required()
def get_metrics_summary():
    """Return a concise summary of key performance indicators"""
    try:
        query_stats = perf_monitor.get_stats('api_query')
        llm_stats = perf_monitor.get_stats('llm_api_call')
        semantic_stats = perf_monitor.get_stats('semantic_search')
        system_metrics = perf_monitor.get_system_metrics()
        embedding_accuracy = perf_monitor.get_embedding_accuracy_stats()
        query_details = perf_monitor.get_query_details_stats()
        
        summary = {
            'status': 'healthy' if system_metrics['memory_percent'] < 80 else 'warning',
            'uptime_hours': round(system_metrics['uptime_seconds'] / 3600, 2),
            'total_queries': query_stats['count'] if query_stats else 0,
            'avg_query_time_ms': round(query_stats['avg_time'] * 1000, 2) if query_stats else 0,
            'avg_latency_seconds': round(query_details['avg_response_time_seconds'], 3) if query_details else 0,
            'avg_query_time_seconds': round(query_details['avg_response_time_seconds'], 3) if query_details else 0,
            'avg_llm_time_ms': round(llm_stats['avg_time'] * 1000, 2) if llm_stats else 0,
            'avg_llm_time_seconds': round(query_details['avg_llm_time_seconds'], 3) if query_details else 0,
            'avg_search_time_ms': round(semantic_stats['avg_time'] * 1000, 2) if semantic_stats else 0,
            'memory_usage_mb': round(system_metrics['memory_mb'], 2),
            'cpu_percent': round(system_metrics['cpu_percent'], 2),
            'success_rate': round(query_stats['success_rate'], 2) if query_stats else 100.0,
            'embedding_match_rate': round(embedding_accuracy['overall_success_rate'], 2) if embedding_accuracy else 0,
            'semantic_match_score': round(embedding_accuracy['semantic_avg_score'], 3) if embedding_accuracy else 0,
            'adherence_rate': round(query_details['adherence_rate'], 2) if query_details else 0,
            'queries_with_length_preference': query_details['queries_with_length_preference'] if query_details else 0
        }
        
        return jsonify(summary)
        
    except Exception as e:
        print(f"Error fetching metrics summary: {e}")
        return jsonify({"error": "Failed to fetch metrics summary"}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)