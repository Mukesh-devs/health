"""
Health LLM Application Package
"""
import os

from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager

# Initialize extensions (without app)
db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

def create_app(config_name='default'):
    """Application factory pattern"""
    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static')
    
    # Load configuration
    from .config import config
    app.config.from_object(config[config_name])
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize extensions with app
    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)
    
    # Register blueprints
    from .auth import auth_bp
    from .chat_sessions import sessions_bp
    from .query import query_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(sessions_bp, url_prefix='/api')
    app.register_blueprint(query_bp, url_prefix='/api')

    @app.route('/')
    def home():
        return send_from_directory(STATIC_DIR, 'login.html')

    @app.route('/login')
    def login_page():
        return send_from_directory(STATIC_DIR, 'login.html')

    @app.route('/login.html')
    def login_page_html():
        return send_from_directory(STATIC_DIR, 'login.html')

    @app.route('/chat')
    def chat_page():
        return send_from_directory(STATIC_DIR, 'chat.html')

    @app.route('/chat.html')
    def chat_page_html():
        return send_from_directory(STATIC_DIR, 'chat.html')

    @app.route('/chat/<public_id>')
    def chat_session(public_id):
        return send_from_directory(STATIC_DIR, 'chat.html')

    @app.route('/evidence')
    def evidence_page():
        return send_from_directory(STATIC_DIR, 'evidence.html')

    @app.route('/evidence.html')
    def evidence_page_html():
        return send_from_directory(STATIC_DIR, 'evidence.html')

    @app.route('/evidence/<public_id>')
    def evidence_session(public_id):
        return send_from_directory(STATIC_DIR, 'evidence.html')
    
    # Initialize knowledge graph and embeddings
    with app.app_context():
        # Create database tables
        db.create_all()
        
        # Load knowledge graph and embeddings
        from .kg_loader import load_knowledge_graph
        from .embedding import load_embeddings, set_kg_data, initialize_model
        from .metrics_logger import retrieval_logger
        from .contradiction_detector import init_contradiction_detector
        
        print("Initializing knowledge graph and embeddings...")
        
        from .hf_cache import download_if_missing

        download_if_missing()
        kg_df, nodes_df = load_knowledge_graph()

        # Initialize sentence transformer model
        initialize_model()

        # Load embeddings
        load_embeddings(nodes_df)
        # # Initialize sentence transformer model
        # initialize_model()
        
        # # Load embeddings
        # load_embeddings(nodes_df)
        
        # Set KG data for triple validation
        set_kg_data(kg_df)
        
        # Initialize contradiction detector (IMPROVEMENT 3)
        print("Initializing contradiction detector...")
        init_contradiction_detector(kg_df)
        print("✓ Contradiction detector initialized!")
        
        # Log startup retrieval metrics (will be called from embedding.py after first evaluation)
        print("Application initialized successfully!")
        print("Retrieval metrics will be logged at: retrieval_metrics.log")
        print("Query metrics will be logged at: query_metrics.jsonl")
        print("⚡ Live PubMed integration: ENABLED")
        print("⚠️  Contradiction detection: ENABLED")
    
    return app
