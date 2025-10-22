"""
Health LLM Application Package
"""
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager

# Initialize extensions (without app)
db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()

def create_app(config_name='default'):
    """Application factory pattern"""
    app = Flask(__name__)
    
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
    
    # Initialize knowledge graph and embeddings
    with app.app_context():
        # Create database tables
        db.create_all()
        
        # Load knowledge graph and embeddings
        from .kg_loader import load_knowledge_graph
        from .embedding import load_embeddings, set_kg_data, initialize_model
        from .metrics_logger import retrieval_logger
        
        print("Initializing knowledge graph and embeddings...")
        kg_df, nodes_df = load_knowledge_graph()
        
        # Initialize sentence transformer model
        initialize_model()
        
        # Load embeddings
        load_embeddings(nodes_df)
        
        # Set KG data for triple validation
        set_kg_data(kg_df)
        
        # Log startup retrieval metrics (will be called from embedding.py after first evaluation)
        print("Application initialized successfully!")
        print("Retrieval metrics will be logged at: retrieval_metrics.log")
        print("Query metrics will be logged at: query_metrics.jsonl")
    
    return app
