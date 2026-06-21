"""
Application Configuration
"""
import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///health_app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', SECRET_KEY)
    
    # Gemini API Configuration
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    # FAISS Configuration
    FAISS_INDEX_PATH = os.environ.get(
        'FAISS_INDEX_PATH',
        'cache/faiss.index'
    )

    FAISS_METADATA_PATH = os.environ.get(
        'FAISS_METADATA_PATH',
        'cache/metadata.pkl'
    )
    
    # Knowledge Graph paths
    KG_REL_PATH = os.environ.get('KG_REL_PATH', 'dataset/neo4j_rel.csv')
    KG_NODE_PATH = os.environ.get('KG_NODE_PATH', 'dataset/neo4j_node.csv')
    EMBEDDINGS_CACHE_PATH = os.environ.get('EMBEDDINGS_CACHE_PATH', 'cache/entity_embeddings_cache.npz')
    
    # Sentence Transformer model
    SENTENCE_MODEL_NAME = os.environ.get('SENTENCE_MODEL_NAME', 'all-MiniLM-L6-v2')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
