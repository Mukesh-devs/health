"""
Database Models
"""
from datetime import datetime, timezone
from . import db, bcrypt
import uuid

class User(db.Model):
    """User model for authentication"""
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
    public_id = db.Column(
    db.String(36),
    unique=True,
    nullable=False,
    default=lambda: str(uuid.uuid4())
    )
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False, default="New Chat")
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    user = db.relationship('User', backref=db.backref('chat_sessions', lazy=True))

class ChatHistory(db.Model):
    """Model to store chat messages"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=True)  # Nullable for backward compatibility
    query = db.Column(db.Text, nullable=False)
    response_data = db.Column(db.JSON, nullable=False) 
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    user = db.relationship('User', backref=db.backref('chat_histories', lazy=True))
    session = db.relationship('ChatSession', backref=db.backref('messages', lazy=True))
