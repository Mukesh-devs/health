"""
Chat Session Management Blueprint
"""
from datetime import datetime, timezone
from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from . import db
from .models import ChatSession, ChatHistory

sessions_bp = Blueprint('sessions', __name__)

@sessions_bp.route('/sessions', methods=['GET'])
@jwt_required()
def get_sessions():
    """Get all chat sessions for the current user"""
    user_id = int(get_jwt_identity())
    sessions = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).all()
    
    sessions_data = []
    for session in sessions:
        # Get message count for this session
        message_count = db.session.query(db.func.count(ChatHistory.id)).filter_by(session_id=session.id).scalar()
        
        sessions_data.append({
            'id': session.id,
            'title': session.title,
            'created_at': session.created_at.isoformat(),
            'updated_at': session.updated_at.isoformat(),
            'message_count': message_count or 0
        })
    
    return jsonify(sessions_data), 200

@sessions_bp.route('/sessions/<int:session_id>', methods=['GET'])
@jwt_required()
def get_session(session_id):
    """Get a specific chat session"""
    user_id = int(get_jwt_identity())
    session = ChatSession.query.filter_by(id=session_id, user_id=user_id).first()
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    return jsonify({
        'id': session.id,
        'title': session.title,
        'created_at': session.created_at.isoformat(),
        'updated_at': session.updated_at.isoformat()
    }), 200

@sessions_bp.route('/sessions', methods=['POST'])
@jwt_required()
def create_session():
    """Create a new chat session"""
    user_id = int(get_jwt_identity())
    
    new_session = ChatSession(
        user_id=user_id,
        title="New Chat"
    )
    
    db.session.add(new_session)
    db.session.commit()
    
    return jsonify({
        'id': new_session.id,
        'title': new_session.title,
        'created_at': new_session.created_at.isoformat(),
        'updated_at': new_session.updated_at.isoformat(),
        'message_count': 0
    }), 201

@sessions_bp.route('/sessions/<int:session_id>', methods=['DELETE'])
@jwt_required()
def delete_session(session_id):
    """Delete a chat session"""
    user_id = int(get_jwt_identity())
    session = ChatSession.query.filter_by(id=session_id, user_id=user_id).first()
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    # Delete all messages in this session
    ChatHistory.query.filter_by(session_id=session_id).delete()
    
    # Delete the session
    db.session.delete(session)
    db.session.commit()
    
    return jsonify({"message": "Session deleted successfully"}), 200

@sessions_bp.route('/sessions/<int:session_id>/history', methods=['GET'])
@jwt_required()
def get_session_history(session_id):
    """Get chat history for a specific session"""
    user_id = int(get_jwt_identity())
    
    # Verify session belongs to user
    session = ChatSession.query.filter_by(id=session_id, user_id=user_id).first()
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    # Get chat history for this session
    history = ChatHistory.query.filter_by(session_id=session_id).order_by(ChatHistory.timestamp.asc()).all()
    
    history_data = []
    for item in history:
        history_data.append({
            'query': item.query,
            'response': item.response_data,
            'timestamp': item.timestamp.isoformat()
        })
    
    return jsonify(history_data), 200
