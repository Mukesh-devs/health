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
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    user_id = int(get_jwt_identity())
    
    print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
    print(f"{CYAN}{BOLD}[DATABASE MODULE] Get All Sessions{RESET}")
    print(f"{CYAN}{'='*80}{RESET}")
    print(f"{YELLOW}User ID: {user_id}{RESET}")
    
    print(f"{YELLOW}[Query] SELECT * FROM chat_sessions WHERE user_id={user_id} ORDER BY updated_at DESC{RESET}")
    sessions = db.session.scalars(
        db.select(ChatSession)
        .filter_by(user_id=user_id)
        .order_by(ChatSession.updated_at.desc())
    ).all()
    
    print(f"{GREEN}✓ Found {len(sessions)} sessions{RESET}")
    
    sessions_data = []
    for session in sessions:
        # Get message count for this session
        message_count = db.session.scalar(
            db.select(db.func.count(ChatHistory.id)).where(ChatHistory.session_id == session.id)
        )
        
        sessions_data.append({
            'id': session.id,
            'public_id': session.public_id,
            'title': session.title,
            'created_at': session.created_at.isoformat(),
            'updated_at': session.updated_at.isoformat(),
            'message_count': message_count or 0
        })

    print(f"{YELLOW}  - Session {session.id} (public_id={session.public_id}): {session.title} ({message_count or 0} messages){RESET}")    
    print(f"{CYAN}{'='*80}{RESET}\n")
    return jsonify(sessions_data), 200

@sessions_bp.route('/sessions/<public_id>', methods=['GET'])
@jwt_required()
def get_session(public_id):
    """Get a specific chat session"""
    user_id = int(get_jwt_identity())
    session = db.session.scalar(
        db.select(ChatSession).filter_by(public_id=public_id, user_id=user_id)
    )
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    return jsonify({
        'id': session.id,
        'public_id': session.public_id,
        'title': session.title,
        'created_at': session.created_at.isoformat(),
        'updated_at': session.updated_at.isoformat()
    }), 200

@sessions_bp.route('/sessions', methods=['POST'])
@jwt_required()
def create_session():
    """Create a new chat session"""
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    user_id = int(get_jwt_identity())
    
    print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
    print(f"{CYAN}{BOLD}[DATABASE MODULE] Create New Session{RESET}")
    print(f"{CYAN}{'='*80}{RESET}")
    print(f"{YELLOW}User ID: {user_id}{RESET}")
    
    new_session = ChatSession(
        user_id=user_id,
        title="New Chat"
    )
    
    print(f"{YELLOW}[Query] INSERT INTO chat_sessions (user_id, title) VALUES ({user_id}, 'New Chat'){RESET}")
    db.session.add(new_session)
    db.session.commit()
    
    print(f"{GREEN}✓ Session created with ID: {new_session.id}, public_id: {new_session.public_id}{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")
    
    return jsonify({
        'id': new_session.id,
        'public_id': new_session.public_id,
        'title': new_session.title,
        'created_at': new_session.created_at.isoformat(),
        'updated_at': new_session.updated_at.isoformat(),
        'message_count': 0
    }), 201

@sessions_bp.route('/sessions/<public_id>', methods=['DELETE'])
@jwt_required()
def delete_session(public_id):
    """Delete a chat session"""
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    user_id = int(get_jwt_identity())
    
    print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
    print(f"{CYAN}{BOLD}[DATABASE MODULE] Delete Session{RESET}")
    print(f"{CYAN}{'='*80}{RESET}")
    print(f"{YELLOW}User ID: {user_id}{RESET}")
    print(f"{YELLOW}Public ID: {public_id}{RESET}")
    
    session = db.session.scalar(
        db.select(ChatSession).filter_by(public_id=public_id, user_id=user_id)
    )    
    if not session:
        print(f"{YELLOW}[ERROR] Session not found or unauthorized{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")
        return jsonify({"error": "Session not found"}), 404

    session_id = session.id
    # Delete all messages in this session
    print(f"{YELLOW}[Query] DELETE FROM chat_history WHERE session_id={session_id}{RESET}")
    db.session.execute(
        db.delete(ChatHistory).where(ChatHistory.session_id == session_id)
    )
    print(f"{GREEN}✓ Messages deleted{RESET}")

    print(f"{YELLOW}[Query] DELETE FROM chat_sessions WHERE id={session_id}{RESET}")
    db.session.delete(session)
    db.session.commit()

    print(f"{GREEN}✓ Session {session_id} (public_id={public_id}) deleted successfully{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")

    return jsonify({"message": "Session deleted successfully"}), 200

@sessions_bp.route('/sessions/<public_id>/history', methods=['GET'])
@jwt_required()
def get_session_history(public_id):
    """Get chat history for a specific session"""
    user_id = int(get_jwt_identity())
    
    # Verify session belongs to user
    session = db.session.scalar(
        db.select(ChatSession).filter_by(public_id=public_id, user_id=user_id)
    )
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    # Get chat history for this session
    history = db.session.scalars(
        db.select(ChatHistory)
        .filter_by(session_id=session_id)
        .order_by(ChatHistory.timestamp.asc())
    ).all()
    
    history_data = []
    for item in history:
        history_data.append({
            'query': item.query,
            'response': item.response_data,
            'timestamp': item.timestamp.isoformat()
        })
    
    return jsonify(history_data), 200
