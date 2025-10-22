"""
Query Processing Blueprint
"""
import time
import json
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, g
from flask_jwt_extended import jwt_required, get_jwt_identity
import requests
from . import db
from .models import ChatHistory, ChatSession
from .embedding import (
    validate_triple, 
    generate_recommendations, 
    get_cui_from_name
)
from .metrics_logger import query_logger

query_bp = Blueprint('query', __name__)

# Rule-Based Dialog Manager
RULE_BASED_RESPONSES = {
    "hello": "Hello again! How can I assist you with your health questions today?",
    "hi": "Hi there! What health information can I help you explore?",
    "how are you": "I'm a computer program, but I'm functioning perfectly. Thanks for asking! What can I help you with?",
    "thanks": "You're welcome! Let me know if you have more questions.",
    "thank you": "You're most welcome! Is there anything else I can help you explore?"
}

@query_bp.route('/query', methods=['POST'])
@jwt_required()
def handle_query():
    """Main query processing endpoint"""
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

        # Handle rule-based responses
        if user_query in RULE_BASED_RESPONSES:
            response_data = {"predefinedResponse": RULE_BASED_RESPONSES[user_query]}
            history_entry = ChatHistory(
                user_id=current_user_id, 
                session_id=session_id, 
                query=user_query, 
                response_data=response_data
            )
            db.session.add(history_entry)
            db.session.commit()
            
            # Update session timestamp
            if session_id:
                session = db.session.get(ChatSession, session_id)
                if session:
                    session.updated_at = datetime.now(timezone.utc)
                    db.session.commit()
            
            # Log query metrics (predefined response - no entities/triples)
            query_logger.log_query_metrics(
                query=user_query,
                latency=time.time() - query_start_time,
                entities_found=0,
                triples_extracted=0,
                session_id=session_id
            )
            return jsonify(response_data)

        # Get recent chats from the same session if provided
        if session_id:
            recent_chats_query = (
                db.select(ChatHistory)
                .filter_by(user_id=current_user_id, session_id=session_id)
                .order_by(ChatHistory.timestamp.desc())
                .limit(5)
            )
        else:
            recent_chats_query = (
                db.select(ChatHistory)
                .filter_by(user_id=current_user_id)
                .order_by(ChatHistory.timestamp.desc())
                .limit(5)
            )
        
        recent_chats = db.session.scalars(recent_chats_query).all()
        recent_chats = list(reversed(recent_chats))

        # Build conversation history for API
        history_for_api = []
        for chat in recent_chats:
            history_for_api.append({'source': 'user', 'text': chat.query})
            ai_text = (
                chat.response_data.get('predefinedResponse') or 
                chat.response_data.get('textualResponse')
            )
            if ai_text:
                history_for_api.append({'source': 'model', 'text': ai_text})

        # Prepare system instruction
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
        
        # Define response schema
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "textualResponse": {"type": "STRING"},
                "highlightedResponse": {"type": "STRING"},
                "triples": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "subject": {"type": "STRING"}, 
                            "relation": {"type": "STRING"}, 
                            "object": {"type": "STRING"}
                        },
                        "required": ["subject", "relation", "object"]
                    }
                }
            },
            "required": ["textualResponse", "highlightedResponse", "triples"]
        }
        
        # Format history for Gemini API
        formatted_history = []
        for entry in history_for_api:
            role = "user" if entry.get('source') == 'user' else "model"
            formatted_history.append({
                "role": role, 
                "parts": [{"text": entry.get('text', '')}]
            })
        
        formatted_history.append({
            "role": "user", 
            "parts": [{"text": user_query}]
        })

        # Prepare API payload
        payload = {
            "contents": formatted_history,
            "system_instruction": system_instruction,
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        }
        
        # Call Gemini API
        api_url = current_app.config['GEMINI_API_URL']
        llm_start_time = time.time()
        response = requests.post(
            api_url, 
            headers={'Content-Type': 'application/json'}, 
            data=json.dumps(payload)
        )
        response.raise_for_status()
        llm_duration = time.time() - llm_start_time
        
        llm_response = response.json()
        
        if not llm_response.get('candidates'):
            raise ValueError("Invalid LLM response: No candidates found.")
        
        llm_text = llm_response['candidates'][0]['content']['parts'][0]['text']
        llm_data = json.loads(llm_text)

        textual_response = llm_data.get(
            'textualResponse', 
            'I could not generate a textual response.'
        )
        highlighted_response = llm_data.get('highlightedResponse', textual_response)
        triples = llm_data.get('triples', [])

        # Build graph data
        nodes, edges, node_ids = [], [], set()
        validated_triples = []

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
        
        # Generate recommendations
        recommendations = generate_recommendations(list(node_ids))
        
        # Calculate total response time
        total_response_time = time.time() - query_start_time
        
        # Build final response
        final_response_data = {
            "textualResponse": textual_response,
            "highlightedResponse": highlighted_response,
            "graphData": {"nodes": nodes, "edges": edges},
            "validatedTriples": validated_triples,
            "recommendations": recommendations,
            "performance": {
                "latency_seconds": round(total_response_time, 3),
                "response_time_seconds": round(total_response_time, 3),
                "llm_time_seconds": round(llm_duration, 3),
                "entities_found": len(node_ids),
                "triples_extracted": len(triples),
            }
        }

        # Save to chat history
        history_entry = ChatHistory(
            user_id=current_user_id,
            session_id=session_id,
            query=data.get("query", ""),
            response_data=final_response_data
        )
        db.session.add(history_entry)
        
        # Update session timestamp and title
        if session_id:
            session = db.session.get(ChatSession, session_id)
            if session:
                session.updated_at = datetime.now(timezone.utc)
                
                # Auto-generate title from first query if still "New Chat"
                if session.title == "New Chat":
                    # Use first few words of query as title
                    query_words = data.get("query", "").strip().split()
                    title = " ".join(query_words[:6])
                    if len(query_words) > 6:
                        title += "..."
                    session.title = title
        
        db.session.commit()
        
        # Log query metrics with entities and triples
        query_logger.log_query_metrics(
            query=data.get("query", ""),
            latency=total_response_time,
            entities_found=len(node_ids),
            triples_extracted=len(triples),
            session_id=session_id
        )
        
        return jsonify(final_response_data), 200
        
    except Exception as e:
        # Log failed query
        query_logger.log_query_metrics(
            query=data.get("query", "") if data else "",
            latency=time.time() - query_start_time,
            entities_found=0,
            triples_extracted=0,
            session_id=data.get("session_id") if data else None
        )
        print(f"Error in query processing: {str(e)}")
        return jsonify({"error": str(e)}), 500

@query_bp.route('/history', methods=['GET'])
@jwt_required()
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
        
    return jsonify(formatted_history), 200
