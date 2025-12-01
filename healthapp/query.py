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
    """Handle user query and return AI response"""
    query_start_time = time.time()
    
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        user_query = data.get("query", "").strip()
        session_id = data.get("session_id")
        answer_length = data.get("answerLength", "Moderate")
        
        print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
        print(f"{CYAN}{BOLD}[QUERY MODULE] New Query Received{RESET}")
        print(f"{CYAN}{'='*80}{RESET}")
        print(f"{BLUE}User ID:{RESET} {current_user_id}")
        print(f"{BLUE}Session ID:{RESET} {session_id}")
        print(f"{BLUE}Query:{RESET} {user_query}")
        print(f"{BLUE}Answer Length:{RESET} {answer_length}")
        
        if not user_query:
            print(f"{YELLOW}[WARNING] Empty query received{RESET}")
            return jsonify({"error": "Empty query"}), 400

        # Handle rule-based responses
        print(f"\n{YELLOW}{'─'*80}{RESET}")
        print(f"{YELLOW}[STEP 1/8] Rule-Based Response Check{RESET}")
        print(f"{YELLOW}{'─'*80}{RESET}")
        print(f"{CYAN}Checking if query matches predefined patterns...{RESET}")
        print(f"{BLUE}  Query (lowercase): '{user_query.lower()}'{RESET}")
        print(f"{BLUE}  Available rules: {list(RULE_BASED_RESPONSES.keys())}{RESET}")
        
        query_lower = user_query.lower().strip()
        if query_lower in RULE_BASED_RESPONSES:
            print(f"{GREEN}✓ RULE MATCH FOUND!{RESET}")
            print(f"{GREEN}  Matched rule: '{query_lower}'{RESET}")
            print(f"{GREEN}  Response: {RULE_BASED_RESPONSES[query_lower]}{RESET}")
            print(f"{GREEN}  Action: Skip LLM, return predefined response{RESET}")
            
            response_data = {"predefinedResponse": RULE_BASED_RESPONSES[query_lower]}
            history_entry = ChatHistory(
                user_id=current_user_id, 
                session_id=session_id, 
                query=user_query, 
                response_data=response_data
            )
            
            print(f"\n{YELLOW}[STEP 2/8] Saving to database...{RESET}")
            print(f"{BLUE}  Table: chat_history{RESET}")
            print(f"{BLUE}  User ID: {current_user_id}{RESET}")
            print(f"{BLUE}  Session ID: {session_id}{RESET}")
            db.session.add(history_entry)
            db.session.commit()
            
            # Update session timestamp
            if session_id:
                session = db.session.get(ChatSession, session_id)
                if session:
                    session.updated_at = datetime.now(timezone.utc)
                    db.session.commit()
            
            print(f"{GREEN}✓ Database saved successfully{RESET}")
            
            # Log query metrics (predefined response - no entities/triples)
            query_logger.log_query_metrics(
                query=user_query,
                latency=time.time() - query_start_time,
                entities_found=0,
                triples_extracted=0,
                session_id=session_id
            )
            
            total_time = time.time() - query_start_time
            print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
            print(f"{CYAN}{BOLD}[QUERY MODULE] Completed with Rule-Based Response{RESET}")
            print(f"{CYAN}{BOLD}{'='*80}{RESET}")
            print(f"{GREEN}✓ Total time: {total_time:.3f}s{RESET}")
            print(f"{GREEN}✓ Response type: Predefined rule{RESET}")
            print(f"{GREEN}✓ LLM calls: 0{RESET}")
            print(f"{CYAN}{'='*80}{RESET}\n")
            return jsonify(response_data)
        
        print(f"{YELLOW}⚠ No rule match found{RESET}")
        print(f"{YELLOW}  Action: Proceed to LLM for intelligent response{RESET}")

        # Get recent chats from the same session if provided
        print(f"\n{YELLOW}{'─'*80}{RESET}")
        print(f"{YELLOW}[STEP 2/8] Loading Conversation History{RESET}")
        print(f"{YELLOW}{'─'*80}{RESET}")
        print(f"{CYAN}Retrieving conversation context from database...{RESET}")
        
        if session_id:
            print(f"{BLUE}  Filter: session_id={session_id}, user_id={current_user_id}{RESET}")
            recent_chats_query = (
                db.select(ChatHistory)
                .filter_by(user_id=current_user_id, session_id=session_id)
                .order_by(ChatHistory.timestamp.desc())
                .limit(5)
            )
        else:
            print(f"{BLUE}  Filter: user_id={current_user_id} (all sessions){RESET}")
            recent_chats_query = (
                db.select(ChatHistory)
                .filter_by(user_id=current_user_id)
                .order_by(ChatHistory.timestamp.desc())
                .limit(5)
            )
        
        print(f"{BLUE}  Limit: 5 most recent messages{RESET}")
        recent_chats = db.session.scalars(recent_chats_query).all()
        recent_chats = list(reversed(recent_chats))
        
        print(f"{GREEN}✓ Loaded {len(recent_chats)} previous messages{RESET}")
        if recent_chats:
            print(f"{BLUE}  Context window: {len(recent_chats)} messages{RESET}")
            for i, chat in enumerate(recent_chats[-3:], 1):  # Show last 3
                preview = chat.query[:50] + "..." if len(chat.query) > 50 else chat.query
                print(f"{BLUE}    {i}. [{chat.timestamp.strftime('%H:%M:%S')}] {preview}{RESET}")
        else:
            print(f"{YELLOW}  ℹ No previous conversation history{RESET}")

        # Build conversation history for API
        print(f"\n{CYAN}Building conversation history for LLM context...{RESET}")
        history_for_api = []
        for chat in recent_chats:
            history_for_api.append({'source': 'user', 'text': chat.query})
            ai_text = (
                chat.response_data.get('predefinedResponse') or 
                chat.response_data.get('textualResponse')
            )
            if ai_text:
                history_for_api.append({'source': 'model', 'text': ai_text})
        
        print(f"{GREEN}✓ History prepared: {len(history_for_api)} messages{RESET}")
        
        # Calculate token estimate
        total_chars = sum(len(msg['text']) for msg in history_for_api)
        estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
        print(f"{BLUE}  Total characters: {total_chars:,}{RESET}")
        print(f"{BLUE}  Estimated tokens: ~{estimated_tokens}{RESET}")

        # Prepare system instruction
        print(f"\n{YELLOW}{'─'*80}{RESET}")
        print(f"{YELLOW}[STEP 3/8] Preparing LLM Request{RESET}")
        print(f"{YELLOW}{'─'*80}{RESET}")
        print(f"{CYAN}Creating system instruction for LLM...{RESET}")
        
        system_instruction_text = f"""You are an expert health information system. Your task is to respond to the user's query and extract key facts as a list of subject-relation-object triples.
- Provide a factual, concise response in the 'textualResponse' field.
- **Critically, you must also provide a 'highlightedResponse'. This response should be identical to the textualResponse, but with all extracted subject and object entities wrapped in '*|' and '|*' markers.** For example, if a sentence is "Aspirin treats headaches", the highlighted version must be "*|Aspirin|* treats *|headaches|*".
- Identify and extract key entities and relationships into the 'triples' field.
- Follow the JSON schema strictly. Do not add extra explanations.
- The user's desired answer length is "{answer_length}"."""
        
        system_instruction = {
            "role": "system",
            "parts": [{
                "text": system_instruction_text
            }]
        }
        
        print(f"{GREEN}✓ System instruction created{RESET}")
        print(f"{BLUE}  Length: {len(system_instruction_text)} characters{RESET}")
        print(f"{BLUE}  Answer length preference: {answer_length}{RESET}")
        print(f"{BLUE}  Response format: JSON with schema validation{RESET}")
        
        # Define response schema
        print(f"\n{CYAN}Defining response schema for structured output...{RESET}")
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
        print(f"{GREEN}✓ Response schema defined{RESET}")
        print(f"{BLUE}  Required fields: textualResponse, highlightedResponse, triples{RESET}")
        print(f"{BLUE}  Triple fields: subject, relation, object{RESET}")
        
        # Format history for Gemini API
        print(f"\n{CYAN}Formatting conversation history for Gemini API...{RESET}")
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
        print(f"{GREEN}✓ Formatted {len(formatted_history)} messages for LLM{RESET}")
        print(f"{BLUE}  User messages: {sum(1 for m in formatted_history if m['role'] == 'user')}{RESET}")
        print(f"{BLUE}  Model messages: {sum(1 for m in formatted_history if m['role'] == 'model')}{RESET}")

        # Prepare API payload
        print(f"\n{CYAN}Assembling final API payload...{RESET}")
        payload = {
            "contents": formatted_history,
            "system_instruction": system_instruction,
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        }
        
        payload_size = len(json.dumps(payload))
        print(f"{GREEN}✓ Payload assembled{RESET}")
        print(f"{BLUE}  Payload size: {payload_size:,} bytes (~{payload_size/1024:.2f} KB){RESET}")
        print(f"{BLUE}  Contents messages: {len(formatted_history)}{RESET}")
        print(f"{BLUE}  Response format: application/json{RESET}")
        
        # Call Gemini API
        print(f"\n{YELLOW}{'─'*80}{RESET}")
        print(f"{YELLOW}[STEP 4/8] Calling Gemini LLM API{RESET}")
        print(f"{YELLOW}{'─'*80}{RESET}")
        
        api_url = current_app.config['GEMINI_API_URL']
        print(f"{CYAN}Sending request to Gemini API...{RESET}")
        print(f"{BLUE}  API URL: {api_url[:80]}...{RESET}")
        print(f"{BLUE}  Model: gemini-1.5-flash{RESET}")
        print(f"{BLUE}  Request method: POST{RESET}")
        print(f"{BLUE}  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}{RESET}")
        
        llm_start_time = time.time()
        print(f"{MAGENTA}⏳ Waiting for LLM response...{RESET}")
        
        response = requests.post(
            api_url, 
            headers={'Content-Type': 'application/json'}, 
            data=json.dumps(payload)
        )
        response.raise_for_status()
        llm_duration = time.time() - llm_start_time
        
        print(f"{GREEN}✓ LLM API call completed successfully!{RESET}")
        print(f"{GREEN}  Response time: {llm_duration:.3f}s{RESET}")
        print(f"{GREEN}  Status code: {response.status_code}{RESET}")
        
        response_size = len(response.content)
        print(f"{BLUE}  Response size: {response_size:,} bytes (~{response_size/1024:.2f} KB){RESET}")
        
        print(f"\n{YELLOW}{'─'*80}{RESET}")
        print(f"{YELLOW}[STEP 5/8] Processing LLM Response{RESET}")
        print(f"{YELLOW}{'─'*80}{RESET}")
        
        print(f"{CYAN}Parsing JSON response from LLM...{RESET}")
        llm_response = response.json()
        
        print(f"{BLUE}  Response structure:{RESET}")
        print(f"{BLUE}    - candidates: {len(llm_response.get('candidates', []))}{RESET}")
        
        if not llm_response.get('candidates'):
            print(f"{RED}✗ ERROR: No candidates in LLM response{RESET}")
            raise ValueError("Invalid LLM response: No candidates found.")
        
        print(f"{GREEN}✓ Found {len(llm_response['candidates'])} candidate(s){RESET}")
        
        # Extract the text from the first candidate
        llm_text = llm_response['candidates'][0]['content']['parts'][0]['text']
        print(f"{BLUE}  Raw LLM output size: {len(llm_text)} characters{RESET}")
        
        print(f"\n{CYAN}Parsing structured JSON output...{RESET}")
        llm_data = json.loads(llm_text)
        
        print(f"{GREEN}✓ JSON parsed successfully{RESET}")
        print(f"{BLUE}  Keys found: {list(llm_data.keys())}{RESET}")

        textual_response = llm_data.get(
            'textualResponse', 
            'I could not generate a textual response.'
        )
        highlighted_response = llm_data.get('highlightedResponse', textual_response)
        triples = llm_data.get('triples', [])
        
        print(f"\n{CYAN}Extracting response components...{RESET}")
        print(f"{GREEN}✓ Textual response: {len(textual_response)} characters{RESET}")
        print(f"{BLUE}  Preview: {textual_response[:100]}...{RESET}")
        
        print(f"{GREEN}✓ Highlighted response: {len(highlighted_response)} characters{RESET}")
        entity_markers = highlighted_response.count('*|')
        print(f"{BLUE}  Entity markers: {entity_markers} entities highlighted{RESET}")
        
        print(f"{GREEN}✓ Triples extracted: {len(triples)}{RESET}")
        if triples:
            print(f"{BLUE}  Sample triples:{RESET}")
            for i, triple in enumerate(triples[:3], 1):
                print(f"{BLUE}    {i}. ({triple['subject']}, {triple['relation']}, {triple['object']}){RESET}")
            if len(triples) > 3:
                print(f"{BLUE}    ... and {len(triples) - 3} more{RESET}")

        # Build graph data
        print(f"\n{YELLOW}{'─'*80}{RESET}")
        print(f"{YELLOW}[STEP 6/8] Building Knowledge Graph and Validating Triples{RESET}")
        print(f"{YELLOW}{'─'*80}{RESET}")
        
        print(f"{CYAN}Creating graph structure from extracted triples...{RESET}")
        nodes, edges, node_ids = [], [], set()
        validated_triples = []

        print(f"{BLUE}  Total triples to process: {len(triples)}{RESET}")
        
        for idx, triple in enumerate(triples, 1):
            subject = triple.get('subject', '').strip()
            relation = triple.get('relation', '').strip()
            obj = triple.get('object', '').strip()

            if not all([subject, relation, obj]):
                print(f"{YELLOW}  [{idx}/{len(triples)}] Skipping incomplete triple{RESET}")
                continue

            print(f"\n{CYAN}[{idx}/{len(triples)}] Processing triple:{RESET}")
            print(f"{BLUE}  Subject: {subject}{RESET}")
            print(f"{BLUE}  Relation: {relation}{RESET}")
            print(f"{BLUE}  Object: {obj}{RESET}")

            # Add nodes
            if subject not in node_ids:
                nodes.append({"id": subject, "label": subject})
                node_ids.add(subject)
                print(f"{GREEN}  ✓ Added subject node: {subject}{RESET}")
            
            if obj not in node_ids:
                nodes.append({"id": obj, "label": obj})
                node_ids.add(obj)
                print(f"{GREEN}  ✓ Added object node: {obj}{RESET}")
            
            # Add edge
            edges.append({"from": subject, "to": obj, "label": relation})
            print(f"{GREEN}  ✓ Added edge: {subject} --[{relation}]--> {obj}{RESET}")

            # Validate triple
            print(f"{MAGENTA}  🔍 Validating triple against knowledge base...{RESET}")
            validation = validate_triple(subject, relation, obj)
            validated_triples.append({**triple, **validation})
            
            # Show validation status
            status_display = validation.get('status', 'unknown')
            evidence_quality = validation.get('evidence_quality', 'none')
            
            print(f"{GREEN}  ✓ Validation complete{RESET}")
            print(f"{BLUE}    Status: {status_display.upper()}{RESET}")
            print(f"{BLUE}    Evidence quality: {evidence_quality.upper()}{RESET}")
        
        print(f"\n{GREEN}✓ Graph construction complete{RESET}")
        print(f"{GREEN}  Total nodes: {len(nodes)}{RESET}")
        print(f"{GREEN}  Total edges: {len(edges)}{RESET}")
        print(f"{GREEN}  Validated triples: {len(validated_triples)}{RESET}")
        print(f"{GREEN}  Unique entities: {len(node_ids)}{RESET}")
        
        # Generate recommendations
        print(f"\n{YELLOW}{'─'*80}{RESET}")
        print(f"{YELLOW}[STEP 7/8] Generating Follow-up Recommendations{RESET}")
        print(f"{YELLOW}{'─'*80}{RESET}")
        
        print(f"{CYAN}Analyzing entity relationships for recommendations...{RESET}")
        print(f"{BLUE}  Input entities: {list(node_ids)[:5]}{RESET}")
        if len(node_ids) > 5:
            print(f"{BLUE}  ... and {len(node_ids) - 5} more{RESET}")
        
        recommendations = generate_recommendations(list(node_ids))
        
        print(f"{GREEN}✓ Generated {len(recommendations)} recommendations{RESET}")
        if recommendations:
            print(f"{BLUE}  Sample recommendations:{RESET}")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"{BLUE}    {i}. {rec}{RESET}")
            if len(recommendations) > 3:
                print(f"{BLUE}    ... and {len(recommendations) - 3} more{RESET}")
        
        # Calculate total response time
        total_response_time = time.time() - query_start_time
        
        # Build final response
        print(f"\n{YELLOW}{'─'*80}{RESET}")
        print(f"{YELLOW}[STEP 8/8] Saving Results and Finalizing{RESET}")
        print(f"{YELLOW}{'─'*80}{RESET}")
        
        print(f"{CYAN}Assembling final response payload...{RESET}")
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
        
        print(f"{GREEN}✓ Response payload assembled{RESET}")
        print(f"{BLUE}  Components: textualResponse, highlightedResponse, graphData, validatedTriples, recommendations, performance{RESET}")

        # Save to chat history
        print(f"\n{CYAN}Saving to database...{RESET}")
        print(f"{BLUE}  Table: chat_history{RESET}")
        print(f"{BLUE}  User ID: {current_user_id}{RESET}")
        print(f"{BLUE}  Session ID: {session_id}{RESET}")
        
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
                    print(f"{GREEN}✓ Session title updated: {title}{RESET}")
        
        db.session.commit()
        print(f"{GREEN}✓ Database saved successfully{RESET}")
        
        # Log query metrics with entities and triples
        query_logger.log_query_metrics(
            query=data.get("query", ""),
            latency=total_response_time,
            entities_found=len(node_ids),
            triples_extracted=len(triples),
            session_id=session_id
        )
        
        print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
        print(f"{CYAN}{BOLD}[QUERY MODULE] Query Processing Complete!{RESET}")
        print(f"{CYAN}{BOLD}{'='*80}{RESET}")
        
        print(f"\n{GREEN}📊 PERFORMANCE SUMMARY:{RESET}")
        print(f"{GREEN}  Total processing time: {total_response_time:.3f}s{RESET}")
        print(f"{GREEN}  LLM API time: {llm_duration:.3f}s ({llm_duration/total_response_time*100:.1f}%){RESET}")
        print(f"{GREEN}  Other processing: {total_response_time - llm_duration:.3f}s ({(1-llm_duration/total_response_time)*100:.1f}%){RESET}")
        
        print(f"\n{GREEN}📈 RESULTS SUMMARY:{RESET}")
        print(f"{GREEN}  Entities found: {len(node_ids)}{RESET}")
        print(f"{GREEN}  Triples extracted: {len(triples)}{RESET}")
        print(f"{GREEN}  Validated triples: {len(validated_triples)}{RESET}")
        print(f"{GREEN}  Recommendations: {len(recommendations)}{RESET}")
        print(f"{GREEN}  Graph nodes: {len(nodes)}{RESET}")
        print(f"{GREEN}  Graph edges: {len(edges)}{RESET}")
        
        print(f"\n{GREEN}✓ Response ready to send to client{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")
        
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
