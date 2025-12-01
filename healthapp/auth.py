"""
Authentication Blueprint
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from . import db
from .models import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
    print(f"{CYAN}{BOLD}[AUTH MODULE] User Registration{RESET}")
    print(f"{CYAN}{'='*80}{RESET}")
    print(f"{YELLOW}Username: {username}{RESET}")
    
    if not username or not password:
        print(f"{YELLOW}[ERROR] Missing username or password{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")
        return jsonify({"error": "Username and password are required"}), 400
    
    if User.query.filter_by(username=username).first():
        print(f"{YELLOW}[ERROR] Username already exists{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")
        return jsonify({"error": "Username already exists"}), 400
    
    print(f"{YELLOW}[Step 1] Hashing password with bcrypt...{RESET}")
    new_user = User(username=username, password=password)
    print(f"{GREEN}✓ Password hashed successfully{RESET}")
    
    print(f"{YELLOW}[Step 2] Saving to database...{RESET}")
    db.session.add(new_user)
    db.session.commit()
    print(f"{GREEN}✓ User registered: {username}{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")
    
    return jsonify({"message": "User registered successfully"}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    """Authenticate user and return JWT token"""
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
    print(f"{CYAN}{BOLD}[AUTH MODULE] User Login{RESET}")
    print(f"{CYAN}{'='*80}{RESET}")
    print(f"{YELLOW}Username: {username}{RESET}")
    
    if not username or not password:
        print(f"{YELLOW}[ERROR] Missing credentials{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")
        return jsonify({"error": "Username and password are required"}), 400
    
    print(f"{YELLOW}[Step 1] Searching for user in database...{RESET}")
    user = User.query.filter_by(username=username).first()
    
    if not user:
        print(f"{YELLOW}[ERROR] User not found{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")
        return jsonify({"error": "Invalid username or password"}), 401
    
    print(f"{GREEN}✓ User found{RESET}")
    print(f"{YELLOW}[Step 2] Verifying password with bcrypt...{RESET}")
    
    if not user.check_password(password):
        print(f"{YELLOW}[ERROR] Password verification failed{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")
        return jsonify({"error": "Invalid username or password"}), 401
    
    print(f"{GREEN}✓ Password verified{RESET}")
    print(f"{YELLOW}[Step 3] Generating JWT access token...{RESET}")
    access_token = create_access_token(identity=str(user.id))
    print(f"{GREEN}✓ JWT token generated for user ID: {user.id}{RESET}")
    print(f"{CYAN}[AUTH MODULE] Login successful!{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")
    
    return jsonify({"access_token": access_token}), 200
