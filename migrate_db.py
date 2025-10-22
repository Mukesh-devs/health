"""
Database migration script to add ChatSession table and update ChatHistory
"""
from datetime import datetime, timezone
import sqlite3
import sys

def migrate():
    # Get the database path before importing app
    db_path = 'instance/health_app.db'
    
    # First, add the column using raw SQL
    print("Starting database migration...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if session_id column exists in chat_history
    cursor.execute("PRAGMA table_info(chat_history)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'session_id' not in columns:
        print("Adding session_id column to chat_history table...")
        cursor.execute("ALTER TABLE chat_history ADD COLUMN session_id INTEGER")
        conn.commit()
        print("✓ session_id column added")
    else:
        print("✓ session_id column already exists")
    
    # Create chat_session table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title VARCHAR(200) NOT NULL DEFAULT 'New Chat',
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            FOREIGN KEY(user_id) REFERENCES user (id)
        )
    """)
    conn.commit()
    print("✓ chat_session table created/verified")
    
    # Now migrate orphaned chats
    cursor.execute("""
        SELECT DISTINCT user_id FROM chat_history WHERE session_id IS NULL
    """)
    users_with_orphaned_chats = cursor.fetchall()
    
    for (user_id,) in users_with_orphaned_chats:
        cursor.execute("SELECT username FROM user WHERE id = ?", (user_id,))
        user_result = cursor.fetchone()
        username = user_result[0] if user_result else f"User {user_id}"
        
        print(f"\nProcessing user: {username}")
        
        # Count orphaned chats
        cursor.execute("""
            SELECT COUNT(*) FROM chat_history 
            WHERE user_id = ? AND session_id IS NULL
        """, (user_id,))
        orphaned_count = cursor.fetchone()[0]
        
        if orphaned_count > 0:
            print(f"  Found {orphaned_count} messages without session")
            
            # Create a default session
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
                INSERT INTO chat_session (user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (user_id, "Previous Conversations", now, now))
            
            session_id = cursor.lastrowid
            
            # Assign all orphaned chats to this session
            cursor.execute("""
                UPDATE chat_history 
                SET session_id = ? 
                WHERE user_id = ? AND session_id IS NULL
            """, (session_id, user_id))
            
            conn.commit()
            print(f"  ✓ Created default session (ID: {session_id}) and migrated {orphaned_count} messages")
    
    conn.close()
    print("\n✅ Migration completed successfully!")

if __name__ == "__main__":
    migrate()
