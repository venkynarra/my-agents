"""
Database utilities for the Career Assistant application.
"""
import sqlite3
import logging
from .config import DB_PATH

logger = logging.getLogger(__name__)

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Initializes the database and creates the interactions table if it doesn't exist."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("✅ Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

def log_interaction(query: str, response: str):
    """Logs a user query and the AI's response in the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO interactions (query, response) VALUES (?, ?)",
                (query, response)
            )
            conn.commit()
            logger.info(f"📊 Logged interaction for query: '{query[:30]}...'")
    except sqlite3.Error as e:
        logger.error(f"❌ Failed to log interaction: {e}")

def fetch_all_interactions():
    """Fetches all logged interactions from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, query, response, timestamp FROM interactions ORDER BY timestamp DESC")
            interactions = cursor.fetchall()
            logger.info(f"📈 Fetched {len(interactions)} interactions from the database.")
            return [dict(ix) for ix in interactions]
    except sqlite3.Error as e:
        logger.error(f"❌ Failed to fetch interactions: {e}")
        return []

if __name__ == '__main__':
    initialize_db()
    log_interaction("Test query", "Test response")
    print(fetch_all_interactions()) 