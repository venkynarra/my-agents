#!/usr/bin/env python3
"""
Startup script for MCP Enhanced Career Assistant
Initializes all components and launches the application
"""

import os
import sys
import logging
import asyncio
import subprocess
from pathlib import Path
import sqlite3
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp-startup")

def setup_environment():
    """Setup environment and dependencies."""
    print("🔧 Setting up environment...")
    
    # Ensure we're in the correct directory
    os.chdir(Path(__file__).parent)
    
    # Add current directory to Python path
    current_dir = str(Path.cwd())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    print("✅ Environment setup complete")

def clear_caches():
    """Clear all caches and temporary files."""
    print("🧹 Clearing caches...")
    
    cache_files = [
        "response_cache.db",
        "analytics.db", 
        "career_analytics.db",
        "rag_index",
        "__pycache__"
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                if os.path.isfile(cache_file):
                    os.remove(cache_file)
                    print(f"   🗑️  Removed {cache_file}")
                elif os.path.isdir(cache_file):
                    shutil.rmtree(cache_file)
                    print(f"   🗑️  Removed directory {cache_file}")
            except PermissionError:
                print(f"   ⚠️  Could not remove {cache_file} (permission denied)")
            except Exception as e:
                print(f"   ⚠️  Could not remove {cache_file}: {e}")
    
    print("✅ Caches cleared")

def verify_knowledge_base():
    """Verify knowledge base files exist."""
    print("📚 Verifying knowledge base...")
    
    knowledge_base_path = Path("../agent_knowledge")
    required_files = [
        "resume.md",
        "experience_and_projects.md",
        "profile.md",
        "faq.md",
        "github_readme.md"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = knowledge_base_path / file
        if file_path.exists():
            print(f"   ✅ {file} - {file_path.stat().st_size} bytes")
        else:
            missing_files.append(file)
            print(f"   ❌ {file} - MISSING")
    
    if missing_files:
        print(f"⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ Knowledge base verified")
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "gradio",
        "mcp",
        "httpx",
        "websockets",
        "pydantic",
        "sqlite3"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} - NOT INSTALLED")
    
    # sqlite3 is built-in
    if "sqlite3" in missing_packages:
        missing_packages.remove("sqlite3")
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies available")
    return True

def initialize_database():
    """Initialize the analytics database."""
    print("💾 Initializing database...")
    
    try:
        conn = sqlite3.connect("career_analytics.db")
        cursor = conn.cursor()
        
        # Create analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_query TEXT NOT NULL,
                response_type TEXT NOT NULL,
                response_length INTEGER,
                source_files TEXT,
                user_feedback TEXT,
                session_id TEXT
            )
        ''')
        
        # Create contact submissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contact_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                company TEXT,
                message TEXT,
                response_sent BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def run_mcp_enhanced_app():
    """Run the MCP enhanced application."""
    print("🚀 Starting MCP Enhanced Career Assistant...")
    
    try:
        # Import and run the enhanced app
        from mcp_enhanced_app import main
        main()
        
    except ImportError as e:
        print(f"❌ Failed to import MCP enhanced app: {e}")
        print("💡 Please ensure mcp_enhanced_app.py is present")
        return False
    
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    print("=" * 60)
    print("🤖 MCP Enhanced Career Assistant - Startup")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Clear caches for fresh start
    try:
        clear_caches()
    except Exception as e:
        print(f"⚠️  Cache clearing encountered issues: {e}")
        print("🔄 Continuing with application startup...")
    
    # Verify knowledge base
    if not verify_knowledge_base():
        print("❌ Knowledge base verification failed!")
        print("📋 Please ensure all knowledge base files are present in ../agent_knowledge/")
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed!")
        print("📦 Please install missing dependencies")
        return False
    
    # Initialize database
    if not initialize_database():
        print("❌ Database initialization failed!")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All checks passed! Starting application...")
    print("=" * 60)
    
    # Start the application
    success = run_mcp_enhanced_app()
    
    if success:
        print("\n✅ Application started successfully!")
        print("🌐 Access at: http://localhost:7860")
        print("💡 All responses use actual resume data")
        print("🔄 Real-time knowledge base integration active")
    else:
        print("\n❌ Application startup failed!")
        print("📧 Contact: venkateshnarra368@gmail.com")
    
    return success

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("📧 Contact: venkateshnarra368@gmail.com") 