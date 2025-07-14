#!/usr/bin/env python3
"""
Enhanced AI Career Assistant Startup Script
Always uses the enhanced_gradio_app.py for the best experience
"""
import subprocess
import sys
import os

def main():
    print("""
🚀 Starting Enhanced AI Career Assistant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
✨ Features:
• Multi-Agent System Architecture
• Real-time analytics dashboard
• AI-powered chat with enhanced routing
• Comprehensive profile display
• Professional contact form
• Meeting scheduling integration
• Sub-2-second response times
• Production-ready for cloud deployment

🎯 Using: enhanced_gradio_app.py (Full-Featured Version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Run the enhanced Gradio app (only version available now)
        subprocess.run([sys.executable, "enhanced_gradio_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Enhanced AI Career Assistant stopped by user")
    except Exception as e:
        print(f"❌ Error running enhanced app: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        print("💡 Note: run_app.py has been removed - use enhanced_gradio_app.py only")
        sys.exit(1)

if __name__ == "__main__":
    main() 