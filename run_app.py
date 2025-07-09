import subprocess
import sys
import time
import os

def main():
    """
    Starts the backend AI server and the frontend Gradio web app.
    Redirects backend output to log files for debugging.
    """
    backend_process = None
    frontend_process = None
    
    python_executable = sys.executable
    backend_command = [python_executable, "-m", "foundations.ai_server"]
    frontend_command = [python_executable, "-m", "foundations.mcp_enhanced_app"]

    # Open log files for the backend process
    try:
        print("🚀 Starting backend AI server...")
        backend_process = subprocess.Popen(
            backend_command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        print("✅ Backend server process started.")
        
        # Give the backend a moment to start before launching the frontend
        time.sleep(5) # Reduced wait time

        # Check if the backend process has already terminated
        if backend_process.poll() is not None:
            print("❌ Backend server failed to start.")
            return

        print("\n🚀 Starting frontend Gradio application...")
        frontend_process = subprocess.Popen(
            frontend_command,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print("✅ Frontend application process started.")
        
        # Wait for the frontend to complete
        if frontend_process:
            frontend_process.wait()

    except KeyboardInterrupt:
        print("\n🚫 Shutdown requested by user.")
    except Exception as e:
        print(f"\n🔥 An unexpected error occurred: {e}")
    finally:
        print("\n🔌 Shutting down all processes...")
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
            print("Frontend process terminated.")
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            print("Backend process terminated.")

if __name__ == "__main__":
    main() 