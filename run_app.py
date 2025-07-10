import subprocess
import sys
import time
import grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
import os

def wait_for_grpc_server(service_name: str, timeout: int = 90):
    """
    Waits for a gRPC server to report its status as SERVING.
    """
    print(f"⏳ Waiting for gRPC service '{service_name}' to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            with grpc.insecure_channel('localhost:50051') as channel:
                stub = health_pb2_grpc.HealthStub(channel)
                request = health_pb2.HealthCheckRequest(service=service_name)
                response = stub.Check(request, timeout=1)
                
                if response.status == health_pb2.HealthCheckResponse.SERVING:
                    print(f"✅ gRPC service '{service_name}' is ready.")
                    return True
        except grpc.RpcError as e:
            # Server not yet available or not serving the requested service
            time.sleep(1)
        except Exception as e:
            print(f"An unexpected error occurred while checking gRPC health: {e}")
            time.sleep(1)
            
    print(f"❌ Timeout: gRPC service '{service_name}' was not ready within {timeout} seconds.")
    return False

def main():
    """
    Starts the backend AI server and the frontend Gradio web app.
    Ensures the backend is ready before starting the frontend.
    """
    backend_process = None
    frontend_process = None
    
    python_executable = sys.executable
    backend_command = [python_executable, "-m", "foundations.ai_server"]
    frontend_command = [python_executable, "-m", "foundations.mcp_enhanced_app"]

    try:
        print("🚀 Starting backend AI server...")
        backend_process = subprocess.Popen(
            backend_command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # Wait for the gRPC server to be healthy before proceeding
        if not wait_for_grpc_server("foundations.CareerAssistantService", timeout=90):
            print("❌ Backend server did not become healthy. Terminating.")
            if backend_process.poll() is None:
                backend_process.terminate()
            return

        print("\n🚀 Starting frontend Gradio application...")
        frontend_process = subprocess.Popen(
            frontend_command,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print("✅ Frontend application process started.")
        
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