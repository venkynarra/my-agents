import asyncio
import yaml
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from career_ai_assistant.ui.interface import create_interface
from career_ai_assistant.core.cache_manager import cache_manager
from career_ai_assistant.core.email_utils import email_manager
from career_ai_assistant.monitoring.logging import performance_monitor, logger
from career_ai_assistant.models.gemma_lora_inference import gemma_lora

class CareerAIAssistant:
    def __init__(self):
        self.config = None
        self.interface = None
        
    async def load_config(self):
        """Load configuration from YAML file"""
        try:
            config_path = Path(__file__).parent / "config" / "settings.yaml"
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Replace environment variable placeholders
            self._replace_env_vars(self.config)
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def _replace_env_vars(self, config_dict):
        """Recursively replace environment variable placeholders in config"""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._replace_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]  # Remove ${ and }
                config_dict[key] = os.getenv(env_var, value)
            
    async def initialize_components(self):
        """Initialize all architecture components"""
        try:
            # Initialize cache manager
            await cache_manager.connect()
            logger.info("Cache manager initialized")
            
            # Initialize email manager
            if email_manager.sendgrid_api_key:
                logger.info("Email manager initialized with SendGrid")
            else:
                logger.warning("Email manager initialized without SendGrid API key")
            
            # Initialize ONNX model
            await gemma_lora.load_model()
            logger.info("ONNX model initialized")
            
            # Create UI interface
            self.interface = create_interface()
            logger.info("UI interface created")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
            
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            await cache_manager.close()
            logger.info("Cache manager closed")
            logger.info("Shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
    async def run(self):
        """Main run loop"""
        try:
            logger.info("Starting Career AI Assistant...")
            
            # Load configuration
            await self.load_config()
            
            # Initialize components
            await self.initialize_components()
            
            logger.info("All components initialized successfully")
            logger.info("Career AI Assistant is ready!")
            logger.info("Email functionality: " + ("Enabled" if email_manager.sendgrid_api_key else "Disabled"))
            
            # Try to launch the interface with port fallback
            port = self.config['ui']['port']
            max_port_attempts = 5
            
            for attempt in range(max_port_attempts):
                try:
                    logger.info(f"Attempting to launch on port {port}")
                    self.interface.launch(
                        server_name=self.config['ui']['host'],
                        server_port=port,
                        share=self.config['ui']['share'],
                        debug=self.config['ui']['debug']
                    )
                    break  # Success, exit the loop
                except Exception as e:
                    if "port" in str(e).lower() or "bind" in str(e).lower():
                        port += 1
                        logger.warning(f"Port {port-1} in use, trying port {port}")
                        if attempt == max_port_attempts - 1:
                            logger.error(f"Could not find available port after {max_port_attempts} attempts")
                            raise
                    else:
                        # Not a port issue, re-raise
                        raise
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.shutdown()

async def main():
    """Main entry point"""
    assistant = CareerAIAssistant()
    await assistant.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 