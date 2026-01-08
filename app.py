import os
import sys
import logging

# Configure logging for Hugging Face
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Debug: Print current directory and Python path
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"__file__ location: {os.path.abspath(__file__)}")
logger.info(f"Python path: {sys.path}")

# Setup Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)
logger.info(f"Added to Python path: {app_dir}")

# Debug: Check if src directory exists
src_dir = os.path.join(app_dir, 'src')
logger.info(f"src directory exists: {os.path.exists(src_dir)}")
if os.path.exists(src_dir):
    logger.info(f"src contents: {os.listdir(src_dir)}")
    models_dir = os.path.join(src_dir, 'models')
    logger.info(f"src/models exists: {os.path.exists(models_dir)}")
    if os.path.exists(models_dir):
        logger.info(f"src/models contents: {os.listdir(models_dir)}")

# Import after path setup
logger.info("Attempting to import AnalysisServer...")
from src.api.server import AnalysisServer
logger.info("Successfully imported AnalysisServer!")

def main():
    """Launch the Gradio interface for Hugging Face Spaces"""
    try:
        logger.info("Starting Financial Research Agent on Hugging Face Spaces...")

        # Create and launch server
        server = AnalysisServer()

        # Launch with Hugging Face Spaces settings
        interface = server.create_interface()

        # Hugging Face Spaces specific configuration
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            auth=None,
        )

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

if __name__ == "__main__":
    main()
