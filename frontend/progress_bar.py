import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressManager:
    """
    Singleton class to manage progress updates between frontend and backend.
    This ensures the same instance is used throughout the application.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating new ProgressManager instance")
            cls._instance = super(ProgressManager, cls).__new__(cls)
            cls._instance.current_callback = None
        return cls._instance
    
    def set_callback(self, callback):
        """Set the callback function to be called on progress updates"""
        logger.info(f"Setting callback: {callback}")
        self.current_callback = callback
    
    def get_callback(self):
        """Get the current callback function"""
        logger.info(f"Getting callback: {self.current_callback}")
        return self.current_callback
    
    def update_progress(self, value, message):
        """Update the progress with a value and message"""
        logger.info(f"Updating progress: {value}% - {message}")
        if self.current_callback:
            try:
                self.current_callback(value, message)
                logger.info("Progress callback executed successfully")
            except Exception as e:
                logger.error(f"Error in progress callback: {e}", exc_info=True)
        else:
            logger.warning("No callback set, progress update ignored")

# Create the singleton instance to export
progress_manager = ProgressManager()
logger.info(f"progress_manager initialized: {progress_manager}")