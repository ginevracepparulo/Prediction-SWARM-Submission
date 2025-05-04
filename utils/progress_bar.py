# Add debugging to see when the callback is set and when it's being accessed

# In progress_manager.py
import logging

class ProgressManager:
    def __init__(self):
        self.current_callback = None
        logging.info("ProgressManager initialized")
    
    def set_callback(self, callback):
        self.current_callback = callback
        logging.info(f"Callback set: {callback is not None}")
    
    def get_callback(self):
        logging.info(f"Getting callback: {self.current_callback is not None}")
        return self.current_callback
    
    def update_progress(self, progress_value, status_message):
        if self.current_callback:
            self.current_callback(progress_value, status_message)
            return True
        else:
            logging.warning("Attempted to update progress but no callback is set")
            return False

# Create a singleton instance
progress_manager = ProgressManager()


# In your Agent.py file, update the progress using the manager's method
# Replace direct callback usage like this:
# if progress_manager.current_callback:
#     progress_manager.current_callback(100, "✅ Analysis complete!")

# With this:
# progress_manager.update_progress(100, "✅ Analysis complete!")