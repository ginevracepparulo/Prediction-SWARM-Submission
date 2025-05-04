class ProgressManager:
    def __init__(self):
        self.current_callback = None
    
    def set_callback(self, callback):
        self.current_callback = callback
    
    def get_callback(self):
        return self.current_callback

# Create a singleton instance
progress_manager = ProgressManager()