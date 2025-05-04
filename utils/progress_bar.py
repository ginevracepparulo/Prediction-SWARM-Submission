# Create a modified progress_manager.py

import logging
import threading
import queue
from typing import Callable, Optional, Dict, Any

class ProgressManager:
    def __init__(self):
        self.current_callback = None
        self._progress_queue = queue.Queue()
        self._thread_local = threading.local()
        logging.info("ProgressManager initialized")
    
    def set_callback(self, callback):
        self.current_callback = callback
        logging.info(f"Callback set: {callback is not None}")
    
    def get_callback(self):
        logging.info(f"Getting callback: {self.current_callback is not None}")
        return self.current_callback
    
    def update_progress(self, progress_value, status_message):
        """
        Thread-safe method to queue progress updates rather than executing them directly
        """
        if self.current_callback:
            # Instead of calling the callback directly, add to queue
            self._progress_queue.put((progress_value, status_message))
            logging.info(f"Progress update queued: {progress_value}% - {status_message}")
            return True
        else:
            logging.warning("Attempted to update progress but no callback is set")
            return False
    
    def process_updates(self):
        """
        Process all queued updates in the main thread
        Call this method regularly from the main Streamlit thread
        """
        updates_processed = 0
        while not self._progress_queue.empty():
            try:
                progress_value, status_message = self._progress_queue.get_nowait()
                if self.current_callback:
                    self.current_callback(progress_value, status_message)
                    updates_processed += 1
            except queue.Empty:
                break
        
        if updates_processed > 0:
            logging.info(f"Processed {updates_processed} queued progress updates")
        
        return updates_processed

# Create a singleton instance
progress_manager = ProgressManager()