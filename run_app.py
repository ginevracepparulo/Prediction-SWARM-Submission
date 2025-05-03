import streamlit.web.cli 
import sys
import os

# Get the directory of the entry point script (the project root)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add both the frontend and backend directories to the Python path
frontend_dir = os.path.join(current_dir, "frontend")
backend_dir = os.path.join(current_dir, "backend")

if frontend_dir not in sys.path:
    sys.path.append(frontend_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Define the path to your main Streamlit app file within the frontend folder
streamlit_app_path = os.path.join(frontend_dir, "app.py") # Pointing specifically to frontend/app.py

if __name__ == "__main__":
    # Check if the app file exists (optional but good practice)
    if not os.path.exists(streamlit_app_path):
         print(f"Error: Streamlit app file not found at {streamlit_app_path}")
         sys.exit(1)

    # Simulate running 'streamlit run frontend/app.py'
    sys.argv = ["streamlit", "run", streamlit_app_path]
    sys.exit(streamlit.web.cli._main_run()) 