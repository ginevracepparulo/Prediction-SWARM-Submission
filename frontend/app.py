import trycourier
import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
import os 

auth_token = os.environ.get["courier_auth_token"]

def check_login():
    # Authentication
    __login__obj = __login__(auth_token = auth_token, 
                        company_name = "Shims",
                        width = 200, height = 250, 
                        logout_button_name = 'Logout', hide_menu_bool = False, 
                        hide_footer_bool = False, 
                        lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

    LOGGED_IN = __login__obj.build_login_ui()

    return LOGGED_IN, __login__obj

# --- Authentication ---
LOGGED_IN, auth_obj = check_login()

if LOGGED_IN:
    st.switch_page("pages/main.py")