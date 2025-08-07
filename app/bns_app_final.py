import os
import sys
import streamlit as st
import speech_recognition as sr
import joblib
import numpy as np
from pathlib import Path
import json

# Set page config
st.set_page_config(
    page_title="BNS Section Finder",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {max-width: 900px; margin: 0 auto; padding: 2rem;}
    .stTextArea>div>div>textarea {min-height: 150px; border-radius: 8px; padding: 1rem;}
    .stButton>button {width: 100%; border-radius: 8px; padding: 0.75rem 1.5rem; background: #4CAF50; color: white; border: none;}
    .section-card {background: #fff; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #4CAF50;}
    .section-title {color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.5rem;}
    .section-punishment {color: #d32f2f; font-weight: 500; margin: 0.75rem 0;}
    .section-desc {color: #374151; line-height: 1.6; margin: 0.5rem 0;}
    </style>
""", unsafe_allow_html=True)

class BNSApp:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.section_details = {}
        self.recognizer = sr.Recognizer()
        self.load_models()
    
    def load_models(self):
        try:
            model_dir = Path('saved_models')
            model_path = model_dir / 'enhanced_bns_classifier.joblib'
            vectorizer_path = model_dir / 'enhanced_bns_vectorizer.joblib'
            details_path = model_dir / 'bns_section_details_enhanced.csv'
            
            if not all(p.exists() for p in [model_path, vectorizer_path, details_path]):
                st.error("Model files not found. Please ensure the model is trained first.")
                return False
            
            with st.spinner('Loading BNS classifier...'):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                # Load section details from CSV
                import pandas as pd
                details_df = pd.read_csv(details_path, index_col=0)
                self.section_details = details_df.to_dict('index')
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_section(self, text):
        """Predict the BNS section for the given text"""
        if not text.strip():
            return None, 0.0
        
        try:
            X = self.vectorizer.transform([text.lower().strip()])
            predicted_class = self.model.predict(X)[0]
            
            # Use decision_function for LinearSVC
            if hasattr(self.model, 'predict_proba'):
                confidence = np.max(self.model.predict_proba(X))
            else:
                # For LinearSVC, use decision function and normalize to [0,1]
                decision_values = self.model.decision_function(X)
                confidence = 1 / (1 + np.exp(-np.max(decision_values)))
                
            return predicted_class, confidence
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, 0.0
    
    def record_voice(self):
        with sr.Microphone() as source:
            try:
                with st.spinner("Listening... Speak now"):
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    return self.recognizer.recognize_google(audio)
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                st.warning("Could not understand audio. Please try again.")
                return ""
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return ""
    
    def display_section(self, section_id):
        if ' - ' in section_id:
            section_id = section_id.split(' - ')[0]
        
        details = self.section_details.get(section_id, {
            'title': 'Section not found',
            'description': 'No description available',
            'punishment': 'No punishment information',
            'example_incidents': ''
        })
        
        st.markdown(f"""
            <div class="section-card">
                <h3 class="section-title">BNS Section {section_id}</h3>
                <p class="section-desc"><strong>Title:</strong> {details['title']}</p>
                <div class="section-punishment">
                    <strong>Punishment:</strong> {details.get('punishment', 'Not specified')}
                </div>
                <div class="section-desc">
                    <strong>Description:</strong> {details.get('description', 'No description available')}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit app"""
        st.title("⚖️ BNS Section Finder")
        st.markdown("Enter the details of a legal incident to find the relevant BNS sections.")
        
        # Initialize session state for voice input
        if 'incident_desc' not in st.session_state:
            st.session_state.incident_desc = ""
        
        # Voice input button (outside form)
        if st.button("🎤 Click to Record Voice", help="Click and speak"):
            voice_text = self.record_voice()
            if voice_text:
                st.session_state.incident_desc = voice_text
        
        # Form for text input and submission
        with st.form("incident_form"):
            incident_desc = st.text_area(
                "Describe the incident in detail:",
                value=st.session_state.incident_desc,
                placeholder="Example: A person was caught stealing a mobile phone from a shop...",
                key="incident_text"
            )
            
            submitted = st.form_submit_button("Find BNS Section")
            
            if submitted:
                if incident_desc.strip():
                    with st.spinner("Analyzing the incident..."):
                        predicted_section, _ = self.predict_section(incident_desc)
                        if predicted_section:
                            st.success("Relevant BNS Section Found!")
                            self.display_section(predicted_section)
                        else:
                            st.warning("Could not determine the relevant BNS section.")
                else:
                    st.warning("Please enter a description of the incident.")
        
        st.markdown("---")
        st.markdown("""
            **Note:** This tool is for informational purposes only and does not constitute legal advice. 
            Always consult with a qualified legal professional for specific legal matters.
        """)

if __name__ == "__main__":
    app = BNSApp()
    if app.model is not None and app.vectorizer is not None:
        app.run()
    else:
        st.error("Failed to load the BNS classifier. Please check if the model files are available.")
