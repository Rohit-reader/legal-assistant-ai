import streamlit as st
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class BNSApp:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.section_details = {}
        self.recognizer = sr.Recognizer()
        self.load_models()
    
    def load_models(self):
        """Load the trained model and vectorizer"""
        try:
            model_dir = Path('saved_models')
            model_path = model_dir / 'bns_classifier.joblib'
            vectorizer_path = model_dir / 'bns_vectorizer.joblib'
            details_path = model_dir / 'bns_section_details.csv'
            
            if not all(p.exists() for p in [model_path, vectorizer_path, details_path]):
                st.error("Model files not found. Please train the model first.")
                return False
            
            with st.spinner('Loading BNS classifier...'):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.section_details = pd.read_csv(details_path, index_col=0).to_dict('index')
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_text(self, text):
        """Preprocess the input text"""
        if not text or not isinstance(text, str):
            return ""
        return ' '.join(text.lower().strip().split())
    
    def predict_section(self, text):
        """Predict the BNS section for the given text"""
        if not text.strip():
            return None
        
        try:
            processed_text = self.preprocess_text(text)
            X = self.vectorizer.transform([processed_text])
            predicted_class = self.model.predict(X)[0]
            return predicted_class
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
    
    def record_voice(self):
        """Record voice input and convert to text"""
        with sr.Microphone() as source:
            st.info("Listening... Speak now")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio)
                return True, text
            except Exception as e:
                return False, str(e)
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        try:
            tts = gTTS(text=text, lang='en')
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
            return None
    
    def display_section(self, section_id):
        """Display section details in an organized card"""
        if ' - ' in section_id:
            section_id = section_id.split(' - ')[0]
        
        details = self.section_details.get(section_id, {
            'title': 'Section not found',
            'description': 'No description available',
            'punishment': 'No punishment information',
            'example_incidents': ''
        })
        
        # Create a card-like container
        st.markdown(
            f"""
            <div style="
                background: #ffffff;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1.5rem;
                border-left: 5px solid #4a6fa5;
            ">
                <h3 style="color: #2c3e50; margin-top: 0;">BNS Section {section_id}</h3>
                <h4 style="color: #4a6fa5; margin-bottom: 1rem;">{details['title']}</h4>
                
                <div style="margin-bottom: 1rem;">
                    <p><strong>Description:</strong> {details['description']}</p>
                </div>
                
                <div style="
                    background: #f8f9fa;
                    padding: 0.75rem;
                    border-radius: 6px;
                    margin: 1rem 0;
                ">
                    <strong>Punishment:</strong> {details['punishment']}
                </div>
                
                {self._format_examples(details.get('example_incidents', ''))}
                
                {self._add_audio_button(section_id, details)}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _format_examples(self, examples_str):
        """Format example incidents as HTML"""
        if not examples_str or not isinstance(examples_str, str):
            return ""
        
        examples = [ex.strip() for ex in examples_str.split(';') if ex.strip()]
        if not examples:
            return ""
        
        examples_html = "<div style='margin: 1rem 0;'><strong>Example Incidents:</strong><ul style='margin-top: 0.5rem;'>"
        for ex in examples:
            examples_html += f"<li style='margin-bottom: 0.5rem;'>{ex}</li>"
        examples_html += "</ul></div>"
        
        return examples_html
    
    def _add_audio_button(self, section_id, details):
        """Add audio button with text-to-speech functionality"""
        if st.button(f"🔊 Listen to Section {section_id}", key=f"audio_{section_id}"):
            text_to_speak = f"""
            BNS Section {section_id}.
            {details['title']}.
            {details['description']}
            Punishment: {details['punishment']}
            """
            audio = self.text_to_speech(text_to_speak)
            if audio:
                audio_base64 = base64.b64encode(audio.read()).decode('utf-8')
                return f"""
                <audio autoplay="true" controls style="width: 100%; margin-top: 1rem;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                """
        return ""
    
    def run(self):
        """Run the Streamlit app"""
        st.set_page_config(
            page_title="BNS Section Finder",
            page_icon="⚖️",
            layout="wide"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
            .main-header {color: #2c3e50;}
            .sidebar .sidebar-content {background-color: #f8f9fa;}
            .stButton>button {width: 100%;}
            .stTextArea>div>div>textarea {min-height: 150px;}
            .stMarkdown h3 {color: #2c3e50;}
            .stMarkdown h4 {color: #4a6fa5;}
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("⚖️ BNS Section Finder")
        page = st.sidebar.radio("Navigation", ["Home", "Find Section", "Browse Sections"])
        
        # Home Page
        if page == "Home":
            st.title("Welcome to BNS Section Finder")
            st.markdown("""
            This application helps you find relevant sections of the Bharatiya Nyaya Sanhita (BNS) 
            based on incident descriptions.
            
            ### Features:
            - 🔍 Find BNS sections by describing an incident
            - 🎙️ Voice input support
            - 📚 Browse all BNS sections
            - 🔊 Listen to section details with text-to-speech
            
            ### How to use:
            1. Go to **Find Section** and describe your incident
            2. Use voice input if preferred
            3. Browse through suggested BNS sections
            4. Listen to section details with audio playback
            """)
        
        # Find Section Page
        elif page == "Find Section":
            st.title("🔍 Find BNS Section")
            
            # Input method selection
            input_method = st.radio("Choose input method:", ["Text", "Voice"])
            
            if input_method == "Text":
                text = st.text_area("Describe the incident:", 
                                  placeholder="E.g., A person intentionally killed another person during a robbery...",
                                  height=150)
                if st.button("Find Section"):
                    if text.strip():
                        with st.spinner("Analyzing incident and finding relevant BNS section..."):
                            section = self.predict_section(text)
                            if section:
                                st.success(f"Found relevant BNS section:")
                                self.display_section(section)
                            else:
                                st.warning("Could not determine a relevant BNS section. Please try with a more detailed description.")
                    else:
                        st.warning("Please enter a description of the incident.")
            
            else:  # Voice input
                if st.button("🎤 Start Recording"):
                    success, result = self.record_voice()
                    if success:
                        st.text_area("Recognized Text:", value=result, height=100)
                        with st.spinner("Analyzing incident and finding relevant BNS section..."):
                            section = self.predict_section(result)
                            if section:
                                st.success(f"Found relevant BNS section:")
                                self.display_section(section)
                            else:
                                st.warning("Could not determine a relevant BNS section. Please try with a more detailed description.")
                    else:
                        st.error(f"Error: {result}")
        
        # Browse Sections Page
        elif page == "Browse Sections":
            st.title("📚 Browse BNS Sections")
            
            # Search functionality
            search_query = st.text_input("Search sections by keyword:", "")
            
            # Filter sections based on search query
            filtered_sections = []
            for section_id, details in self.section_details.items():
                search_text = f"{section_id} {details.get('title', '')} {details.get('description', '')} {details.get('punishment', '')}".lower()
                if not search_query or search_query.lower() in search_text:
                    filtered_sections.append((section_id, details.get('title', '')))
            
            # Sort sections numerically
            filtered_sections.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
            
            if not filtered_sections:
                st.info("No sections match your search criteria.")
            else:
                for section_id, title in filtered_sections:
                    self.display_section(f"{section_id} - {title}")

if __name__ == "__main__":
    app = BNSApp()
    if app.model is not None and app.vectorizer is not None:
        app.run()
    else:
        st.error("Failed to load the BNS classifier. Please check if the model files are available.")
