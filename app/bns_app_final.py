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

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
        background-color: #f5f5f5;
    }
    .stTextArea>div>div>textarea {
        min-height: 120px;
        border-radius: 8px;
        padding: 1rem;
        font-size: 16px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        background: #1a73e8;
        color: white;
        border: none;
        font-weight: 500;
        font-size: 16px;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background: #1557b0;
    }
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
                details_df = pd.read_csv(details_path)
                # Convert to dictionary with section number as string key
                self.section_details = {}
                for _, row in details_df.iterrows():
                    # Use iloc[0] for position-based indexing to avoid deprecation warning
                    section_id = str(int(row.iloc[0]))  # Ensure section ID is string without .0
                    self.section_details[section_id] = {
                        'title': row.get('title', 'Section not found'),
                        'description': row.get('description', 'No description available'),
                        'punishment': row.get('punishment', 'No punishment information'),
                        'example_incidents': row.get('example_incidents', '')
                    }
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_section(self, text):
        """Predict the BNS section for the given text with enhanced matching"""
        if not text.strip():
            return None, 0.0
        
        try:
            # Preprocess text
            text_processed = text.lower().strip()
            
            # Check for exact matches in example incidents first
            best_match = self._find_best_match(text_processed)
            if best_match:
                return best_match, 0.95  # High confidence for exact matches
            
            # If no exact match, use the model
            X = self.vectorizer.transform([text_processed])
            predicted_class = self.model.predict(X)[0]
            
            # Get confidence score
            if hasattr(self.model, 'predict_proba'):
                confidence = np.max(self.model.predict_proba(X))
            else:
                decision_values = self.model.decision_function(X)
                confidence = 1 / (1 + np.exp(-np.max(decision_values)))
            
            # Boost confidence for higher section numbers (more specific sections)
            try:
                section_num = int(predicted_class.split()[0])
                confidence = min(confidence * (1 + section_num * 0.01), 0.99)
            except (ValueError, AttributeError):
                pass
                
            return predicted_class, confidence
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, 0.0
    
    def _find_best_match(self, text):
        """Find the best matching section based on example incidents"""
        text = text.lower()
        best_score = 0
        best_section = None
        
        for section_id, details in self.section_details.items():
            # Check example incidents
            examples = details.get('example_incidents', '')
            if not examples:
                continue
                
            # Check if any example is a substring of the input
            for example in examples.split(';'):
                example = example.strip().lower()
                if not example:
                    continue
                    
                # Simple substring matching for now
                if example in text:
                    # Longer matches are better
                    if len(example) > best_score:
                        best_score = len(example)
                        best_section = str(section_id)
                        
        return best_section
    
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
        if not section_id:
            st.warning("No section ID provided")
            return
            
        # Clean and extract section number
        if ' - ' in section_id:
            section_id = section_id.split(' - ')[0]
        section_id = str(section_id).strip()
        
        # Get section details with proper error handling
        try:
            # Try exact match first
            details = self.section_details.get(section_id, {})
            
            # If not found, try to find a matching section (handles cases where ID might be an integer)
            if not details and section_id.isdigit():
                # Try with integer key
                details = self.section_details.get(str(int(section_id)), {})
            
            # If still not found, check all sections for a match
            if not details:
                for sec_id, sec_details in self.section_details.items():
                    if str(sec_id).startswith(section_id) or section_id in str(sec_id):
                        details = sec_details
                        section_id = sec_id  # Use the matched section ID
                        break
            
            # If we still don't have details, show an error
            if not details:
                st.error(f"Could not find details for BNS Section {section_id}")
                st.json(self.section_details)  # Debug: Show available sections
                return
            
            # Get all fields with defaults
            title = details.get('title', f'BNS Section {section_id}')
            description = details.get('description', 'No description available')
            punishment = details.get('punishment', 'No punishment information')
            examples = details.get('example_incidents', '')
            
            # Create a clean, professional card layout
            st.markdown(f"""
                <div style="
                    background: #ffffff;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 15px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    border-left: 5px solid #1a73e8;
                    font-family: Arial, sans-serif;
                ">
                    <h3 style="
                        color: #1a73e8; 
                        margin: 0 0 15px 0;
                        padding-bottom: 10px;
                        border-bottom: 1px solid #e0e0e0;
                    ">
                        BNS Section {section_id}: {title}
                    </h3>
                    
                    <div style="
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 6px;
                        margin: 15px 0;
                        font-size: 15px;
                        line-height: 1.6;
                        color: #333;
                    ">
                        <div style="font-weight: 600; color: #1a73e8; margin-bottom: 8px;">📝 Description</div>
                        {description}
                    </div>
                    
                    <div style="
                        background: #fff8e1;
                        padding: 15px;
                        border-radius: 6px;
                        margin: 15px 0;
                        border-left: 4px solid #ffc107;
                        font-size: 15px;
                        line-height: 1.6;
                    ">
                        <div style="font-weight: 600; color: #e65100; margin-bottom: 8px;">⚖️ Punishment</div>
                        {punishment}
                    </div>
                    
                    {self._format_examples(examples) if examples else ''}
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error displaying section {section_id}: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
    
    def _format_examples(self, examples_str):
        """Format example incidents as HTML"""
        if not examples_str or not isinstance(examples_str, str):
            return ""
        
        examples = [ex.strip() for ex in str(examples_str).split(';') if ex.strip()]
        if not examples:
            return ""
        
        # Limit to 3 examples and format with better styling
        examples = examples[:3]
        
        examples_html = """
        <div style="
            margin: 20px 0 0 0;
            font-size: 15px;
        ">
            <div style="
                font-weight: 600; 
                color: #2e7d32; 
                margin-bottom: 10px;
                display: flex;
                align-items: center;
            ">
                <span style="margin-right: 8px;">📌</span> Example Incidents
            </div>
            <div style="
                background: #f5f5f5;
                border-radius: 6px;
                padding: 12px 15px;
                border-left: 3px solid #4caf50;
            ">
                <ul style="
                    margin: 0;
                    padding-left: 20px;
                    list-style-type: none;
                ">
        """
        
        for ex in examples:
            examples_html += f"""
                <li style="
                    margin: 0 0 10px 0;
                    padding: 0 0 10px 0;
                    border-bottom: 1px dashed #e0e0e0;
                    position: relative;
                    padding-left: 20px;
                ">
                    <span style="
                        position: absolute;
                        left: 0;
                        color: #4caf50;
                        font-weight: bold;
                    ">•</span>
                    {ex}
                </li>
            """
            
        examples_html += """
                </ul>
            </div>
        </div>
        """
        return examples_html
    
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
