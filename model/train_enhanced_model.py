import os
import pandas as pd
import numpy as np
import joblib
import spacy
import re
import random
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class BNSTextClassifier:
    def __init__(self, model_dir='saved_models'):
        """Initialize the BNS text classifier with enhanced settings"""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLP pipeline with only necessary components for better performance
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
        
        # Enhanced vectorizer with improved parameters
        self.vectorizer = TfidfVectorizer(
            max_features=15000,  # Slightly reduced to prevent overfitting
            ngram_range=(1, 2),  # Focus on unigrams and bigrams only
            stop_words='english',
            min_df=3,  # Increased to filter out rare terms
            max_df=0.85,  # Slightly lower to filter out very common terms
            sublinear_tf=True,
            analyzer='word',
            norm='l2',
            use_idf=True,
            smooth_idf=True
        )
        
        # Enhanced classifier with optimized parameters
        self.classifier = LinearSVC(
            class_weight='balanced',
            random_state=42,
            max_iter=10000,  # Increased for better convergence
            C=0.75,  # Slightly higher C for less regularization
            dual=False,
            loss='squared_hinge',
            penalty='l2',
            tol=1e-4
        )
        
        self.calibrated_classifier = None
        self.section_details = {}
        self.classes_ = None
    
    def preprocess_text(self, text):
        """Preprocess the input text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        
        # Use spaCy for advanced preprocessing
        doc = self.nlp(text)
        
        # Lemmatize and filter tokens
        tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                lemma = token.lemma_.lower().strip()
                if lemma and len(lemma) > 2:  # Remove very short tokens
                    tokens.append(lemma)
        
        return ' '.join(tokens)
    
    def augment_text(self, text, n_augment=2):
        """Generate augmented versions of the text"""
        if not text or not isinstance(text, str):
            return []
            
        augmented_texts = []
        words = text.split()
        
        # Only augment if we have enough words
        if len(words) < 4:
            return [text]
        
        for _ in range(n_augment):
            # Randomly shuffle words (but keep the meaning somewhat intact)
            if random.random() > 0.5 and len(words) > 3:
                # Keep first and last words in place, shuffle the middle
                first_word = words[0]
                last_word = words[-1]
                middle = words[1:-1]
                random.shuffle(middle)
                new_text = ' '.join([first_word] + middle + [last_word])
                augmented_texts.append(new_text)
            
            # Randomly drop some words (but not too many)
            if random.random() > 0.7 and len(words) > 5:
                n_drop = min(2, len(words) // 4)
                drop_indices = sorted(random.sample(range(len(words)), len(words) - n_drop))
                new_text = ' '.join([words[i] for i in drop_indices])
                augmented_texts.append(new_text)
        
        return list(set(augmented_texts))  # Remove duplicates
    
    def load_data(self, data_path):
        """Load and preprocess the expanded BNS dataset with enhanced data augmentation"""
        # Load the dataset with explicit encoding
        df = pd.read_csv(data_path, encoding='utf-8')
        
        # Handle missing values more robustly
        required_columns = ['section', 'title', 'description', 'punishment', 'example_incidents', 'category', 'keywords']
        df = df.dropna(subset=required_columns[:4])  # Only require first 4 columns
        
        # Store section details for later use with all available columns
        self.section_details = df.set_index('section').to_dict('index')
        
        texts = []
        labels = []
        
        print("Processing sections and augmenting data...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sections"):
            section_id = row['section']
            section_title = row['title']
            label = f"{section_id} - {section_title}"
            
            # Create a comprehensive text representation for each section
            section_texts = []
            
            # Add the main description
            if pd.notna(row['description']):
                section_texts.append(f"Description: {row['description']}")
            
            # Add punishment information
            if pd.notna(row['punishment']):
                section_texts.append(f"Punishment: {row['punishment']}")
            
            # Add examples
            if pd.notna(row['example_incidents']):
                examples = [ex.strip() for ex in str(row['example_incidents']).split(';') if ex.strip()]
                section_texts.extend([f"Example: {ex}" for ex in examples])
            
            # Add category and keywords as context
            if pd.notna(row.get('category', '')):
                section_texts.append(f"Category: {row['category']}")
            
            if pd.notna(row.get('keywords', '')):
                section_texts.append(f"Keywords: {row['keywords']}")
            
            # Combine all text for this section
            combined_text = ' '.join(section_texts)
            
            # Process and add the main text
            processed = self.preprocess_text(combined_text)
            if processed:
                texts.append(processed)
                labels.append(label)
            
            # Generate augmented versions with more variety
            augmented_texts = self.augment_text(combined_text, n_augment=3)
            for aug_text in augmented_texts:
                processed_aug = self.preprocess_text(aug_text)
                if processed_aug and processed_aug != processed:
                    texts.append(processed_aug)
                    labels.append(label)
        
        return texts, labels, df
    
    def train(self, X, y, test_size=0.2):
        """Train the classifier with cross-validation and calibration"""
        # Initialize classes_ from the training data
        self.classes_ = np.unique(y)
        
        # Calculate class distribution
        class_counts = pd.Series(y).value_counts()
        min_samples_per_class = min(class_counts) if len(class_counts) > 0 else 0
        
        # For very small datasets, use a simple train-test split without stratification
        use_stratify = min_samples_per_class >= 2 and len(class_counts) < 20
        
        # For extremely small datasets, use all data for training
        if len(X) < 20:
            print("Very small dataset detected. Using all data for training...")
            X_train, X_test, y_train, y_test = X, [], y, []
        else:
            # Adjust test_size if needed
            if min_samples_per_class > 0 and test_size * min_samples_per_class < 1:
                test_size = 0.5 / min_samples_per_class
                test_size = min(0.3, test_size)  # Cap at 30% test size
                print(f"Adjusted test_size to {test_size:.2f} to ensure sufficient samples per class")
            
            # Split data into train and test sets
            split_params = {
                'test_size': test_size,
                'random_state': 42,
                'shuffle': True
            }
            
            if use_stratify:
                split_params['stratify'] = y
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test) if len(X_test) > 0 else 'N/A (using all data for training)'}")
        
        # Vectorize the training data
        print("Fitting vectorizer...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Calculate class weights
        print("Calculating class weights...")
        self.classes_ = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=self.classes_, y=y_train)
        class_weight_dict = dict(zip(self.classes_, class_weights))
        self.classifier.class_weight = class_weight_dict
        
        # Train the final model on the full training set
        print("\nTraining final model...")
        self.classifier.fit(X_train_vec, y_train)
        
        # For very small datasets, skip calibration to avoid errors
        min_samples_per_class = min(pd.Series(y_train).value_counts())
        
        # Only calibrate if we have enough samples and at least 2 classes with sufficient samples
        if (len(X_train) >= 10 and len(self.classes_) >= 2 and 
            min_samples_per_class >= 2 and len(X_train) > len(self.classes_)):
            try:
                print("Calibrating classifier...")
                # Use a smaller number of folds for small datasets
                n_folds = min(3, min_samples_per_class)
                if n_folds < 2:  # Need at least 2 folds for calibration
                    raise ValueError("Insufficient samples per class for calibration")
                    
                self.calibrated_classifier = CalibratedClassifierCV(
                    self.classifier,
                    cv=min(n_folds, len(X_train) // 2),  # Ensure we have enough samples per fold
                    method='sigmoid',
                    n_jobs=-1
                )
                self.calibrated_classifier.fit(X_train_vec, y_train)
                model_to_use = self.calibrated_classifier
                print("Calibration successful.")
            except Exception as e:
                print(f"Calibration failed: {str(e)}. Using uncalibrated model.")
                model_to_use = self.classifier
        else:
            print("Insufficient data for calibration, using uncalibrated model...")
            model_to_use = self.classifier
        
        # Evaluate on test set if we have one
        accuracy, f1 = 0, 0
        if len(X_test) > 0:
            print("\nEvaluating on test set...")
            X_test_vec = self.vectorizer.transform(X_test)
            y_pred = model_to_use.predict(X_test_vec)
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            print(f"\nAccuracy: {accuracy:.3f}")
            print(f"Weighted F1: {f1:.3f}")
        else:
            print("\nNo test samples available for evaluation.")
            
            # If no test set, do a simple cross-validation
            if len(X_train) >= 5 and len(self.classes_) >= 2:
                print("Performing cross-validation on training set...")
                cv = min(3, min(np.bincount([np.where(self.classes_ == cls)[0][0] for cls in y_train])))
                if cv >= 2:
                    cv_scores = cross_val_score(
                        self.classifier, X_train_vec, y_train,
                        cv=cv,
                        scoring='f1_weighted',
                        n_jobs=-1
                    )
                    print(f"CV F1 scores: {cv_scores}")
                    print(f"Mean CV F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
                    accuracy = np.mean(cv_scores)
                    f1 = np.mean(cv_scores)
        
        return X_test, y_test, accuracy, f1
        
        return X_test, y_test
    
    def save_model(self):
        """Save the model and related artifacts"""
        model_path = self.model_dir / 'bns_classifier.joblib'
        vectorizer_path = self.model_dir / 'bns_vectorizer.joblib'
        details_path = self.model_dir / 'bns_section_details.csv'
        
        # Ensure the directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the model components
        model_to_save = self.calibrated_classifier if self.calibrated_classifier is not None else self.classifier
        joblib.dump(model_to_save, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save section details as a DataFrame
        details_df = pd.DataFrame.from_dict(self.section_details, orient='index')
        details_df.to_csv(details_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        print(f"Section details saved to {details_path}")
        
        return model_path, vectorizer_path, details_path
    
    @classmethod
    def load_model(cls, model_dir='saved_models'):
        """Load a trained model"""
        model_path = Path(model_dir) / 'bns_classifier.joblib'
        vectorizer_path = Path(model_dir) / 'bns_vectorizer.joblib'
        details_path = Path(model_dir) / 'bns_section_details.csv'
        
        if not all(p.exists() for p in [model_path, vectorizer_path, details_path]):
            raise FileNotFoundError("One or more model files not found. Please train the model first.")
        
        # Create a new instance
        classifier = cls(model_dir)
        
        # Load the components
        classifier.calibrated_classifier = joblib.load(model_path)
        classifier.classifier = classifier.calibrated_classifier.base_estimator
        classifier.vectorizer = joblib.load(vectorizer_path)
        
        # Load section details
        details_df = pd.read_csv(details_path, index_col=0)
        classifier.section_details = details_df.to_dict('index')
        classifier.classes_ = np.array([f"{idx} - {val['title']}" for idx, val in classifier.section_details.items()])
        
        return classifier

def main():
    # Configuration
    DATA_PATH = os.path.join('data', 'raw', 'bns_extended_concise.csv')  # Updated to use the concise dataset
    MODEL_DIR = 'saved_models'
    
    # Create output directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("=" * 80)
    print("BNS Section Classifier Training")
    print("=" * 80)
    
    # Initialize and train the classifier
    classifier = BNSTextClassifier(MODEL_DIR)
    
    try:
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        X, y, section_details = classifier.load_data(DATA_PATH)
        
        # Print class distribution
        print("\nClass distribution in dataset:")
        class_counts = pd.Series(y).value_counts().sort_index()
        print(class_counts)
        
        # Check if we have enough samples
        min_samples = min(class_counts)
        if min_samples < 2:
            print(f"\nWARNING: Some classes have very few samples (minimum: {min_samples}). "
                  "Consider adding more training data for better performance.")
        
        # Train the model
        print("\nStarting model training...")
        X_test, y_test, accuracy, f1 = classifier.train(X, y)
        
        # Save the model
        print("\nSaving model and artifacts...")
        classifier.save_model()
        
        # Print final summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total training samples: {len(X)}")
        print(f"Number of classes: {len(class_counts)}")
        if len(X_test) > 0:
            print(f"Test set size: {len(X_test)}")
            print(f"Test accuracy: {accuracy:.3f}")
            print(f"Test weighted F1: {f1:.3f}")
        print("\nModel and artifacts saved to:")
        print(f"- Model: {os.path.join(MODEL_DIR, 'bns_classifier.joblib')}")
        print(f"- Vectorizer: {os.path.join(MODEL_DIR, 'bns_vectorizer.joblib')}")
        print(f"- Section details: {os.path.join(MODEL_DIR, 'bns_section_details.csv')}")
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTraining failed. Please check the error message above.")
        return False
    
    return True

if __name__ == "__main__":
    main()
