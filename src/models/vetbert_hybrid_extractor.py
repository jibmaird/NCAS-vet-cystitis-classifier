"""
Hybrid VetBERT Feature Extractor

Combines both VetBERT embeddings (768-dim) and syndrome probabilities (~50-dim)
to create enhanced feature vectors for binary classification tasks.
"""

import torch
import numpy as np
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

class VetBERTHybridFeatureExtractor:
    """
    Extracts both embedding features and syndrome probabilities from VetBERT
    to create comprehensive feature vectors for downstream classification.
    """
    
    def __init__(self, model_path='havocy28/VetBERTDx', device=None):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer (shared between both models)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load embedding model (without classification head)
        self.embedding_model = AutoModel.from_pretrained(model_path)
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        
        # Freeze embedding model parameters
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        
        # Load classification model (with syndrome head)
        self.syndrome_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.syndrome_model.to(self.device)
        self.syndrome_model.eval()
        
        # Freeze syndrome model parameters
        for param in self.syndrome_model.parameters():
            param.requires_grad = False
        
        # Get syndrome labels
        self.syndrome_labels = list(self.syndrome_model.config.id2label.values())
        self.num_syndromes = len(self.syndrome_labels)
        
        print(f"Initialized VetBERT Hybrid Feature Extractor:")
        print(f"  - Embedding dimensions: 768")
        print(f"  - Syndrome categories: {self.num_syndromes}")
        print(f"  - Total feature dimensions: {768 + self.num_syndromes}")
        print(f"  - Device: {self.device}")
    
    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract 768-dimensional CLS token embeddings.
        
        Args:
            texts: List of input texts
            
        Returns:
            numpy array of shape (n_texts, 768)
        """
        embeddings = []
        
        for text in tqdm(texts, desc="Extracting embeddings"):
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Extract CLS token (first token)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])
        
        return np.array(embeddings)
    
    def extract_syndrome_probabilities(self, texts: List[str]) -> np.ndarray:
        """
        Extract syndrome probability distributions.
        
        Args:
            texts: List of input texts
            
        Returns:
            numpy array of shape (n_texts, num_syndromes)
        """
        syndrome_probs = []
        
        for text in tqdm(texts, desc="Extracting syndrome probabilities"):
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get syndrome probabilities
            with torch.no_grad():
                outputs = self.syndrome_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                syndrome_probs.append(probabilities[0])
        
        return np.array(syndrome_probs)
    
    def extract_hybrid_features(self, texts: List[str], 
                              include_embeddings: bool = True,
                              include_syndromes: bool = True,
                              syndrome_weight: float = 1.0) -> np.ndarray:
        """
        Extract hybrid features combining embeddings and syndrome probabilities.
        
        Args:
            texts: List of input texts
            include_embeddings: Whether to include 768-dim embeddings
            include_syndromes: Whether to include syndrome probabilities
            syndrome_weight: Scaling factor for syndrome probabilities
            
        Returns:
            numpy array of shape (n_texts, feature_dims)
        """
        if not include_embeddings and not include_syndromes:
            raise ValueError("Must include at least one feature type")
        
        features_list = []
        
        if include_embeddings:
            print("Extracting VetBERT embeddings...")
            embeddings = self.extract_embeddings(texts)
            features_list.append(embeddings)
            print(f"Embeddings shape: {embeddings.shape}")
        
        if include_syndromes:
            print("Extracting syndrome probabilities...")
            syndrome_probs = self.extract_syndrome_probabilities(texts)
            # Apply scaling if specified
            if syndrome_weight != 1.0:
                syndrome_probs = syndrome_probs * syndrome_weight
            features_list.append(syndrome_probs)
            print(f"Syndrome probabilities shape: {syndrome_probs.shape}")
        
        # Concatenate features
        hybrid_features = np.concatenate(features_list, axis=1)
        print(f"Final hybrid features shape: {hybrid_features.shape}")
        
        return hybrid_features
    
    def get_feature_names(self, include_embeddings: bool = True,
                         include_syndromes: bool = True) -> List[str]:
        """
        Get feature names for the hybrid feature vector.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        if include_embeddings:
            # Embedding feature names
            feature_names.extend([f"embedding_{i}" for i in range(768)])
        
        if include_syndromes:
            # Syndrome feature names
            feature_names.extend([f"syndrome_{label}" for label in self.syndrome_labels])
        
        return feature_names
    
    def extract_features_to_dataframe(self, texts: List[str],
                                    include_embeddings: bool = True,
                                    include_syndromes: bool = True,
                                    syndrome_weight: float = 1.0) -> pd.DataFrame:
        """
        Extract hybrid features and return as DataFrame with named columns.
        
        Returns:
            pandas DataFrame with feature columns
        """
        features = self.extract_hybrid_features(
            texts, include_embeddings, include_syndromes, syndrome_weight
        )
        feature_names = self.get_feature_names(include_embeddings, include_syndromes)
        
        return pd.DataFrame(features, columns=feature_names)
    
    def get_top_syndromes(self, texts: List[str], top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Get top-k syndrome predictions for each text.
        
        Args:
            texts: List of input texts
            top_k: Number of top syndromes to return
            
        Returns:
            List of lists containing (syndrome_name, probability) tuples
        """
        syndrome_probs = self.extract_syndrome_probabilities(texts)
        
        results = []
        for i, probs in enumerate(syndrome_probs):
            # Get top-k indices
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_syndromes = [
                (self.syndrome_labels[idx], probs[idx])
                for idx in top_indices
            ]
            results.append(top_syndromes)
        
        return results


def demonstrate_hybrid_features():
    """
    Demonstrate the hybrid feature extraction with example texts.
    """
    # Example veterinary consultation texts
    example_texts = [
        "Dog presented with frequent urination and blood in urine. Possible cystitis.",
        "Cat has been scratching excessively. Red inflamed skin on neck and behind ears.",
        "Patient shows signs of diarrhea and vomiting. Started yesterday after eating new food."
    ]
    
    print("=== VetBERT Hybrid Feature Extraction Demo ===\n")
    
    # Initialize extractor
    extractor = VetBERTHybridFeatureExtractor()
    
    print("\n1. Extracting hybrid features (embeddings + syndromes)...")
    hybrid_features = extractor.extract_hybrid_features(example_texts)
    print(f"Hybrid features shape: {hybrid_features.shape}")
    
    print("\n2. Extracting only embeddings...")
    embedding_features = extractor.extract_hybrid_features(
        example_texts, include_embeddings=True, include_syndromes=False
    )
    print(f"Embedding-only features shape: {embedding_features.shape}")
    
    print("\n3. Extracting only syndrome probabilities...")
    syndrome_features = extractor.extract_hybrid_features(
        example_texts, include_embeddings=False, include_syndromes=True
    )
    print(f"Syndrome-only features shape: {syndrome_features.shape}")
    
    print("\n4. Top syndrome predictions:")
    top_syndromes = extractor.get_top_syndromes(example_texts, top_k=3)
    for i, (text, syndromes) in enumerate(zip(example_texts, top_syndromes)):
        print(f"\nText {i+1}: {text[:50]}...")
        for syndrome, prob in syndromes:
            print(f"  {syndrome}: {prob:.4f}")
    
    print("\n5. Creating DataFrame with named features...")
    df = extractor.extract_features_to_dataframe(example_texts)
    print(f"DataFrame shape: {df.shape}")
    print(f"Feature columns: {df.columns.tolist()[:10]}...")  # Show first 10
    
    return extractor, hybrid_features


if __name__ == "__main__":
    # Run demonstration
    extractor, features = demonstrate_hybrid_features()
    
    print("\n=== Example Usage for Binary Classification ===")
    print("""
    # Usage example:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Extract hybrid features
    extractor = VetBERTHybridFeatureExtractor()
    X = extractor.extract_hybrid_features(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
    
    # Train classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    accuracy = classifier.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Feature importance analysis
    feature_names = extractor.get_feature_names()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    """)