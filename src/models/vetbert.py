import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm

class VetBERT:
    def __init__(self, model_name='havocy28/VetBERTDx', device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.label_map = None
        self._load_model()

    def _load_model(self):
        """Initialize the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_safetensors=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, use_safetensors=True)
        self.model.to(self.device)
        self.label_map = self.model.config.id2label

    def get_embeddings(self, text):
        """Get sorted (probability, label) list for a single piece of text"""
        if pd.isnull(text):
            return None
        self.model.eval()
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            logits = self.model(**encoded_input).logits
            probabilities = torch.softmax(logits, dim=-1)
        sorted_probs = sorted(
            ((prob.item(), self.label_map[idx]) for idx, prob in enumerate(probabilities[0])),
            reverse=True,
            key=lambda x: x[0]
        )
        return sorted_probs

    def classify_texts(self, df, input_column="Examination Text", output_column=None):
        """Classify a DataFrame of texts"""
        tqdm.pandas()
        classifications = df[input_column].progress_apply(self.get_embeddings)
        predictions = [item[0][1] if item is not None else None for item in classifications]
        if output_column:
            return pd.DataFrame({
                input_column: df[input_column],
                "Classification": predictions,
                output_column: df[output_column]
            })
        return predictions

    def classify_texts_with_probs(self, df, input_column="Examination Text", output_column=None, long_format=False):
        """Return all label probabilities in long or wide format per text"""
        tqdm.pandas()
        results = []

        include_true = output_column and output_column in df.columns

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = row[input_column]
            true_label = row[output_column] if include_true else None
            probs = self.get_embeddings(text)
            if probs is not None:
                if long_format:
                    for prob, pred_label in probs:
                        entry = {
                            input_column: text,
                            "Label": pred_label,
                            "Probability": prob,
                        }
                        if include_true:
                            entry["True_Label"] = true_label
                        results.append(entry)
                else:
                    entry = {label: prob for prob, label in probs}
                    entry[input_column] = text
                    if include_true:
                        entry["True_Label"] = true_label
                    results.append(entry)

        return pd.DataFrame(results).reset_index(drop=True)
