import torch
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file
import numpy as np
import random

class EmotionPredictor:
    def __init__(self, model_path):
        """
        Initialize the emotion prediction model
        
        Args:
            model_path (str): Path to the SafeTensors model file
        """
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Define emotion labels (consistent with training)
        self.EMOTION_LABELS = {
            'Excitement': 0,
            'Satisfaction': 1,
            'Neutral': 2,
            'Confusion': 3,
            'Frustration': 4,
            'Disappointment': 5,
            'Optimism': 6,
            'Pessimism': 7
        }
        self.id_to_emotion = {v: k for k, v in self.EMOTION_LABELS.items()}
        
        # Load pre-trained BERT model for classification
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=len(self.EMOTION_LABELS)
        )
        
        # Load SafeTensors weights
        state_dict = load_file(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set model to evaluation mode
        
        # Add reasoning templates
        self.reasoning_templates = {
            'Excitement': [
                "The use of enthusiastic language like '{words}' indicates strong positive emotion",
                "Multiple exclamation marks and words like '{words}' show high enthusiasm",
                "Strong positive descriptors like '{words}' suggest excitement about the product"
            ],
            'Satisfaction': [
                "Positive but measured words like '{words}' indicate contentment",
                "The balanced tone and words like '{words}' suggest satisfaction",
                "The use of moderate positive terms like '{words}' shows satisfaction"
            ],
            'Neutral': [
                "The balanced language including '{words}' suggests a neutral stance",
                "The mix of positive and negative elements in '{words}' indicates neutrality",
                "The measured tone and phrases like '{words}' point to a neutral perspective"
            ],
            'Confusion': [
                "Uncertain language like '{words}' indicates confusion",
                "Questions and phrases like '{words}' suggest uncertainty",
                "The use of tentative words like '{words}' shows confusion about the product"
            ],
            'Frustration': [
                "Negative expressions like '{words}' indicate frustration",
                "Strong negative language like '{words}' shows frustration",
                "The presence of complaint-related words like '{words}' suggests frustration"
            ],
            'Disappointment': [
                "Words expressing letdown like '{words}' indicate disappointment",
                "The contrast between expectations and reality in phrases like '{words}' shows disappointment",
                "Expressions of unmet expectations like '{words}' suggest disappointment"
            ],
            'Optimism': [
                "Forward-looking positive words like '{words}' indicate optimism",
                "Hopeful language like '{words}' suggests optimism",
                "The positive future outlook in phrases like '{words}' shows optimism"
            ],
            'Pessimism': [
                "Negative outlook words like '{words}' indicate pessimism",
                "The presence of doubtful terms like '{words}' suggests pessimism",
                "Negative future expectations expressed in '{words}' show pessimism"
            ]
        }
    
    def get_important_words(self, review_text, attention_weights):
        """Extract words that received high attention scores"""
        tokens = self.tokenizer.tokenize(review_text)
        
        if attention_weights is None or attention_weights.numel() == 0:
            return []
            
        try:
            if attention_weights.dim() > 2:
                token_importance = attention_weights.mean(dim=(0, 1))
            else:
                token_importance = attention_weights
                
            if token_importance.dim() == 1 and len(token_importance) == len(tokens):
                top_indices = token_importance.argsort(descending=True)[:3]
                important_words = [tokens[i] for i in top_indices if i < len(tokens)]
                return important_words
            else:
                return []
        except Exception as e:
            return []
            
    def generate_reasoning(self, emotion, important_words, confidence, review_text):
        """Generate a more detailed, context-aware reasoning explanation"""
        def analyze_sentiment_nuances(text):
            """Extract key sentiment indicators"""
            positive_indicators = [
                'good', 'great', 'love', 'best', 'beautiful', 'fantastic',
                'amazing', 'wonderful', 'excellent', 'nice', 'pleasant'
            ]
            negative_indicators = [
                'bad', 'worst', 'terrible', 'horrible', 'disappointing',
                'poor', 'not good', 'issue', 'problem', 'fail', 'doesn\'t'
            ]
            
            pos_count = sum(1 for word in positive_indicators if word in text.lower())
            neg_count = sum(1 for word in negative_indicators if word in text.lower())
            
            return pos_count, neg_count
            
        # Generate nuanced reasoning based on emotion and text analysis
        pos_count, neg_count = analyze_sentiment_nuances(review_text)
        
        # Base template for more detailed reasoning
        reasoning_templates = {
            'Neutral': [
                "The review contains a mix of positive and negative elements, resulting in a balanced, neutral tone. "
                "The reviewer mentions both favorable and unfavorable aspects without strong emotional emphasis.",
                
                "While the review includes both positive and negative observations, the overall sentiment remains even-keeled. "
                "The language suggests a measured, objective assessment rather than a strongly emotional response."
            ],
            'Satisfaction': [
                "The review primarily focuses on positive aspects, with a measured and content tone. "
                "The reviewer appears generally pleased, expressing moderate appreciation for the product.",
                
                "The language suggests a comfortable level of contentment. "
                "While not overly enthusiastic, the review indicates a positive overall impression."
            ],
            'Frustration': [
                "The review highlights significant drawbacks that seem to provoke a sense of frustration. "
                "The language indicates disappointment with specific product characteristics.",
                
                "Multiple negative observations suggest the reviewer is experiencing clear dissatisfaction. "
                "The tone implies a level of irritation with the product's performance."
            ],
            'Disappointment': [
                "The review reveals unmet expectations, with the tone suggesting a sense of letdown. "
                "The contrast between hoped-for and actual product performance is evident.",
                
                "The reviewer's language indicates a gap between expectations and reality. "
                "The subtle undertones of disappointment emerge through detailed critical observations."
            ]
        }
        
        # Select reasoning template
        templates = reasoning_templates.get(emotion, [
            f"The review suggests a {emotion.lower()} emotional response. "
            "The language and tone indicate nuanced sentiments beyond simple categorization."
        ])
        
        # Choose a template and add context
        template = random.choice(templates)
        
        # Add confidence qualifier
        if confidence > 0.8:
            confidence_statement = "The model is highly confident in this emotional interpretation."
        elif confidence > 0.6:
            confidence_statement = "The model shows moderate confidence in this emotional assessment."
        else:
            confidence_statement = "The model's interpretation has some uncertainty and should be considered cautiously."
            
        # Combine template with context
        full_reasoning = f"{template} {confidence_statement}"
        
        return full_reasoning
        
    def predict_emotion(self, review_text):
        """Predict emotion with reasoning"""
        encoding = self.tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                output_attentions=True
            )
            
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get attention weights and important words
        attention_weights = outputs.attentions[-1] if hasattr(outputs, 'attentions') and outputs.attentions else None
        important_words = self.get_important_words(review_text, attention_weights[0] if attention_weights is not None else None)
        
        # Generate reasoning
        emotion = self.id_to_emotion[predicted_class]
        reasoning = self.generate_reasoning(emotion, important_words, confidence, review_text)
        
        # Calculate entropy loss safely
        try:
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                entropy = float(outputs.loss)
            else:
                # Calculate cross-entropy loss manually
                criterion = torch.nn.CrossEntropyLoss()
                entropy = float(criterion(outputs.logits, torch.tensor([predicted_class])).item())
        except:
            entropy = 0.0
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': {
                emotion: float(prob) 
                for emotion, prob in zip(self.EMOTION_LABELS.keys(), probabilities[0])
            },
            'reasoning': reasoning,
            'important_words': important_words,
            'entropy': entropy
        }