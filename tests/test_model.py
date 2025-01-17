import pytest
from model import EmotionPredictor
import os

def test_model_initialization():
    """Test if model loads correctly"""
    model_path = 'models/best_model.safetensors'
    assert os.path.exists(model_path), "Model file not found"
    model = EmotionPredictor(model_path)
    assert model is not None, "Model failed to initialize"

def test_prediction_structure():
    """Test prediction output structure"""
    model = EmotionPredictor('models/best_model.safetensors')
    test_text = "This product is amazing!"
    prediction = model.predict_emotion(test_text)
    
    required_fields = ['emotion', 'confidence', 'reasoning', 'entropy']
    for field in required_fields:
        assert field in prediction, f"Missing required field: {field}"

def test_emotion_categories():
    """Verify emotion predictions are within defined categories"""
    model = EmotionPredictor('models/best_model.safetensors')
    test_text = "This product is amazing!"
    prediction = model.predict_emotion(test_text)
    
    valid_emotions = {
        'Excitement', 'Satisfaction', 'Neutral', 'Confusion',
        'Frustration', 'Disappointment', 'Optimism', 'Pessimism'
    }
    assert prediction['emotion'] in valid_emotions, "Invalid emotion category"

def test_confidence_range():
    """Verify confidence scores are valid probabilities"""
    model = EmotionPredictor('models/best_model.safetensors')
    test_text = "This product is amazing!"
    prediction = model.predict_emotion(test_text)
    
    assert 0 <= prediction['confidence'] <= 1, "Confidence score out of range"