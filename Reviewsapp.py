import streamlit as st
import pandas as pd
import json
from model import EmotionPredictor
import plotly.express as px

# Define emotion emoji mapping
EMOTION_EMOJIS = {
    'Excitement': '🌟',
    'Satisfaction': '😊',
    'Neutral': '😐',
    'Confusion': '🤔',
    'Frustration': '😤',
    'Disappointment': '😞',
    'Optimism': '🌈',
    'Pessimism': '🌧️'
}

@st.cache_resource
def load_model():
    model_path = 'models/best_model.safetensors'
    return EmotionPredictor(model_path)

# Rest of your helper functions remain the same until add_filters()

def display_review_analysis(review_text, prediction, index):
    """Display the analysis for a single review"""
    emotion = prediction['emotion']
    emoji = EMOTION_EMOJIS.get(emotion, '❓')
    
    with st.expander(f"Review {index + 1}: {review_text[:50]}...", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Review Text:**")
            st.write(review_text)
            st.markdown(f"**Emotion:** {emoji} {emotion}")
            
            # Confidence score
            confidence = prediction['confidence']
            confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            col1_1, col1_2 = st.columns([3, 1])
            col1_1.markdown(f"**Confidence:** {confidence_color} {confidence:.2%}")
            
            # Entropy score
            entropy = prediction['entropy']
            entropy_color = "🟢" if entropy < 0.5 else "🟡" if entropy < 1.0 else "🔴"
            col1_3, col1_4 = st.columns([3, 1])
            col1_3.markdown(f"**Entropy:** {entropy_color} {entropy:.4f}")
        
        with col2:
            st.markdown("**Reasoning:**")
            st.write(prediction['reasoning'])
            
            # Add probability distribution chart
            if 'probabilities' in prediction:
                probs = prediction['probabilities']
                prob_df = pd.DataFrame({
                    'Emotion': [f"{EMOTION_EMOJIS[e]} {e}" for e in probs.keys()],
                    'Probability': list(probs.values())
                })
                fig = px.bar(prob_df, x='Emotion', y='Probability',
                            title='Emotion Probability Distribution')
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)

def display_batch_summary(all_predictions):
    """Display summary statistics for batch processing"""
    if all_predictions:
        st.subheader("Batch Analysis Summary")
        
        # Count emotions by creating a dictionary that tallies each emotion type
        emotion_counts = {}
        for pred in all_predictions:
            emotion = pred['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Create a summary chart using pandas and plotly
        summary_df = pd.DataFrame({
            'Emotion': [f"{EMOTION_EMOJIS[e]} {e}" for e in emotion_counts.keys()],
            'Count': list(emotion_counts.values())
        })
        
        # Display a pie chart showing the distribution of emotions
        fig = px.pie(summary_df, values='Count', names='Emotion',
                     title='Distribution of Emotions in Reviews')
        st.plotly_chart(fig)
        
        # Calculate and display the average confidence score
        avg_confidence = sum(pred['confidence'] for pred in all_predictions) / len(all_predictions)
        st.metric("Average Confidence Score", f"{avg_confidence:.2%}")

def add_export_button(all_predictions, reviews):
    """Add button to export analysis results as a CSV file"""
    if st.button("Export Results"):
        # Create a DataFrame containing all analysis results
        export_df = pd.DataFrame({
            'Review': reviews,
            'Emotion': [p['emotion'] for p in all_predictions],
            'Confidence': [p['confidence'] for p in all_predictions],
            'Entropy': [p['entropy'] for p in all_predictions],
            'Reasoning': [p['reasoning'] for p in all_predictions]
        })
        
        # Convert the DataFrame to CSV format
        csv = export_df.to_csv(index=False)
        
        # Create a download button for the CSV file
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="emotion_analysis_results.csv",
            mime="text/csv"
        )

def create_sidebar():
    """Creates and manages all sidebar elements including collapsible filters"""
    with st.sidebar:
        # First section: Emotion Legend
        st.header("Emotion Legend")
        for emotion, emoji in EMOTION_EMOJIS.items():
            st.write(f"{emoji} {emotion}")
        
        # Second section: Filters in a collapsible expander
        with st.expander("Filter by Emotions", expanded=False):  # expanded=True means it starts open
            selected_emotions = st.multiselect(
                "Select emotions to display:",  # Made label more descriptive
                options=list(EMOTION_EMOJIS.keys()),
                default=list(EMOTION_EMOJIS.keys()),
                help="Choose which emotions you want to see in the analysis results"  # Added helpful tooltip
            )
        
        # Third section: Understanding Metrics with existing collapsible sections
        st.header("Understanding the Metrics")
        
        with st.expander("What is Confidence Score?"):
            st.markdown("""
            The confidence score indicates how certain the model is about its emotion prediction for a given review. Think of it like a student's certainty about their answer on a test.

            What it tells you:
            - 🟢 High (>80%): The model is very confident about its prediction, suggesting strong emotional signals in the text
            - 🟡 Medium (60-80%): The model sees clear but not overwhelming evidence for the predicted emotion
            - 🔴 Low (<60%): The model is less certain, suggesting mixed or subtle emotional signals

            A higher confidence score generally means you can rely more on the prediction, but it should always be considered alongside the entropy score.
            """)
        
        with st.expander("What is Entropy Loss?"):
            st.markdown("""
            Entropy measures how much uncertainty exists across all possible emotions for a review. Think of it as how much the model is "second-guessing" itself between different emotions.

            What it tells you:
            - 🟢 Low (<0.5): The model strongly favors one emotion over others, indicating a clear emotional signal
            - 🟡 Medium (0.5-1.0): The model sees evidence for multiple emotions, showing some ambiguity
            - 🔴 High (>1.0): The model detects multiple competing emotions, suggesting complex or mixed emotional content

            Lower entropy generally indicates more reliable predictions, as the model is clearly distinguishing one emotion from others.
            """)
        
        return selected_emotions


def display_header():
    """Creates a header section with title and collapsible description of the emotion analyzer model"""
    
    # Add custom CSS for styling the expander and its content
    st.markdown("""
        <style>
            /* Enhanced styling for the expander header */
            .streamlit-expanderHeader {
                font-size: 1.3em !important;
                background: linear-gradient(90deg, #1E3D59 0%, #2B5876 50%, #4E4376 100%);
                color: white !important;
                padding: 15px 20px !important;
                border-radius: 8px !important;
                border: none !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            }
            
            /* Hover effect for better interactivity */
            .streamlit-expanderHeader:hover {
                background: linear-gradient(90deg, #2B5876 0%, #4E4376 50%, #1E3D59 100%);
                transition: background 0.3s ease;
            }
            
            /* Styling for the expanded content area */
            .streamlit-expanderContent {
                border: 1px solid #e0e0e0;
                border-radius: 0 0 8px 8px;
                padding: 20px;
                background-color: #ffffff;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Main title section with professional styling
    st.markdown("""
        <h1 style='text-align: center; 
                   color: #1E3D59; 
                   padding: 20px 0 10px 0; 
                   font-size: 2.2em; 
                   font-weight: 600;'>
            🎯 ProductReview Emotion Analyzer
        </h1>
        """, unsafe_allow_html=True)
    
    # Create columns for better space utilization
    col1, col2, col3 = st.columns([0.05, 3.9, 0.05])
    
    with col2:
        # Top decorative line with gradient
        st.markdown("""
            <div style='background: linear-gradient(to right, transparent, #1E3D59, transparent); 
                        height: 2px; 
                        margin: 10px 0;'>
            </div>
            """, unsafe_allow_html=True)
        
        # Collapsible description section with enhanced formatting
        with st.expander("📖 About this Model - Understanding the Emotion Analyzer"):
            st.markdown("""
            <div style='font-size: 0.8em; line-height: 1.6; color: #2C3E50;'>
            
            <p><strong>Product review emotion analyzer</strong> represents a sophisticated approach to understanding customer sentiment. 
            Let's explore how it works:</p>
            
            <p>Our emotion classification system utilizes <strong>BERT</strong> (Bidirectional Encoder Representations from Transformers), 
            an advanced AI language model that excels at understanding context and nuances in text. This model has been 
            carefully fine-tuned on 1,500 expertly labeled luxury fashion reviews to ensure accurate emotion detection 
            specifically within this domain.</p>
            
            <p>The system analyzes customer feedback across eight distinct emotional categories, each chosen to capture 
            the full spectrum of customer experiences in luxury fashion:</p>
            
            <div style='background: linear-gradient(to right, #f8f9fa, #ffffff);
                        padding: 20px 25px;
                        border-radius: 8px;
                        margin: 15px 0;'>
                <p style='margin: 10px 0;'>✨ <strong>Excitement</strong> - Captures moments of delight and enthusiasm</p>
                <p style='margin: 10px 0;'>😊 <strong>Satisfaction</strong> - Reflects contentment and met expectations</p>
                <p style='margin: 10px 0;'>😐 <strong>Neutral</strong> - Indicates balanced or objective observations</p>
                <p style='margin: 10px 0;'>🤔 <strong>Confusion</strong> - Highlights areas needing clarity</p>
                <p style='margin: 10px 0;'>😤 <strong>Frustration</strong> - Identifies pain points in customer experience</p>
                <p style='margin: 10px 0;'>😞 <strong>Disappointment</strong> - Reveals unmet expectations</p>
                <p style='margin: 10px 0;'>🌈 <strong>Optimism</strong> - Shows positive outlook and potential</p>
                <p style='margin: 10px 0;'>🌧️ <strong>Pessimism</strong> - Indicates concerns or skepticism</p>
            </div>
            
            <p>Beyond simple classification, the analyzer provides confidence scores and explains its 
            reasoning by highlighting key phrases in the review text. This detailed analysis helps luxury retailers 
            gain actionable insights into customer experiences, enabling targeted improvements in products and services.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Bottom decorative line with gradient
        st.markdown("""
            <div style='background: linear-gradient(to right, transparent, #1E3D59, transparent); 
                        height: 2px; 
                        margin: 10px 0 20px 0;'>
            </div>
            """, unsafe_allow_html=True)
def main():
    display_header()
    
    # Create sidebar and get selected emotions
    selected_emotions = create_sidebar()
    
    # Load model
    predictor = load_model()
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Upload CSV/JSON", "Manual Text Input"])
    
    if input_method == "Upload CSV/JSON":
        uploaded_file = st.file_uploader("Upload CSV or JSON file", type=['csv', 'json'])
        
        if uploaded_file:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    content = uploaded_file.getvalue().decode('utf-8')
                    reviews = [json.loads(line) for line in content.strip().split('\n')]
                    df = pd.DataFrame(reviews)
                
                # Process reviews
                st.subheader(f"Analyzing {len(df)} Reviews")
                progress_bar = st.progress(0)
                all_predictions = []
                
                for idx, row in df.iterrows():
                    review = row.get('reviewText', '')
                    if review:
                        prediction = predictor.predict_emotion(review)
                        all_predictions.append(prediction)
                        progress_bar.progress((idx + 1) / len(df))
                
                # Filter predictions based on selected emotions
                filtered_predictions = [
                    (df['reviewText'][idx], pred) for idx, pred in enumerate(all_predictions)
                    if pred['emotion'] in selected_emotions
                ]
                
                # Display filtered reviews
                if filtered_predictions:
                    for idx, (review, prediction) in enumerate(filtered_predictions):
                        display_review_analysis(review, prediction, idx)
                else:
                    st.warning("No reviews match the selected filters.")
                
                # Display batch summary and export button
                if filtered_predictions:
                    display_batch_summary([pred for _, pred in filtered_predictions])
                    add_export_button([pred for _, pred in filtered_predictions], 
                                    [review for review, _ in filtered_predictions])
                    st.success("Analysis Complete! 🎉")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:
        # Manual text input
        review_text = st.text_area("Enter your review here:")
        
        if st.button("Analyze") and review_text:
            prediction = predictor.predict_emotion(review_text)
            display_review_analysis(review_text, prediction, 0)

if __name__ == "__main__":
    main()