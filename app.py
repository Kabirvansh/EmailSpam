import streamlit as st
from model import SpamDetector

def main():
    st.set_page_config(
        page_title="Email Spam Detector",
        page_icon="📧",
        layout="centered"
    )
    
    st.title("📧 Email Spam Detector")
    st.write("""
    ### Check if your email is spam or not!
    Simply paste your email content below and we'll analyze it for you.
    """)
    
    # Load the pre-trained model
    try:
        detector = SpamDetector.load_model('vectorizer.pkl', 'model.pkl')
    except FileNotFoundError:
        st.error("""
        Model files not found! Please ensure you've run the training script first:
        ```
        python train.py
        ```
        """)
        return
    
    # Text input
    text_input = st.text_area(
        "Paste your email content here:",
        height=200,
        placeholder="Enter the email content you want to analyze..."
    )
    
    if st.button("Analyze", type="primary"):
        if not text_input:
            st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Analyzing..."):
            # Get prediction
            spam_probability = detector.predict_proba(text_input)
            spam_score = int(spam_probability * 100)
            
            # Create a progress bar
            st.write("### Spam Score:")
            progress_color = "red" if spam_score > 50 else "green"
            st.progress(spam_score / 100)
            
            # Display the score
            st.markdown(f"""
            <div style='text-align: center; font-size: 24px; font-weight: bold; color: {progress_color};'>
                {spam_score}/100
            </div>
            """, unsafe_allow_html=True)
            
            # Classification result
            if spam_score > 50:
                st.error("🚨 This message is likely SPAM!")
            else:
                st.success("✅ This message appears to be legitimate (HAM).")
            
            # Confidence explanation
            st.info(f"""
            Our model is {abs(50 - spam_score)}% confident in this classification.
            The closer the score is to 100, the more likely it is spam.
            The closer to 0, the more likely it is legitimate.
            """)

if __name__ == "__main__":
    main()
