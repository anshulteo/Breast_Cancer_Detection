import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
import pickle

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# Paths
MODEL_PATH = 'models/breast_cancer_model.h5'
CLASS_INDICES_PATH = 'models/class_indices.pkl'

# Load class indices from training
@st.cache_resource
def load_class_mapping():
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'rb') as f:
            class_indices = pickle.load(f)
        # Invert to get index -> label mapping
        index_to_class = {v: k for k, v in class_indices.items()}
        return index_to_class
    else:
        # Fallback
        return {0: 'benign', 1: 'malignant', 2: 'normal'}

# Class information
CLASS_INFO = {
    'normal': {
        'description': 'Healthy tissue - No tumor detected',
        'risk': '✅ No risk',
        'color': 'green',
        'emoji': '✅'
    },
    'benign': {
        'description': 'Non-cancerous tumor',
        'risk': '⚠️ Low risk',
        'color': 'orange',
        'emoji': '⚠️'
    },
    'malignant': {
        'description': 'Cancerous tumor - Requires medical attention',
        'risk': '🔴 High risk',
        'color': 'red',
        'emoji': '🔴'
    }
}

# Load model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH, compile=False)
            # Recompile for inference
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        return None

# Preprocess image
def preprocess_image(image):
    # Resize to model input size
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    elif img_array.shape[2] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Make prediction
def predict(model, image, class_mapping):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    
    # Get prediction details
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    class_name = class_mapping[class_idx]
    
    # Get all probabilities
    all_probs = {
        class_mapping[i].capitalize(): predictions[0][i] * 100 
        for i in range(len(class_mapping))
    }
    
    # Debug info
    debug_info = {
        'raw_predictions': predictions[0].tolist(),
        'class_idx': int(class_idx),
        'class_mapping': class_mapping
    }
    
    return class_name, confidence, all_probs, debug_info

# Main UI
def main():
    # Title
    st.title("🔬 Breast Cancer Classification System")
    st.markdown("### Ultrasound Image Analysis using Deep Learning")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application classifies breast ultrasound images into:
        
        - **Normal**: Healthy tissue
        - **Benign**: Non-cancerous tumor  
        - **Malignant**: Cancerous tumor
        
        **Instructions:**
        1. Upload an ultrasound image
        2. Click 'Analyze Image'
        3. View results
        """)
        
        st.markdown("---")
        st.warning("⚠️ **Disclaimer**: For educational purposes only. Not a substitute for professional medical diagnosis.")
        
        # Model info
        if os.path.exists(MODEL_PATH):
            st.success("✅ Model loaded")
        else:
            st.error("❌ Model not found")
    
    # Load model and class mapping
    model = load_model()
    class_mapping = load_class_mapping()
    
    if model is None:
        st.error("❌ Model not found! Train the model first using `python train_model.py`")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an ultrasound image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a breast ultrasound image"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', width=400)
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        class_name, confidence, all_probs, debug_info = predict(model, image, class_mapping)
                        
                        # Store in session state
                        st.session_state['prediction'] = class_name
                        st.session_state['confidence'] = confidence
                        st.session_state['all_probs'] = all_probs
                        st.session_state['debug_info'] = debug_info
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        st.exception(e)
    
    with col2:
        st.subheader("📊 Results")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            confidence = st.session_state['confidence']
            all_probs = st.session_state['all_probs']
            debug_info = st.session_state.get('debug_info', {})
            
            # Get info for this class
            info = CLASS_INFO.get(prediction.lower(), CLASS_INFO['benign'])
            
            # Display result
            st.markdown(f"## {info['emoji']} Classification: **:{info['color']}[{prediction.capitalize()}]**")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            st.progress(min(confidence / 100, 1.0))
            
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Risk Level:** {info['risk']}")
            
            # Probabilities
            st.markdown("---")
            st.markdown("### 📈 Probability Distribution")
            
            for class_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                class_info = CLASS_INFO.get(class_name.lower(), {})
                emoji = class_info.get('emoji', '•')
                
                cols = st.columns([1, 4])
                with cols[0]:
                    st.write(f"{emoji} **{class_name}**")
                with cols[1]:
                    st.progress(prob / 100)
                    st.caption(f"{prob:.2f}%")
            
            # Debug expander
            with st.expander("🔍 Debug Information"):
                st.json(debug_info)
        
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
    
    # Information section
    st.markdown("---")
    st.markdown("### 📚 Classification Categories")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### ✅ Normal")
        st.markdown("- Healthy tissue")
        st.markdown("- No tumor present")
        st.markdown("- No immediate action needed")
    
    with col2:
        st.markdown("#### ⚠️ Benign")
        st.markdown("- Non-cancerous tumor")
        st.markdown("- May require monitoring")
        st.markdown("- Low risk, treatable")
    
    with col3:
        st.markdown("#### 🔴 Malignant")
        st.markdown("- Cancerous tumor")
        st.markdown("- Requires immediate medical attention")
        st.markdown("- Consult oncologist immediately")

if __name__ == "__main__":
    main()
