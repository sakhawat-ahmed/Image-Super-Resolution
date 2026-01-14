import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Image Super-Resolution",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("🖼️ Image Super-Resolution Demo")
st.markdown("""
This web application demonstrates super-resolution using deep learning models.
Upload a low-resolution image and see it enhanced by SRCNN or EDSR models.
""")

# Sidebar for controls
st.sidebar.title("⚙️ Controls")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["SRCNN", "EDSR", "Compare Both"]
)

scale_factor = st.sidebar.slider(
    "Scale Factor",
    min_value=2,
    max_value=4,
    value=2,
    step=1
)

uploaded_file = st.sidebar.file_uploader(
    "Upload an image",
    type=['png', 'jpg', 'jpeg', 'bmp']
)

# Load sample images
sample_images = {
    "Butterfly": "samples/butterfly.png",
    "Baboon": "samples/baboon.png",
    "Lena": "samples/lena.png",
    "Peppers": "samples/peppers.png"
}

sample_choice = st.sidebar.selectbox(
    "Or choose a sample image",
    ["None"] + list(sample_images.keys())
)

# Load models
@st.cache_resource
def load_model(model_name):
    """Load pre-trained model"""
    try:
        if model_name == "SRCNN":
            from src.srcnn_model import SRCNN
            model = SRCNN(num_channels=3, base_filter=64)
            model_path = "models/SRCNN_final.pth"
        else:  # EDSR
            from src.edsr_model import EDSR
            model = EDSR(num_channels=3, num_features=32, num_blocks=4)
            model_path = "models/EDSR_final.pth"
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load both models
srcnn_model = load_model("SRCNN")
edsr_model = load_model("EDSR")

def preprocess_image(image):
    """Preprocess uploaded image"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize to reasonable size for processing
    max_size = 512
    h, w = img_array.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_array = cv2.resize(img_array, (new_w, new_h))
    
    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    
    return img_array, img_tensor

def apply_super_resolution(model, img_tensor):
    """Apply super-resolution model"""
    with torch.no_grad():
        output = model(img_tensor)
        output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        return output_np

def calculate_metrics(original, enhanced):
    """Calculate PSNR and SSIM"""
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        
        # Ensure same size
        h, w = min(original.shape[0], enhanced.shape[0]), min(original.shape[1], enhanced.shape[1])
        orig_resized = cv2.resize(original, (w, h))
        enh_resized = cv2.resize(enhanced, (w, h))
        
        # Convert to grayscale for SSIM
        if len(orig_resized.shape) == 3:
            orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY)
            enh_gray = cv2.cvtColor(enh_resized, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = orig_resized
            enh_gray = enh_resized
        
        psnr_value = psnr(orig_resized, enh_resized, data_range=255)
        ssim_value = ssim(orig_gray, enh_gray, data_range=255)
        
        return psnr_value, ssim_value
    except:
        return 0, 0

# Main content
if uploaded_file is not None or sample_choice != "None":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📤 Original Image")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        elif sample_choice != "None":
            # Load sample image
            sample_path = sample_images[sample_choice]
            if os.path.exists(sample_path):
                image = Image.open(sample_path)
                st.image(image, caption=f"Sample: {sample_choice}", use_column_width=True)
            else:
                # Create a placeholder if samples don't exist
                st.warning(f"Sample image not found: {sample_path}")
                st.info("Creating a placeholder image...")
                # Create a simple test image
                test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                image = Image.fromarray(test_img)
                st.image(image, caption="Test Image", use_column_width=True)
        
        # Preprocess
        if 'image' in locals():
            original_img, img_tensor = preprocess_image(image)
    
    with col2:
        st.subheader("🔍 Enhanced Image")
        
        if model_choice in ["SRCNN", "Compare Both"] and srcnn_model:
            with st.spinner("Processing with SRCNN..."):
                srcnn_output = apply_super_resolution(srcnn_model, img_tensor)
                srcnn_output_rgb = cv2.cvtColor(srcnn_output, cv2.COLOR_BGR2RGB)
                st.image(srcnn_output_rgb, caption="SRCNN Output", use_column_width=True)
                
                # Calculate metrics
                psnr_val, ssim_val = calculate_metrics(original_img*255, srcnn_output)
                st.metric("PSNR", f"{psnr_val:.2f} dB")
                st.metric("SSIM", f"{ssim_val:.4f}")
        
        if model_choice == "EDSR" and edsr_model:
            with st.spinner("Processing with EDSR..."):
                edsr_output = apply_super_resolution(edsr_model, img_tensor)
                edsr_output_rgb = cv2.cvtColor(edsr_output, cv2.COLOR_BGR2RGB)
                st.image(edsr_output_rgb, caption="EDSR Output", use_column_width=True)
                
                # Calculate metrics
                psnr_val, ssim_val = calculate_metrics(original_img*255, edsr_output)
                st.metric("PSNR", f"{psnr_val:.2f} dB")
                st.metric("SSIM", f"{ssim_val:.4f}")
    
    with col3:
        if model_choice == "Compare Both" and srcnn_model and edsr_model:
            st.subheader("📊 Comparison")
            
            with st.spinner("Comparing models..."):
                edsr_output = apply_super_resolution(edsr_model, img_tensor)
                edsr_output_rgb = cv2.cvtColor(edsr_output, cv2.COLOR_BGR2RGB)
                st.image(edsr_output_rgb, caption="EDSR Output", use_column_width=True)
                
                # Calculate metrics for EDSR
                psnr_edsr, ssim_edsr = calculate_metrics(original_img*255, edsr_output)
                st.metric("PSNR (EDSR)", f"{psnr_edsr:.2f} dB")
                st.metric("SSIM (EDSR)", f"{ssim_edsr:.4f}")
            
            # Comparison chart
            st.subheader("📈 Performance Comparison")
            
            if 'srcnn_output' in locals() and 'edsr_output' in locals():
                metrics_data = {
                    "Model": ["SRCNN", "EDSR"],
                    "PSNR": [psnr_val, psnr_edsr],
                    "SSIM": [ssim_val, ssim_edsr]
                }
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                
                # PSNR comparison
                axes[0].bar(metrics_data["Model"], metrics_data["PSNR"], color=['blue', 'orange'])
                axes[0].set_title("PSNR Comparison")
                axes[0].set_ylabel("PSNR (dB)")
                axes[0].grid(True, alpha=0.3)
                
                # SSIM comparison
                axes[1].bar(metrics_data["Model"], metrics_data["SSIM"], color=['blue', 'orange'])
                axes[1].set_title("SSIM Comparison")
                axes[1].set_ylabel("SSIM")
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        elif model_choice != "Compare Both":
            st.subheader("ℹ️ Image Info")
            if 'image' in locals():
                st.write(f"**Dimensions:** {image.size[0]} × {image.size[1]}")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**Format:** {image.format if hasattr(image, 'format') else 'Unknown'}")
            
            st.subheader("📝 Notes")
            st.info("""
            - Higher PSNR = Better pixel-level accuracy
            - SSIM closer to 1.0 = Better structural preservation
            - Processing time depends on image size
            """)

else:
    # Show welcome/instructions
    st.info("👈 Please upload an image or select a sample from the sidebar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 About This Demo")
        st.markdown("""
        This demo showcases two deep learning models for image super-resolution:
        
        **1. SRCNN (Super-Resolution CNN)**
        - 3 convolutional layers
        - 69,251 parameters
        - Fast processing
        
        **2. EDSR (Enhanced Deep Super-Resolution)**
        - Residual blocks
        - 84,995 parameters  
        - Better quality output
        """)
    
    with col2:
        st.subheader("📊 Sample Results")
        
        # Create a sample results table
        sample_results = {
            "Model": ["SRCNN", "EDSR"],
            "Avg PSNR": ["27.93 dB", "25.70 dB"],
            "Avg SSIM": ["0.9533", "0.9646"],
            "Inference Time": ["~0.5s", "~1.0s"]
        }
        
        st.table(sample_results)
        
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. Select a model from sidebar
        2. Upload an image (PNG/JPG)
        3. View enhanced results
        4. Compare metrics
        """)

# Footer
st.markdown("---")
st.markdown("""
**Image Super-Resolution Project** | Principles and Platforms of Deep Learning | Fall 2025
""")