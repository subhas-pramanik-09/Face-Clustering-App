import streamlit as st
import os
import shutil
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        color: #2E86C1;
        text-align: center;
        padding: 20px;
    }
    .image-container {
        max-width: 300px !important;
        margin: 5px !important;
        display: inline-block !important;
    }
    .group-header {
        font-size: 24px !important;
        color: #148F77;
        margin: 20px 0 !important;
    }
    img {
        max-height: 250px !important;
        width: auto !important;
        object-fit: contain !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<p class="header">👥 Face Clustering System</p>', unsafe_allow_html=True)

# Initialize models with caching
@st.cache_resource
def load_models():
    return MTCNN(image_size=160, margin=20, keep_all=False), InceptionResnetV1(pretrained='vggface2').eval()

mtcnn, resnet = load_models()

# Sidebar Controls
with st.sidebar:
    st.subheader("Settings")
    eps_value = st.slider("Clustering Sensitivity (eps)", 0.1, 2.0, 0.7, 0.1)
    min_samples = st.slider("Minimum Samples", 1, 5, 2)

# File Upload Section
uploaded_files = st.file_uploader("Upload images", 
                                type=['jpg', 'jpeg', 'png'], 
                                accept_multiple_files=True)

# Processing Section
if uploaded_files:
    # Save uploaded files
    os.makedirs("faces", exist_ok=True)
    for file in uploaded_files:
        with open(f"faces/{file.name}", "wb") as f:
            f.write(file.getbuffer())
    
    # Display uploaded images
    st.markdown('<p class="subheader">Uploaded Faces</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for idx, img_path in enumerate(os.listdir("faces")):
        with cols[idx % 4]:
            img = Image.open(f"faces/{img_path}")
            st.image(img, use_container_width=True)

if uploaded_files and st.button("Start Clustering"):
    with st.spinner('🔍 Processing images - Please wait...'):
        # Extract embeddings
        face_embeddings = []
        file_paths = []
        
        for filename in os.listdir("faces"):
            path = f"faces/{filename}"
            try:
                img = Image.open(path).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    embedding = resnet(face.unsqueeze(0))
                    face_embeddings.append(embedding.detach().numpy()[0])
                    file_paths.append(path)
            except Exception as e:
                pass
        
        if face_embeddings:
            # Perform clustering
            face_embeddings = np.array(face_embeddings)
            clustering = DBSCAN(eps=eps_value, min_samples=min_samples, metric='euclidean')
            labels = clustering.fit_predict(face_embeddings)
            
            # Group results
            grouped_faces = {}
            for label, path in zip(labels, file_paths):
                grouped_faces.setdefault(label, []).append(path)
            
            # Display results
            st.markdown('<p class="subheader">Clustering Results</p>', unsafe_allow_html=True)
            
            for group_num, (label, paths) in enumerate(grouped_faces.items(), 1):
                st.markdown(f'<p class="group-header">Group {group_num}</p>', unsafe_allow_html=True)
                cols = st.columns(4)
                
                for idx, path in enumerate(paths):
                    with cols[idx % 4]:
                        img = Image.open(path)
                        # Maintain aspect ratio with fixed height
                        img.thumbnail((300, 300))
                        st.image(img, use_container_width=True)
                
                # Add space between groups
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Cleanup
            shutil.rmtree("faces")