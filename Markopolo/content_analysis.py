import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torchvision.transforms as transforms
from torchvision import models
from moviepy.editor import VideoFileClip
from PIL import Image
from transformers import pipeline

# Load Pretrained Models
image_model = models.resnet50(pretrained=True)
image_model.eval()
txt_model = pipeline("sentiment-analysis")
audio_transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Function to Extract Features from Images
def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = image_model(image_tensor).numpy()
    return features

# Function to Extract Keyframes from Videos
def extract_keyframes(uploaded_file, num_frames=5):
    """Extracts keyframes from a video file."""
    try:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        clip = VideoFileClip(temp_path)
        duration = clip.duration
        frame_times = np.linspace(0, duration, num_frames, endpoint=False)
        keyframes = []

        for time in frame_times:
            frame = clip.get_frame(time)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keyframes.append(frame)

        clip.close()
        os.remove(temp_path)  # Clean up temp file
        return keyframes
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return []

# Function to Transcribe Audio from Video
def transcribe_audio(video_path):
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        audio_path = "temp_audio.wav"
        audio.write_audiofile(audio_path)
        transcription = audio_transcriber(audio_path)
        os.remove(audio_path)
        return transcription['text']
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

# Function to Analyze Performance Data
def analyze_performance(performance_data):
    """Processes performance data from a DataFrame."""
    try:
        df = pd.read_csv(performance_data)
        df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric where possible
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns available for correlation analysis.")
        
        correlation = numeric_df.corr()
        return correlation
    except Exception as e:
        st.error(f"Error analyzing performance data: {e}")
        return None

# Function to Visualize Performance Insights
def visualize_performance(correlation_matrix):
    """Creates a heatmap for correlation analysis."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Feature-Performance Correlation")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error generating visualization: {e}")

# Streamlit Interface
st.title("AI Agent for Advertising Creative Analysis")

# File Uploaders
uploaded_images = st.file_uploader("Upload Image Creatives", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
uploaded_videos = st.file_uploader("Upload Video Creatives", type=["mp4"], accept_multiple_files=True)
performance_data = st.file_uploader("Upload Performance Data (CSV)", type=["csv"])

# Analyze Button
if st.button("Analyze Creatives"):
    
    # Image Analysis
    if uploaded_images:
        st.subheader("Image Analysis")
        for img in uploaded_images:
            try:
                features = extract_image_features(img)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                st.text(f"Extracted Features: {features.shape}")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    # Video Analysis
    if uploaded_videos:
        st.subheader("Video Analysis")
        for vid in uploaded_videos:
            keyframes = extract_keyframes(vid)
            st.text(f"Extracted {len(keyframes)} keyframes.")
            for idx, frame in enumerate(keyframes):
                st.image(frame, caption=f"Keyframe {idx+1}", use_column_width=True)

    # Performance Data Analysis
    if performance_data:
        st.subheader("Performance Analysis")
        correlation_matrix = analyze_performance(performance_data)
        if correlation_matrix is not None:
            visualize_performance(correlation_matrix)
            st.text("Correlation Analysis Completed!")