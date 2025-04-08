# Importing necessary libraries
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
from langdetect import detect
from transformers import pipeline
from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity as ssim

# Loading pre-trained models for Image, Video ads analysis
image_model = models.resnet50(pretrained=True) # for ad image anaysis
image_model.eval()
txt_model = pipeline("sentiment-analysis") # For analyzing the extracted text from above image
audio_transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')  # PADDLEOCR engine

# Function used to extract text from Image ad using OCR
def extract_ocr_text(image_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    result = ocr_model.ocr(image, cls=True)
    extracted_text = " ".join([line[1][0] for line in result[0]]) if result[0] else ""
    return extracted_text

# Function used to extract features from ad images
def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0) # transforming features into tensors
    with torch.no_grad():
        features = image_model(image_tensor).numpy()
    return features

# Function used to delete duplicate video frames using SSIM
def is_similar(img1, img2, threshold=0.95):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score > threshold

# Function used to extract keyframes from video by storing the video as temporary video to remove duplicte frames
def extract_keyframes(uploaded_file, num_frames=10, similarity_threshold=0.95):
    try:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        clip = VideoFileClip(temp_path)
        duration = clip.duration
        frame_times = np.linspace(0, duration, num_frames, endpoint=False)
        keyframes = []

        last_frame = None
        for time in frame_times:
            frame = clip.get_frame(time)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if last_frame is None or not is_similar(frame, last_frame, threshold=similarity_threshold):
                keyframes.append(frame)
                last_frame = frame

        clip.close()
        os.remove(temp_path)
        return keyframes
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return []

# Function used to transcribe audio from ad video using openai whisper
def transcribe_audio(uploaded_file):
    try:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        clip = VideoFileClip(temp_path)
        audio = clip.audio
        audio_path = "transcribed_audio.wav"
        audio.write_audiofile(audio_path, logger=None)
        # Closing the video properly from preventing conflicts
        audio.close()
        clip.close()
        transcription = audio_transcriber(audio_path)
        os.remove(temp_path)
        return transcription['text']
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

# Function used to analyze performance metrics of ads
def analyze_performance(performance_data):
    try:
        df = pd.read_csv(performance_data)
        df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            raise ValueError("No numeric columns available for correlation analysis.")
        correlation = numeric_df.corr()
        return correlation
    except Exception as e:
        st.error(f"Error analyzing performance data: {e}")
        return None

# Function used to visualize correlation heatmap using matplotlib from performance metrics
def visualize_performance(correlation_matrix):
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Feature-Performance Correlation")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error generating visualization: {e}")

# Streamlit UI for user interaction
st.title("AI-powered Agent for Ad Creatives Analysis")

uploaded_images = st.file_uploader("Upload Image Creatives", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
uploaded_videos = st.file_uploader("Upload Video Creatives", type=["mp4"], accept_multiple_files=True)
performance_data = st.file_uploader("Upload Performance Data (CSV)", type=["csv"])

if st.button("Analyze uploaded AD Creatives"):

    # UI image anaysis
    if uploaded_images:
        st.subheader("🖼Image Analysis")
        for img in uploaded_images:
            try:
                features = extract_image_features(img)
                ocr_text = extract_ocr_text(img)
                language = detect(ocr_text) if ocr_text else "Unknown"
                st.image(img, caption="Uploaded Image", use_column_width=True)
                st.text(f"Extracted Feature Shape: {features.shape}")
                st.text(f"OCR Text: {ocr_text}")
                st.text(f"Detected Language: {language}")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    # UI Video analysis
    if uploaded_videos:
        st.subheader("Video Analysis")
        for vid in uploaded_videos:
            st.text(f"Processing: {vid.name}")
            st.markdown("**Extracted Keyframes:**")
            keyframes = extract_keyframes(vid)
            for idx, frame in enumerate(keyframes):
                st.image(frame, caption=f"Keyframe {idx + 1}", use_column_width=True)

            # Reset file stream after reading in extract_keyframes
            vid.seek(0)
            transcription = transcribe_audio(vid)
            language = detect(transcription) if transcription else "Unknown"
            st.markdown("**🗣️ Transcription:**")
            st.write(transcription)
            st.text(f"Detected Language: {language}")
            st.audio("transcribed_audio.wav")

    # CSV Performance Analysis
    if performance_data:
        st.subheader("Performance Analysis")
        correlation_matrix = analyze_performance(performance_data)
        if correlation_matrix is not None:
            visualize_performance(correlation_matrix)
            st.text("Correlation Analysis Completed..Thank You.")
