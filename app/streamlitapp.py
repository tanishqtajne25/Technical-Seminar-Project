# Import all of the dependencies
import streamlit as st
import os
import imageio

import tensorflow as tf
from utils import load_data, num_to_char, DATA_ROOT
from modelutil import load_model

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')

# Speaker selector
speakers = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
selected_speaker = st.selectbox('Choose speaker', speakers)

# List videos for selected speaker
speaker_path = os.path.join(DATA_ROOT, selected_speaker)
options = [f for f in os.listdir(speaker_path) if f.endswith('.mpg')]
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    file_path = os.path.join(speaker_path, selected_video)

    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        import imageio_ffmpeg
        import subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # We write to a consistent temp file, then read it.
        # Streamlit reads byte by byte so caching isn't usually an issue if the file actually updates.
        subprocess.run([ffmpeg_exe, '-i', file_path, '-vcodec', 'libx264', 'test_video.mp4', '-y'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Rendering inside of the app
        with open('test_video.mp4', 'rb') as video:
            video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video_frames, annotations = load_data(tf.convert_to_tensor(file_path))
        
        # Convert float32 tensor to uint8 for GIF creation
        # Normalization was (frame - mean)/std. We need to map it roughly to 0-255.
        video_np = video_frames.numpy()
        video_np = video_np - video_np.min() # shift to 0
        video_np = (video_np / video_np.max() * 255).astype('uint8')
        
        # Imageio pillow plugin expects (H, W) for grayscale, not (H, W, 1)
        import numpy as np
        video_np = np.squeeze(video_np, axis=-1)
        
        imageio.mimsave('animation.gif', video_np, fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video_frames, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
