#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import numpy as np
import cv2
import librosa
import ffmpeg as ff
import random
from typing import List, Tuple, Dict
from datetime import datetime, timezone
import pandas as pd
from tensorflow.keras.utils import Sequence # Ensure this is imported for Sequence class
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Concatenate, Lambda
from tensorflow.keras.layers import TimeDistributed # For the video model specifically
from tensorflow.keras import backend as K # For the Lambda layer and weight normalization
from tensorflow.keras.callbacks import EarlyStopping # For your EarlyStopping callback
from tensorflow.keras.utils import Sequence # For your custom DataGenerators
# ========== Helper Functions (same as before) ==========
def extract_audio_from_video(file_path: str) -> np.ndarray:
    inputfile = ff.input(file_path)
    out = inputfile.output('-', format='f32le', acodec='pcm_f32le', ac=1, ar='44100')
    raw = out.run(capture_stdout=True)
    del inputfile, out
    return np.frombuffer(raw[0], np.float32)

def preprocess_audio_series(raw_data: np.ndarray) -> np.ndarray:
    N, M = 24, 1319
    mfcc_data = librosa.feature.mfcc(y=raw_data, n_mfcc=N)
    mfcc_data_standardized = (mfcc_data - np.mean(mfcc_data)) / np.std(mfcc_data)
    number_of_columns_to_fill = M - mfcc_data_standardized.shape[1]
    padding = np.zeros((N, number_of_columns_to_fill))
    padded_data = np.hstack((padding, mfcc_data_standardized))
    return padded_data.reshape(N, M, 1)

def get_number_of_frames(file_path: str) -> int:
    probe = ff.probe(file_path)
    video_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]
    del probe
    return int(video_streams[0]['nb_frames'])

def extract_N_video_frames(file_path: str, number_of_samples: int = 6) -> List[np.ndarray]:
    nb_frames = int(get_number_of_frames(file_path))
    random_indexes = random.sample(range(0, nb_frames), number_of_samples)

    cap = cv2.VideoCapture(file_path)
    video_frames = []

    for ind in random_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind)
        res, frame = cap.read()
        if res:
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    del cap, random_indexes
    return video_frames

def resize_image(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def crop_image_window(image: np.ndarray, training: bool = True) -> np.ndarray:
    height, width, _ = image.shape
    if training:
        MAX_N = height - 128
        MAX_M = width - 128
        rand_N_index = random.randint(0, MAX_N)
        rand_M_index = random.randint(0, MAX_M)
        return image[rand_N_index:rand_N_index + 128, rand_M_index:rand_M_index + 128, :]
    else:
        N_index = (height - 128) // 2
        M_index = (width - 128) // 2
        return image[N_index:N_index + 128, M_index:M_index + 128, :]

def reading_label_data(file_name: str, dictionary: Dict[str, Dict[str, str]]) -> np.ndarray:
    features = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
    extracted_data = [float(dictionary[label][file_name]) for label in features]
    return np.array(extracted_data).reshape(5, 1)

def preprocessing_input(file_path: str, file_name: str, dictionary: Dict[str, Dict[str, str]], training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    extracted_audio_raw = extract_audio_from_video(file_path)
    preprocessed_audio = preprocess_audio_series(raw_data=extracted_audio_raw)

    sampled = extract_N_video_frames(file_path, number_of_samples=6)
    resized_images = [resize_image(image=im, new_size=(248, 140)) for im in sampled]
    cropped_images = [crop_image_window(image=resi, training=training) / 255.0 for resi in resized_images]
    preprocessed_video = np.stack(cropped_images)

    video_gt = reading_label_data(file_name=file_name, dictionary=dictionary)

    del extracted_audio_raw, sampled, resized_images, cropped_images
    return preprocessed_audio, preprocessed_video, video_gt

def save_data_chunk(data: Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]], save_dir: str, filename: str):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.dat")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

def extract_youtube_id_from_filename(filename: str) -> str:
    return filename.split('.')[0]

# ========== Modified Process Function for Set (Memory Efficient Annotation Loading) ==========
def process_dataset(set_name: str, data_path: str, gt_path: str, output_dir: str, training: bool = True, eth_gender_annotations_path: str = None):
    print(f"\nProcessing {set_name} set...")
    with open(gt_path, "rb") as f:
        gt_dict = pickle.load(f, encoding='latin1')

    eth_gender_annotations = {}
    if eth_gender_annotations_path:
        print(f"Loading ethnicity and gender annotations from: {eth_gender_annotations_path}")
        for chunk in pd.read_csv(eth_gender_annotations_path, sep=';', chunksize=1000):
            for index, row in chunk.iterrows():
                youtube_id = row['YouTubeID']
                ethnicity = row['Ethnicity']
                gender = row['Gender']
                eth_gender_annotations[youtube_id] = {'Ethnicity': ethnicity, 'Gender': gender}
        print("Ethnicity and gender annotations loaded into memory.")

    t1 = datetime.now(timezone.utc)
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if not file_path.lower().endswith((".mp4", ".avi", ".mov")):
            continue  # Skip non-video files

        try:
            print(f"Processing file: {filename}")
            base_filename = filename.split('.')[0]
            data = preprocessing_input(file_path=file_path, file_name=filename, dictionary=gt_dict, training=training)

            labels = {}
            if eth_gender_annotations:
                youtube_id = extract_youtube_id_from_filename(base_filename)
                if youtube_id in eth_gender_annotations:
                    labels = eth_gender_annotations[youtube_id]
                else:
                    labels['Ethnicity'] = None
                    labels['Gender'] = None

            data_with_labels = (*data, labels)  # Add the labels dictionary to the tuple
            save_data_chunk(data_with_labels, output_dir, base_filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    t2 = datetime.now(timezone.utc)
    print(f"{set_name} Set - Elapsed time: {t2 - t1}")

# ========== Run All Sets (Modified to include annotation paths) ==========

training_annotations_path = r"C:\\eth_gender_annotations_dev.csv"
validation_annotations_path = r"C:\\eth_gender_annotations_dev.csv"
test_annotations_path = r"C:\\eth_gender_annotations_test.csv"

process_dataset(
    set_name="Training",
    data_path=r"C:\\train",
    gt_path=r"C:\\annotation_training.pkl",
    output_dir="training_chunks_labeled",  # New output directory
    training=True,
    eth_gender_annotations_path=training_annotations_path
)

process_dataset(
    set_name="Validation",
    data_path=r"C:\\val",
    gt_path=r"C:\r\annotation_validation.pkl",
    output_dir="validation_chunks_labeled",  # New output directory
    training=False,
    eth_gender_annotations_path=validation_annotations_path
)

process_dataset(
    set_name="Test",
    data_path=r"C:\1\test",
    gt_path=r"C:\r\annotation_test.pkl",
    output_dir="test_chunks_labeled",  # New output directory
    training=False,
    eth_gender_annotations_path=test_annotations_path
)

