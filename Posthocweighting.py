#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt # Although not directly used for plotting in this refined script, useful for general utilities
from keras.models import Model
from keras.utils import Sequence
from sklearn.metrics import mean_absolute_error # For MAE calculation

# Import all layers and applications used in your model definition
from keras.layers import Input, Dense, Lambda, TimeDistributed, LSTM
from keras.applications import vgg16, efficientnet # Import EfficientNet for the audio model

# --- Model Architectures (MUST BE IDENTICAL to how you trained your INDIVIDUAL models) ---
# IMPORTANT: Both models now have a final Dense(5, activation='linear') layer
# to output the 5 personality traits directly.

def create_audio_model_pretrained_efficientnet():
    """
    Defines the audio subnetwork using pre-trained EfficientNetB0.
    Output: 5 personality traits.
    """
    audio_input = Input(shape=(24, 1319, 1), name='audio_input')
    # Resize to 128x128 for EfficientNet and replicate to 3 channels
    resized_input = Lambda(lambda img: tf.image.resize(img, (128, 128)))(audio_input)
    x = Lambda(lambda img: tf.concat([img, img, img], axis=-1))(resized_input)

    base_model = efficientnet.EfficientNetB0(
        weights='imagenet',
        include_top=False, # We don't want the classification head
        input_shape=(128, 128, 3),
        pooling='max' # Global Max Pooling
    )
    base_model.trainable = False # Freeze the pre-trained EfficientNet base

    x = base_model(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(5, activation='linear')(x) # Final layer to predict 5 traits
    return Model(inputs=audio_input, outputs=output, name='audio_subnetwork_efficientnet')


def create_video_model_vgg16():
    """
    Defines the video subnetwork using pre-trained VGG16 and LSTM.
    Output: 5 personality traits.
    """
    visual_model_input = Input(shape=(6, 128, 128, 3), name='video_input')

    # Load VGG16 with ImageNet weights, no top (classification) layer, with max pooling
    cnn_base = vgg16.VGG16(weights="imagenet", include_top=False, pooling='max')
    cnn_base.trainable = False # Freeze the VGG16 base for feature extraction

    # The output of VGG16 with pooling='max' for a (128, 128, 3) input is (None, 512)
    # We need to explicitly define the output shape for the Lambda layer if it's dynamic
    # or ensure it's handled by TimeDistributed
    lambda_output_shape = cnn_base.output_shape[1:]

    # Apply VGG16 to each frame using TimeDistributed
    encoded_frame = TimeDistributed(Lambda(lambda x: cnn_base(x), output_shape=lambda_output_shape))(visual_model_input)

    # Process the sequence of encoded frames with LSTM to capture temporal dynamics
    encoded_vid = LSTM(64)(encoded_frame)
    output = Dense(5, activation='linear')(encoded_vid) # Final layer to predict 5 traits
    return Model(inputs=visual_model_input, outputs=output, name='video_subnetwork_vgg16')

# --- Data Generator for Prediction ---
class CombinedDataGenerator(Sequence):
    """
    A Keras Sequence for loading and batching multimodal (audio, video) data
    for prediction. This is a simplified version compared to the training generator
    as it doesn't need to return targets for batching by Keras.
    """
    def __init__(self, data_dir, audio_input_shape, video_input_shape, batch_size=16):
        self.data_dir = data_dir
        self.audio_input_shape = audio_input_shape
        self.video_input_shape = video_input_shape
        self.batch_size = batch_size
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dat')]
        self.data_files = self._filter_valid_files() # Filter out any problematic files
        self.indices = np.arange(len(self.data_files)) # Initialize indices

    def _filter_valid_files(self):
        """Attempts to load a sample from each file to check for integrity."""
        valid_files = []
        print(f"Checking {len(self.data_files)} data files for integrity for prediction...")
        for i, file_path in enumerate(self.data_files):
            try:
                with open(file_path, 'rb') as f:
                    data_chunk = pickle.load(f, encoding='latin1')
                # Basic check: Ensure it has 4 elements and labels can be flattened to 5 traits
                # Also check if the video and audio shapes are as expected
                if len(data_chunk) == 4 and data_chunk[2].flatten().shape[0] == 5 and \
                   data_chunk[0].shape == self.audio_input_shape and data_chunk[1].shape == self.video_input_shape:
                    valid_files.append(file_path)
                else:
                    print(f"Skipping malformed file (wrong element count, label shape, or modal shape): {file_path}")
            except Exception as e:
                print(f"Skipping corrupted file {file_path}: {e}")
        print(f"Found {len(valid_files)} valid files out of {len(self.data_files)} original files.")
        return valid_files

    def __len__(self):
        # Number of batches
        if not self.data_files:
            return 0
        return int(np.ceil(len(self.data_files) / self.batch_size)) # Use ceil to include last partial batch

    def __getitem__(self, index):
        # Generate indices for the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_audio = []
        batch_video = []
        batch_labels = []
        batch_attributes = [] # To store attributes for fairness evaluation

        for idx in batch_indices:
            file_path = self.data_files[idx]
            try:
                with open(file_path, 'rb') as f:
                    data_chunk = pickle.load(f, encoding='latin1')

                audio_data = data_chunk[0].astype(np.float32)
                video_data = data_chunk[1].astype(np.float32)
                true_label_gt = data_chunk[2].flatten().astype(np.float32)
                attributes = data_chunk[3]

                batch_audio.append(audio_data)
                batch_video.append(video_data)
                batch_labels.append(true_label_gt)
                batch_attributes.append(attributes)

            except Exception as e:
                print(f"Warning: Error loading file {file_path} in generator: {e}. Filling with zeros/None.")
                # Fill with zeros for problematic samples to maintain batch integrity
                batch_audio.append(np.zeros(self.audio_input_shape, dtype=np.float32))
                batch_video.append(np.zeros(self.video_input_shape, dtype=np.float32))
                batch_labels.append(np.zeros(5, dtype=np.float32)) # Assuming 5 traits
                batch_attributes.append({'Ethnicity': -1, 'Gender': -1}) # Placeholder for missing attributes

        # Return inputs as a dictionary for named inputs, and all other data separately
        return {
            'audio_input': np.array(batch_audio),
            'video_input': np.array(batch_video)
        }, np.array(batch_labels), batch_attributes


# --- Fairness Metrics and Evaluation Functions ---

# Personality trait names (global or passed as argument)
trait_names = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

DEFAULT_PRED_THRESHOLD = 0.5
DEFAULT_TRUE_LABEL_THRESHOLD = 0.5

def equal_opportunity(predictions, true_labels, attributes, trait_idx, protected_attr, pred_threshold, true_label_threshold):
    """
    Calculate equal opportunity difference for a specific trait with continuous outcomes.
    Compares the rate of predicted positives among true positives across groups.
    """
    groups_true_positives = {}

    for i, attr in enumerate(attributes):
        group_val = attr.get(protected_attr)

        if group_val is None or group_val == -1: # Skip samples with missing/placeholder attribute
            continue

        # Check if true label for this trait is considered "positive"
        is_true_positive = true_labels[i, trait_idx] > true_label_threshold

        if is_true_positive:
            if group_val not in groups_true_positives:
                groups_true_positives[group_val] = []
            groups_true_positives[group_val].append(i)

    tprs = {}
    for group, indices in groups_true_positives.items():
        if len(indices) == 0:
            tprs[group] = 0.0 # No true positives in this group for this trait
            continue

        # Calculate the rate of predicted positives among these true positives
        predicted_positives_among_true_positives = predictions[indices, trait_idx] > pred_threshold
        tpr = np.mean(predicted_positives_among_true_positives)
        tprs[group] = tpr

    if len(tprs) < 2:
        return 0 # Not enough groups with true positives to compare

    # Calculate difference as max - min
    return max(tprs.values()) - min(tprs.values())

def statistical_parity(predictions, attributes, trait_idx, protected_attr, threshold):
    """
    Calculate statistical parity difference for a specific trait.
    Compares the rate of predicted positives across different groups.
    """
    groups = {}
    for i, attr in enumerate(attributes):
        group = attr.get(protected_attr)
        if group is None or group == -1: # Skip samples with missing/placeholder attribute
            continue
        if group not in groups:
            groups[group] = []
        groups[group].append(i)

    pos_rates = {}
    for group, indices in groups.items():
        if len(indices) == 0:
            pos_rates[group] = 0.0
            continue

        group_preds = predictions[indices, trait_idx] > threshold
        pos_rates[group] = np.mean(group_preds)

    if len(pos_rates) < 2:
        return 0 # Not enough groups to compare

    # Calculate difference as max - min
    return max(pos_rates.values()) - min(pos_rates.values())


def evaluate_fairness(predictions, true_labels, attributes, model_name, pred_threshold, true_label_threshold):
    """
    Evaluates fairness metrics (Equal Opportunity and Statistical Parity) for a model.
    Returns the total average fairness score.
    """
    print(f"\nFairness Evaluation for {model_name} Model")
    print("="*50)

    overall_eo_averages = []

    for protected_attr in ['Gender', 'Ethnicity']:
        print(f"\nProtected Attribute: {protected_attr}")
        print("-"*40)

        eo_values = []

        for trait_idx, trait_name in enumerate(trait_names):
            eo_diff = equal_opportunity(predictions, true_labels, attributes, trait_idx, protected_attr,
                                         pred_threshold=pred_threshold,
                                         true_label_threshold=true_label_threshold)
            sp_diff = statistical_parity(predictions, attributes, trait_idx, protected_attr,
                                         threshold=pred_threshold)

            eo_values.append(eo_diff)

            print(f"{trait_name.capitalize():<15} | Equal Opportunity: {eo_diff:.4f} | Statistical Parity: {sp_diff:.4f}")

        if len(eo_values) > 0:
            avg_eo = sum(eo_values) / len(eo_values)
        else:
            avg_eo = 0.0
        overall_eo_averages.append(avg_eo)
        print(f"\nAverage Equal Opportunity for {protected_attr}: {avg_eo:.4f}")

    if len(overall_eo_averages) > 0:
        total_avg_fairness = sum(overall_eo_averages) / len(overall_eo_averages)
    else:
        total_avg_fairness = 0.0
    print(f"\nTotal Average Fairness Score for {model_name} Model: {total_avg_fairness:.4f}")
    return total_avg_fairness # Return the total average fairness score

# --- Function to load individual models and get predictions + attributes ---
# This function is crucial and will load your *individual* audio and video models
# and generate their predictions on the validation data.
def load_individual_predictions_and_attributes(audio_model, video_model, data_dir, audio_input_shape, video_input_shape):
    print("\n--- Generating individual model predictions for post-hoc fusion ---")
    val_generator = CombinedDataGenerator(
        data_dir, audio_input_shape, video_input_shape, batch_size=1 # Use batch_size=1 for consistent data loading
    )

    if len(val_generator) == 0:
        print("Warning: Validation generator is empty. Cannot generate individual predictions.")
        return np.array([]), np.array([]), [], np.array([])

    all_audio_preds = []
    all_video_preds = []
    all_attributes = []
    all_true_labels = []

    for i in range(len(val_generator)):
        inputs, true_labels_batch, attributes_batch = val_generator[i]

        # Predict with individual audio and video models
        # Ensure that inputs['audio_input'] and inputs['video_input'] are correctly shaped
        audio_preds_batch = audio_model.predict(inputs['audio_input'], verbose=0)
        video_preds_batch = video_model.predict(inputs['video_input'], verbose=0)

        all_audio_preds.extend(audio_preds_batch)
        all_video_preds.extend(video_preds_batch)
        all_true_labels.extend(true_labels_batch)
        all_attributes.extend(attributes_batch)

    return np.array(all_audio_preds), np.array(all_video_preds), all_attributes, np.array(all_true_labels)


# --- Main Script Execution ---

if __name__ == "__main__":
    print("--- Script: Weighted Prediction Fusion Evaluation ---")

    # --- IMPORTANT: Configure your model paths for INDIVIDUAL models ---
    # These paths should point to the models you trained ONLY on audio, and ONLY on video
    audio_model_path = "saved_models/audio_only_model.keras" # Path to your trained audio-only model
    video_model_path = "saved_models/video_only_model.keras" # Path to your trained video-only model
    labeled_validation_dir = r"\validation_chunks_labeled"

    # Ensure paths exist
    if not os.path.exists(labeled_validation_dir):
        raise ValueError(f"Validation directory not found: {labeled_validation_dir}")
    if not os.path.exists(audio_model_path):
        raise FileNotFoundError(f"Audio model not found: {audio_model_path}. Please train and save it first!")
    if not os.path.exists(video_model_path):
        raise FileNotFoundError(f"Video model not found: {video_model_path}. Please train and save it first!")


    # --- Define Input Shapes ---
    AUDIO_INPUT_SHAPE = (24, 1319, 1)
    VIDEO_INPUT_SHAPE = (6, 128, 128, 3)

    # --- Load Individual Models ---
    print(f"\n--- Loading Audio Model from {audio_model_path} ---")
    try:
        # Rebuild and load audio model
        audio_only_model = create_audio_model_pretrained_efficientnet()
        audio_only_model.load_weights(audio_model_path)
        audio_only_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
        print("Audio model loaded and compiled.")
        audio_only_model.summary() # Print summary to confirm output shape
    except Exception as e:
        print(f"Error loading audio model: {e}. Attempting direct load.")
        try: # Fallback direct load
            audio_only_model = tf.keras.models.load_model(audio_model_path)
            audio_only_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
            print("Audio model loaded directly.")
            audio_only_model.summary() # Print summary to confirm output shape
        except Exception as e:
            print(f"Failed to load audio model even with direct load_model: {e}")
            exit()

    print(f"\n--- Loading Video Model from {video_model_path} ---")
    try:
        # Rebuild and load video model
        video_only_model = create_video_model_vgg16()
        video_only_model.load_weights(video_model_path)
        video_only_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
        print("Video model loaded and compiled.")
        video_only_model.summary() # Print summary to confirm output shape
    except Exception as e:
        print(f"Error loading video model: {e}. Attempting direct load.")
        try: # Fallback direct load
            video_only_model = tf.keras.models.load_model(video_model_path)
            video_only_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
            print("Video model loaded directly.")
            video_only_model.summary() # Print summary to confirm output shape
        except Exception as e:
            print(f"Failed to load video model: {e}")
            exit()

    # --- Get all individual predictions and attributes once ---
    all_audio_preds, all_video_preds, all_attributes, all_true_labels = \
        load_individual_predictions_and_attributes(audio_only_model, video_only_model,
                                                   labeled_validation_dir, AUDIO_INPUT_SHAPE, VIDEO_INPUT_SHAPE)

    if all_audio_preds.size == 0 or all_video_preds.size == 0 or all_true_labels.size == 0:
        print("No valid data loaded for evaluation. Exiting.")
        exit()

    # --- Calculate and Evaluate for Specific Weighted Combination (Audio 0.8, Video 0.2) ---
    print("\n--- Evaluating Specific Weighted Combination (Audio: 0.8, Video: 0.2) ---")
    w_audio_specific = 0.8
    w_video_specific = 0.2

    # Perform the weighted fusion of predictions
    # This assumes all_audio_preds and all_video_preds now have shape (num_samples, 5)
    specific_combined_preds = (w_audio_specific * all_audio_preds) + (w_video_specific * all_video_preds)

    # Calculate MAE for this specific combination
    specific_mae = mean_absolute_error(specific_combined_preds, all_true_labels)

    # Evaluate fairness for this specific combination
    specific_fairness_score = evaluate_fairness(
        specific_combined_preds,
        all_true_labels,
        all_attributes,
        f"Weighted Fusion (A:{w_audio_specific:.1f}, V:{w_video_specific:.1f})",
        DEFAULT_PRED_THRESHOLD,
        DEFAULT_TRUE_LABEL_THRESHOLD
    )

    print(f"\nSummary for Weighted Fusion (A:{w_audio_specific:.1f}, V:{w_video_specific:.1f}):")
    print(f"MAE: {specific_mae:.4f}")
    print(f"Fairness Score (Total Average Equal Opportunity): {specific_fairness_score:.4f}")

    print("\n--- Weighted Fusion Evaluation Complete ---")


# In[ ]:


--- Individual Model Baseline Performance ---

Fairness Evaluation for Audio Only Model
==================================================

Protected Attribute: Gender
----------------------------------------
Extraversion    | Equal Opportunity: 0.0046 | Statistical Parity: 0.0016
Neuroticism     | Equal Opportunity: 0.0000 | Statistical Parity: 0.0000
Agreeableness   | Equal Opportunity: 0.0024 | Statistical Parity: 0.0003
Conscientiousness | Equal Opportunity: 0.0000 | Statistical Parity: 0.0000
Openness        | Equal Opportunity: 0.0000 | Statistical Parity: 0.0000

Average Equal Opportunity for Gender: 0.0014

Protected Attribute: Ethnicity
----------------------------------------
Extraversion    | Equal Opportunity: 0.0018 | Statistical Parity: 0.0008
Neuroticism     | Equal Opportunity: 0.0000 | Statistical Parity: 0.0000
Agreeableness   | Equal Opportunity: 0.0012 | Statistical Parity: 0.0057
Conscientiousness | Equal Opportunity: 0.0000 | Statistical Parity: 0.0000
Openness        | Equal Opportunity: 0.0000 | Statistical Parity: 0.0000

Average Equal Opportunity for Ethnicity: 0.0006

Total Average Fairness Score for Audio Only Model: 0.0010
Audio Only Model MAE: 0.1206
Fairness Evaluation for Video Only Model
==================================================

Protected Attribute: Gender
----------------------------------------
Extraversion    | Equal Opportunity: 0.4766 | Statistical Parity: 0.3995
Neuroticism     | Equal Opportunity: 0.2058 | Statistical Parity: 0.2014
Agreeableness   | Equal Opportunity: 0.0535 | Statistical Parity: 0.0571
Conscientiousness | Equal Opportunity: 0.1175 | Statistical Parity: 0.1501
Openness        | Equal Opportunity: 0.0931 | Statistical Parity: 0.1271

Average Equal Opportunity for Gender: 0.1893

Protected Attribute: Ethnicity
----------------------------------------
Extraversion    | Equal Opportunity: 0.2715 | Statistical Parity: 0.2938
Neuroticism     | Equal Opportunity: 0.2111 | Statistical Parity: 0.2697
Agreeableness   | Equal Opportunity: 0.1053 | Statistical Parity: 0.0990
Conscientiousness | Equal Opportunity: 0.1900 | Statistical Parity: 0.2135
Openness        | Equal Opportunity: 0.0475 | Statistical Parity: 0.0668

Average Equal Opportunity for Ethnicity: 0.1651

Total Average Fairness Score for Video Only Model: 0.1772
Video Only Model MAE: 0.1074
---------------------------------------------
Weights (A:0.00, V:1.00) | MAE: 0.1074 | Fairness Score: 0.1772
Weights (A:0.05, V:0.95) | MAE: 0.1073 | Fairness Score: 0.1652
Weights (A:0.10, V:0.90) | MAE: 0.1073 | Fairness Score: 0.1547
Weights (A:0.15, V:0.85) | MAE: 0.1073 | Fairness Score: 0.1421
Weights (A:0.20, V:0.80) | MAE: 0.1075 | Fairness Score: 0.1265
Weights (A:0.25, V:0.75) | MAE: 0.1077 | Fairness Score: 0.1172
Weights (A:0.30, V:0.70) | MAE: 0.1080 | Fairness Score: 0.1007
Weights (A:0.35, V:0.65) | MAE: 0.1083 | Fairness Score: 0.0781
Weights (A:0.40, V:0.60) | MAE: 0.1088 | Fairness Score: 0.0633
Weights (A:0.45, V:0.55) | MAE: 0.1094 | Fairness Score: 0.0543
Weights (A:0.50, V:0.50) | MAE: 0.1100 | Fairness Score: 0.0495
Weights (A:0.55, V:0.45) | MAE: 0.1108 | Fairness Score: 0.0259
Weights (A:0.60, V:0.40) | MAE: 0.1116 | Fairness Score: 0.0215
Weights (A:0.65, V:0.35) | MAE: 0.1125 | Fairness Score: 0.0164
Weights (A:0.70, V:0.30) | MAE: 0.1134 | Fairness Score: 0.0112
Weights (A:0.75, V:0.25) | MAE: 0.1144 | Fairness Score: 0.0064
Weights (A:0.80, V:0.20) | MAE: 0.1155 | Fairness Score: 0.0021
Weights (A:0.85, V:0.15) | MAE: 0.1167 | Fairness Score: 0.0006
Weights (A:0.90, V:0.10) | MAE: 0.1179 | Fairness Score: 0.0006
Weights (A:0.95, V:0.05) | MAE: 0.1192 | Fairness Score: 0.0010
Weights (A:1.00, V:0.00) | MAE: 0.1206 | Fairness Score: 0.0010

--- Plotting Fairness vs. Accuracy Trade-off ---


--- Weighted Fusion Evaluation Complete ---

