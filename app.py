import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from scipy.stats import chisquare
import joblib  # For SPAM classifier loading
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import pandas as pd # Added for CSV report

# Initialize lists to store data for reports globally, cleared per batch
standard_report_entries = []
detailed_report_entries = []

# Load models (outside the loop to load only once)
# Wrap in try-except for robustness during startup
try:
    cnn_model = load_model("cnn_model_real_stegoappdb.h5", compile=False)
    spam_model = joblib.load("spam_rf_model.pkl")  # Ensure you have this model file
except Exception as e:
    st.error(f"🚨 Critical Error: Could not load machine learning models: {e}")
    st.error("Please ensure 'cnn_model_real_stegoappdb.h5' and 'spam_rf_model.pkl' are in the correct path.")
    st.stop() # Stop execution if models can't be loaded

st.set_page_config(page_title="StegoAppDB Classifier", layout="centered")
st.title("🕵️‍♂️ Steganography Classifier (Multi-Image Analysis)")

# === Helper functions (Mostly from the provided code with minor improvements) ===

# === CNN Prediction ===
def cnn_predict(gray_img):
    if gray_img is None or gray_img.size == 0:
        st.warning("CNN: Input image is empty.")
        return np.array([1/3, 1/3, 1/3]) # Default neutral prediction
    resized = cv2.resize(gray_img, (64, 64))
    input_tensor = resized.reshape(1, 64, 64, 1).astype(np.float32) / 255.0
    try:
        prediction = cnn_model.predict(input_tensor)[0]
    except Exception as e:
        st.warning(f"CNN prediction failed: {e}")
        return np.array([1/3, 1/3, 1/3])
    return prediction

# === LSB Extraction and Chi-Square Test ===
def extract_lsb(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        st.warning("LSB: Input image is empty for LSB extraction.")
        return np.array([]), np.array([]) # Return empty arrays
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lsb_plane = np.bitwise_and(gray, 1)
    return lsb_plane, gray

def chi_square_test(lsb_plane):
    if lsb_plane is None or lsb_plane.size == 0:
        return 0.0, 0, 0 # Statistic, zeros, ones
    flat = lsb_plane.flatten()
    zeros = np.count_nonzero(flat == 0)
    ones = np.count_nonzero(flat == 1)
    total = zeros + ones
    if total == 0: return 0.0, zeros, ones
    expected = total / 2.0
    if expected == 0: return 0.0, zeros, ones
    
    # Chisquare requires observed frequencies > 0 if expected > 0 for stable results.
    # However, for [N, 0] vs [N/2, N/2] it's usually fine.
    # Scipy's chisquare handles zeros in observed if f_exp is provided.
    try:
        stat, _ = chisquare([zeros, ones], f_exp=[expected, expected])
    except ValueError as e: # Can happen if total is small, or other array issues
        # st.warning(f"Chi-square calculation issue: {e}. Zeros: {zeros}, Ones: {ones}, Expected: {expected}")
        # Fallback: if one count is zero and other is total, it's max difference
        if (zeros == 0 and ones == total) or (ones == 0 and zeros == total):
            stat = total # Max possible chi-square like value for 2 bins is N (if expected is N/2, N/2)
        else:
            stat = 0.0
    return stat, zeros, ones

# === Histogram Analysis ===
def histogram_analysis(gray_img):
    if gray_img is None or gray_img.size == 0: return 0.0
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    if np.sum(hist) == 0: return 0.0
    hist = hist / np.sum(hist)
    uniform = np.ones_like(hist) / len(hist)
    diff = np.sum(np.abs(hist - uniform))
    return diff

# === Noise Residual Analysis ===
def noise_residual(gray_img):
    if gray_img is None or gray_img.size == 0: return 0.0
    median_filtered = cv2.medianBlur(gray_img, 3) # Kernel size 3
    residual = cv2.absdiff(gray_img, median_filtered)
    noise_score = np.mean(residual) / 255.0 if residual.size > 0 else 0.0
    return noise_score

# === RS Steganalysis (Simplified Proxy - current implementation is same as noise_residual) ===
def rs_steganalysis(gray_img):
    # For a true RS Steganalysis, this function would need a more complex implementation.
    # Currently, it mirrors noise_residual for simplicity as in the original code.
    if gray_img is None or gray_img.size == 0: return 0.0
    median_filtered = cv2.medianBlur(gray_img, 3)
    residual = cv2.absdiff(gray_img, median_filtered)
    score = np.mean(residual) / 255.0 if residual.size > 0 else 0.0
    return score

# --- SPAM Feature Extraction helpers ---
def quantize_residuals(residuals, T=3):
    residuals = np.clip(residuals, -T, T) # Clip to [-T, T]
    residuals = residuals.astype(int)
    residuals += T  # Shift to [0, 2T] range for indexing
    return residuals

def compute_transitions(residuals_quantized_1d):
    size = residuals_quantized_1d.shape[0]
    T_val = 3 # Quantization threshold T used in quantize_residuals
    matrix_dim = 2 * T_val + 1 # This will be 7 for T=3
    
    transition_matrix = np.zeros((matrix_dim, matrix_dim), dtype=np.float32)

    if size < 2: # Need at least two elements for a transition
        return transition_matrix.flatten()

    for i in range(size - 1):
        r1 = residuals_quantized_1d[i]
        r2 = residuals_quantized_1d[i+1]
        # Ensure r1 and r2 are within expected range [0, 2T]
        if 0 <= r1 < matrix_dim and 0 <= r2 < matrix_dim:
            transition_matrix[r1, r2] += 1
        # else:
            # This case should ideally not happen if quantization is correct
            # st.warning(f"SPAM: Residuals out of bound: {r1}, {r2} for dim {matrix_dim}")


    total_transitions = np.sum(transition_matrix)
    if total_transitions > 0:
        transition_matrix /= total_transitions
    return transition_matrix.flatten()


def spam_features(image_gray):
    if image_gray is None or image_gray.size < 4: # Need at least 2x2 for some diffs
        return np.zeros(2044, dtype=np.float32) # Return default if image too small

    T = 3  # quantization threshold
    img_int = image_gray.astype(np.int32)
    
    all_transition_features = []

    # Horizontal differences and transitions
    if img_int.shape[1] > 1:
        res_h = img_int[:, 1:] - img_int[:, :-1]
        res_h_q = quantize_residuals(res_h, T)
        for i in range(res_h_q.shape[0]):
            all_transition_features.extend(compute_transitions(res_h_q[i, :]))

    # Vertical differences and transitions
    if img_int.shape[0] > 1:
        res_v = img_int[1:, :] - img_int[:-1, :]
        res_v_q = quantize_residuals(res_v, T)
        for i in range(res_v_q.shape[1]): # Iterate over columns
            all_transition_features.extend(compute_transitions(res_v_q[:, i]))

    # Diagonal (main diagonal) differences and transitions
    if img_int.shape[0] > 1 and img_int.shape[1] > 1:
        res_d1 = img_int[1:, 1:] - img_int[:-1, :-1]
        res_d1_q = quantize_residuals(res_d1, T)
        # This needs careful iteration for diagonals, or process row-wise/col-wise on the residual map
        for i in range(res_d1_q.shape[0]): # Simpler: process rows of residual map
             all_transition_features.extend(compute_transitions(res_d1_q[i,:]))


    # Anti-diagonal differences and transitions
    if img_int.shape[0] > 1 and img_int.shape[1] > 1:
        res_d2 = img_int[1:, :-1] - img_int[:-1, 1:]
        res_d2_q = quantize_residuals(res_d2, T)
        for i in range(res_d2_q.shape[0]): # Simpler: process rows of residual map
             all_transition_features.extend(compute_transitions(res_d2_q[i,:]))

    features_concatenated = np.array(all_transition_features, dtype=np.float32)
    
    target_len = 2044
    current_len = features_concatenated.size

    if current_len == 0 and target_len > 0 : # No features extracted, pad with zeros
        return np.zeros(target_len, dtype=np.float32)
    if current_len > target_len:
        final_features = features_concatenated[:target_len]
    elif current_len < target_len:
        final_features = np.pad(features_concatenated, (0, target_len - current_len), 'constant', constant_values=0)
    else:
        final_features = features_concatenated
        
    return final_features


# === SPAM Prediction ===
def spam_predict(gray_img):
    if gray_img is None or gray_img.size == 0:
        st.warning("SPAM: Input image is empty.")
        return np.array([1/3, 1/3, 1/3])

    features = spam_features(gray_img)
    features = features.reshape(1, -1) # Reshape for the model
    
    if features.shape[1] != 2044: # Double check feature length
        st.warning(f"SPAM: Feature length mismatch. Expected 2044, got {features.shape[1]}. Using neutral prediction.")
        return np.array([1/3, 1/3, 1/3])
        
    try:
        proba = spam_model.predict_proba(features)[0]
    except Exception as e:
        st.warning(f"SPAM prediction failed: {e}. Using neutral prediction.")
        proba = np.array([1/3, 1/3, 1/3]) # Default probabilities on error
    return proba # Expected: [Cover_prob, Stego_prob, Original_prob]

# === Scoring Helpers for Ensemble (heuristic mapping to [Cover, Stego, Original] belief) ===
# These functions return a 3-element array representing belief weights for [Cover, Stego, Original]
# where the "Stego" component is primarily affected by the input metric.

def lsb_chi_score_for_ensemble(chi_val):
    # High chi suggests steganography. Scale score between 0 and 1 for "stego-ness".
    # 3.84 is a common threshold for significance (p=0.05, 1 DoF).
    # Let's say chi > 20 is strong evidence.
    stego_likelihood = min(max((chi_val - 3.84) / (50 - 3.84), 0), 1) # Normalized stego evidence
    # If stego_likelihood is high, lower belief in Cover/Original.
    # This is a simple heuristic distribution.
    return np.array([(1 - stego_likelihood) * 0.4, stego_likelihood, (1 - stego_likelihood) * 0.6])


def histogram_score_for_ensemble(hist_diff_val):
    # High difference from uniform may indicate stego.
    # Assume diff > 0.15 is significant. Max diff can be ~2.0.
    stego_likelihood = min(max(hist_diff_val / 0.20, 0), 1) # Normalize to 0-1
    return np.array([(1 - stego_likelihood) * 0.4, stego_likelihood, (1 - stego_likelihood) * 0.6])

def noise_score_for_ensemble(noise_val):
    # Elevated noise might indicate stego. Assume score > 0.1 is significant. Max practical is ~0.2-0.3
    stego_likelihood = min(max(noise_val / 0.15, 0), 1)
    return np.array([(1 - stego_likelihood) * 0.4, stego_likelihood, (1 - stego_likelihood) * 0.6])

def rs_score_for_ensemble(rs_metric_val):
    # Similar to noise for this implementation.
    stego_likelihood = min(max(rs_metric_val / 0.15, 0), 1)
    return np.array([(1 - stego_likelihood) * 0.4, stego_likelihood, (1 - stego_likelihood) * 0.6])

# SPAM probability is already in [Cover, Stego, Original] format from the model
def spam_score_for_ensemble(spam_probabilities):
    return spam_probabilities


# === Other Analysis Functions ===
def compression_diff(image_bgr): # Expects BGR image from cv2
    if image_bgr is None or image_bgr.size == 0: return 0.0
    temp_dir = tempfile.gettempdir()
    orig_tmp_path = os.path.join(temp_dir, next(tempfile._get_candidate_names()) + ".jpg")
    low_q_tmp_path = os.path.join(temp_dir, next(tempfile._get_candidate_names()) + ".jpg")

    try:
        cv2.imwrite(orig_tmp_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        orig_size = os.path.getsize(orig_tmp_path) if os.path.exists(orig_tmp_path) else 0
        
        cv2.imwrite(low_q_tmp_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 40])
        low_size = os.path.getsize(low_q_tmp_path) if os.path.exists(low_q_tmp_path) else 0
    finally:
        if os.path.exists(orig_tmp_path): os.remove(orig_tmp_path)
        if os.path.exists(low_q_tmp_path): os.remove(low_q_tmp_path)

    if orig_size == 0: return 0.0
    diff_ratio = (orig_size - low_size) / orig_size
    return diff_ratio

def fourier_analysis(gray_img):
    if gray_img is None or gray_img.shape[0] < 2 or gray_img.shape[1] < 2: return 0.0 # Need min 2x2 for FFT
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8) # Epsilon for log safety
    
    # Consider a relative measure of high-frequency energy
    rows, cols = gray_img.shape
    center_row, center_col = rows // 2, cols // 2
    # Example: high frequencies are outside a central radius (e.g., 1/4 of dimensions)
    radius_ratio = 0.25 
    mask = np.ones(magnitude_spectrum.shape, dtype=bool)
    r_max, c_max = int(center_row * radius_ratio), int(center_col * radius_ratio)
    mask[center_row-r_max:center_row+r_max, center_col-c_max:center_col+c_max] = False
    
    high_freq_values = magnitude_spectrum[mask]
    
    if high_freq_values.size == 0: return 0.0
    
    # Normalize by overall magnitude average or a fixed scale
    mean_total_magnitude = np.mean(magnitude_spectrum)
    if mean_total_magnitude < 1e-6 : return 0.0 # Avoid division by zero if image is flat

    freq_score = np.mean(high_freq_values) / mean_total_magnitude
    # Clip and scale score to be somewhat intuitive (e.g. 0 to 1 or more)
    return min(max(freq_score / 5.0, 0), 1) # Heuristic scaling


def patch_based_noise(gray_img, patch_size=64):
    if gray_img is None or gray_img.size == 0: return 0.0, 0.0
    h, w = gray_img.shape
    patches_noise_scores = []

    if h < patch_size or w < patch_size : # If image is smaller than patch size
        # Analyze the whole image as one "patch"
        base_noise = noise_residual(gray_img)
        return base_noise, 0.0 # Mean is the noise, std dev is 0 as only one sample

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = gray_img[y:y+patch_size, x:x+patch_size]
            if patch.size > 0: # Ensure patch is not empty
                noise = noise_residual(patch)
                patches_noise_scores.append(noise)
    
    if not patches_noise_scores: # If no patches were processed (e.g., image smaller than patch_size)
        base_noise = noise_residual(gray_img) # Should have been caught above
        return base_noise, 0.0

    return np.mean(patches_noise_scores), np.std(patches_noise_scores)

def metadata_analysis(uploaded_file_obj):
    # Ensure stream is at the beginning
    uploaded_file_obj.seek(0)
    format_from_pil = "Unknown"
    has_exif = False
    metadata_info = {}
    
    try:
        image_pil = Image.open(uploaded_file_obj)
        format_from_pil = image_pil.format
        # EXIF data can be in .info or .getexif()
        info_dict = image_pil.info
        exif_data = image_pil.getexif()
        has_exif = ("exif" in info_dict and info_dict["exif"]) or (exif_data and len(exif_data) > 0)
        metadata_info = {**info_dict, **exif_data} # Combine them, exif_data is more structured
    except Exception as e:
        # st.warning(f"Metadata: Could not read image with PIL: {e}")
        pass # Soft fail

    # Get extension from filename as fallback or primary
    filename_ext = os.path.splitext(uploaded_file_obj.name)[1].lower()
    
    # Prefer PIL format if available and seems valid, else use file extension.
    effective_format = format_from_pil if format_from_pil and format_from_pil != "Unknown" else filename_ext
    
    return effective_format, has_exif, metadata_info # metadata_info can be large


def estimate_lsb_payload_size(cv_image_bgr, bits_used_per_channel_pixel=1):
    if cv_image_bgr is None or cv_image_bgr.size == 0: return 0
    h, w = cv_image_bgr.shape[:2]
    channels = 1 if len(cv_image_bgr.shape) == 2 else cv_image_bgr.shape[2] # Grayscale or Color
    
    total_pixels = h * w
    total_bits_embeddable = total_pixels * channels * bits_used_per_channel_pixel
    max_bytes_embeddable = total_bits_embeddable // 8
    return max_bytes_embeddable

def estimate_payload_from_chi(chi_square_val, max_lsb_capacity_bytes):
    # This is a VERY ROUGH HEURISTIC and should not be taken as accurate.
    # It assumes payload percentage might correlate with chi-square value.
    
    # Chi-square critical value for p=0.05, 1 DoF is ~3.84.
    # Values below this (or slightly above due to noise) suggest no significant deviation.
    if chi_square_val < 10: # If chi-square is low, assume no or negligible payload
        return 0
    
    # Assume a non-linear scaling. Chi-square can grow very large with high payload.
    # Let's cap the "effective" chi-square for payload estimation to avoid extreme values.
    # Max observed chi can be image_size if it's perfectly 000... vs 111...
    # This range (10 to, say, 1000 for "full" LSB effect) is arbitrary.
    
    effective_chi = min(chi_square_val, 1000) # Cap effective chi at 1000 for this heuristic
    
    # Scale payload from 0% (at chi=10) to ~75% (at chi=1000) of max LSB capacity.
    # (Stego rarely uses 100% LSB capacity perfectly due to overhead or strategy)
    payload_ratio = ((effective_chi - 10) / (1000 - 10)) * 0.75 
    payload_ratio = max(0, min(payload_ratio, 0.75)) # Ensure ratio is within [0, 0.75]
        
    estimated_bytes = int(max_lsb_capacity_bytes * payload_ratio)
    return estimated_bytes


# === Weighted Ensemble ===
def weighted_ensemble_prediction(cnn_probs, chi_val, hist_difference, noise_metric, spam_probs, rs_metric, comp_diff_ratio, fourier_score, patch_noise_std, has_exif, estimated_payload_bytes):
    # Weights for each classifier/metric's contribution (NEW WEIGHTS)
    weights = {
        'cnn': 0.30,
        'spam': 0.20,
        'chi': 0.10,
        'hist': 0.05,
        'noise': 0.05,
        'rs': 0.03,
        'compression': 0.07,
        'fourier': 0.07,
        'patch_noise': 0.08,
        'metadata': 0.03,
        'payload_est': 0.02
    }

    # Get scores from helper functions, ensuring they are [Cover, Stego, Original] vectors
    cnn_s = cnn_probs # Already in correct format
    spam_s = spam_score_for_ensemble(spam_probs)
    chi_s = lsb_chi_score_for_ensemble(chi_val)
    hist_s = histogram_score_for_ensemble(hist_difference)
    noise_s = noise_score_for_ensemble(noise_metric)
    rs_s = rs_score_for_ensemble(rs_metric)

    # Heuristic scores for the new methods (need to map their output to [Cover, Stego, Original])
    # These are basic examples; you might need more sophisticated mappings based on typical ranges
    compression_s = np.array([1 - min(max(comp_diff_ratio * 5, 0), 1), min(max(comp_diff_ratio * 5, 0), 1), 0.5 - min(max(comp_diff_ratio * 2.5, 0), 0.5)]) # Low diff -> Stego
    fourier_s = np.array([1 - min(max(fourier_score * 3, 0), 1), min(max(fourier_score * 3, 0), 1), 0.5 - min(max(fourier_score * 1.5, 0), 0.5)]) # High score -> Stego
    patch_noise_s = np.array([1 - min(max(patch_noise_std * 100, 0), 1), min(max(patch_noise_std * 100, 0), 1), 0.5 - min(max(patch_noise_std * 50, 0), 0.5)]) # High std -> Stego
    metadata_s = np.array([0.7 if has_exif else 0.3, 0.3 if has_exif else 0.7, 0.5]) # Missing EXIF -> Slightly more Stego
    payload_est_s = np.array([1 - min(max(estimated_payload_bytes / 100000.0, 0), 1), min(max(estimated_payload_bytes / 100000.0, 0), 1), 0.5 - min(max(estimated_payload_bytes / 50000.0, 0), 0.5)]) # Larger payload -> More Stego

    # Weighted sum of probabilities/scores
    combined_score_vector = (cnn_s * weights['cnn'] +
                             spam_s * weights['spam'] +
                             chi_s * weights['chi'] +
                             hist_s * weights['hist'] +
                             noise_s * weights['noise'] +
                             rs_s * weights['rs'] +
                             compression_s * weights['compression'] +
                             fourier_s * weights['fourier'] +
                             patch_noise_s * weights['patch_noise'] +
                             metadata_s * weights['metadata'] +
                             payload_est_s * weights['payload_est'])

    # Normalize the combined vector
    total_weight_sum = sum(weights.values())
    if total_weight_sum > 0:
        final_probabilities = combined_score_vector / total_weight_sum
    else:
        final_probabilities = np.array([1/3, 1/3, 1/3])

    return final_probabilities # Returns [Cover_prob, Stego_prob, Original_prob]


# === Streamlit UI & Main Processing Logic ===
# Use st.file_uploader with accept_multiple_files=True
uploaded_files = st.file_uploader("Upload one or more PNG/JPG/JPEG images",
                                  type=["png", "jpg", "jpeg"],
                                  accept_multiple_files=True,
                                  help="Analyzes images for signs of steganography.")

if uploaded_files:
    # Initialize/clear report lists for this batch of uploads
    standard_report_entries.clear()
    detailed_report_entries.clear()

    for uploaded_file_obj in uploaded_files:
        st.markdown("---")
        st.header(f"🔎 Analysis for: {uploaded_file_obj.name}")
        
        # Read image bytes first
        image_bytes = uploaded_file_obj.getvalue() # Reads all bytes into memory
        file_bytes_np_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        
        # Decode image using OpenCV
        image_cv_bgr = cv2.imdecode(file_bytes_np_array, cv2.IMREAD_COLOR)

        if image_cv_bgr is None:
            st.error(f"❌ Could not decode image: {uploaded_file_obj.name}. It might be corrupted or an unsupported format variation. Skipping.")
            # Add a basic entry for detailed report indicating failure
            detailed_report_entries.append({
                "filename": uploaded_file_obj.name,
                "error_message": "Failed to decode image",
            })
            continue # Skip to the next file

        # Display uploaded image
        st.image(cv2.cvtColor(image_cv_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        # --- Start of analysis for a single image ---
        lsb_plane_data, gray_image_data = extract_lsb(image_cv_bgr)

        # LSB Analysis
        with st.expander("🧪 LSB Analysis", expanded=False):
            if lsb_plane_data.size > 0 :
                st.image(lsb_plane_data * 255, caption="LSB Plane (Black=0, White=1)", use_container_width=True, channels="GRAY")
                chi_sq_stat, lsb_zeros, lsb_ones = chi_square_test(lsb_plane_data)
                lsb_total = lsb_zeros + lsb_ones
                lsb_ratio_ones = lsb_ones / lsb_total if lsb_total > 0 else 0.0
                
                st.markdown(f"- **Count of 0s in LSB:** {lsb_zeros}")
                st.markdown(f"- **Count of 1s in LSB:** {lsb_ones}")
                st.markdown(f"- **Ratio of 1s to Total:** `{lsb_ratio_ones:.4f}`")
                st.markdown(f"- **Chi-Square Statistic:** `{chi_sq_stat:.4f}` (Higher values may indicate non-randomness)")
                
                if lsb_ratio_ones < 0.47 or lsb_ratio_ones > 0.53 or chi_sq_stat > 10: # Adjusted thresholds
                    st.warning("⚠️ Potentially suspicious LSB pattern detected (non-random distribution).")
                else:
                    st.success("✅ LSB distribution appears normal/random.")
            else:
                st.info("LSB analysis could not be performed (e.g., image read issue).")
                chi_sq_stat, lsb_zeros, lsb_ones, lsb_ratio_ones = 0.0, 0, 0, 0.0


        # Histogram Analysis
        hist_diff_metric = histogram_analysis(gray_image_data)
        with st.expander("📊 Histogram Analysis", expanded=False):
            st.markdown(f"- **Histogram Difference from Uniform Distribution:** `{hist_diff_metric:.4f}`")
            if hist_diff_metric > 0.18: # Slightly adjusted threshold
                st.warning("⚠️ Histogram shows some deviation from expected uniformity (potential steganographic artifacts).")
            else:
                st.success("✅ Histogram profile looks statistically uniform.")

        # Noise Residual Analysis
        noise_residual_metric = noise_residual(gray_image_data)
        with st.expander("📉 Noise Residual Analysis", expanded=False):
            st.markdown(f"- **Mean Noise Score (0-1):** `{noise_residual_metric:.4f}`")
            if noise_residual_metric > 0.08: # Adjusted threshold
                st.warning("⚠️ Elevated noise level detected (might indicate embedded data or specific image processing).")
            else:
                st.success("✅ Noise pattern appears within typical bounds.")

        # RS Steganalysis (Simplified Proxy)
        rs_metric_value = rs_steganalysis(gray_image_data) # Note: currently same as noise_residual
        with st.expander("📉 RS Steganalysis (Simplified Proxy)", expanded=False):
            st.markdown(f"- **RS Score (Proxy):** `{rs_metric_value:.4f}`")
            if rs_metric_value > 0.08:
                st.warning("⚠️ Simplified RS analysis suggests characteristics that could align with steganography.")
            else:
                st.success("✅ Simplified RS analysis appears normal.")
        
        # --- Machine Learning Model Predictions ---
        st.subheader("🧠 ML Model Predictions")

        # SPAM Feature-Based Classifier
        spam_probabilities = spam_predict(gray_image_data) # Expected: [Cover, Stego, Original]
        spam_class_labels = ["Cover", "Stego", "Original"]
        spam_predicted_idx = np.argmax(spam_probabilities)
        spam_predicted_label = spam_class_labels[spam_predicted_idx]
        st.markdown(f"**SPAM Classifier:** Predicts `{spam_predicted_label}` (Cover: `{spam_probabilities[0]:.3f}`, Stego: `{spam_probabilities[1]:.3f}`, Original: `{spam_probabilities[2]:.3f}`)")
        if spam_predicted_idx == 1 and spam_probabilities[1] > 0.65:
             st.warning(f"⚠️ SPAM model indicates potential *Stego* characteristics.")
        
        # CNN Classification
        cnn_probabilities = cnn_predict(gray_image_data) # Expected: [Cover, Stego, Original]
        cnn_class_labels = ["Cover", "Stego", "Original"] # Ensure this order matches model output
        cnn_predicted_idx = np.argmax(cnn_probabilities)
        cnn_predicted_label = cnn_class_labels[cnn_predicted_idx]
        st.markdown(f"**CNN Classifier:** Predicts `{cnn_predicted_label}` (Cover: `{cnn_probabilities[0]:.3f}`, Stego: `{cnn_probabilities[1]:.3f}`, Original: `{cnn_probabilities[2]:.3f}`)")
        if cnn_predicted_idx == 1 and cnn_probabilities[1] > 0.65:
            st.warning(f"⚠️ CNN model indicates potential *Stego* characteristics.")


        # --- Additional Analyses ---
        st.subheader("🔬 Additional Heuristic Analyses")
        
        comp_diff_ratio_metric = compression_diff(image_cv_bgr)
        with st.expander("🗜️ Compression Difference Test (JPEG Heuristic)", expanded=False):
            st.markdown(f"- **JPEG Re-compression Reduction Ratio:** `{comp_diff_ratio_metric:.4f}`")
            if comp_diff_ratio_metric < 0.15 and image_cv_bgr.shape[0]*image_cv_bgr.shape[1] > 10000: # Apply only if not tiny image
                st.warning("⚠️ Low difference in file size after re-compression (might suggest incompressible embedded data).")
            else:
                st.success("✅ Normal JPEG re-compression behavior observed or test less relevant.")

        fourier_score_metric = fourier_analysis(gray_image_data)
        with st.expander("📈 Fourier Transform Analysis (High-Frequency Check)", expanded=False):
            st.markdown(f"- **Relative High-Frequency Energy Score:** `{fourier_score_metric:.4f}`")
            if fourier_score_metric > 0.30: # Adjusted threshold
                st.warning("⚠️ Anomalous high-frequency components detected in Fourier spectrum.")
            else:
                st.success("✅ Frequency distribution appears typical.")

        patch_mean_noise, patch_std_dev_noise = patch_based_noise(gray_image_data)
        with st.expander("🧩 Patch-Based Noise Consistency", expanded=False):
            st.markdown(f"- **Mean Noise Across Patches:** `{patch_mean_noise:.4f}`")
            st.markdown(f"- **Std. Deviation of Noise Across Patches:** `{patch_std_dev_noise:.4f}`")
            if patch_std_dev_noise > 0.015: # Adjusted threshold
                st.warning("⚠️ Noise levels vary significantly across image patches (may indicate localized hidden data).")
            else:
                st.success("✅ Noise pattern appears relatively consistent across the image.")

        # Metadata Analysis - Pass the uploaded_file_obj
        effective_file_format, has_exif_data, _ = metadata_analysis(uploaded_file_obj) # Full metadata_info not displayed to save space
        with st.expander("📂 File Metadata Check", expanded=False):
            st.markdown(f"- **Detected File Format (PIL/Extension):** `{effective_file_format}`")
            st.markdown(f"- **Contains EXIF Data:** `{'Yes' if has_exif_data else 'No'}`")
            if not has_exif_data and effective_file_format.lower() in ['.jpeg', '.jpg', '.tiff']:
                st.warning("⚠️ EXIF data is missing from this JPEG/TIFF file (could be stripped).")
            elif effective_file_format.lower() == '.png' and has_exif_data:
                 st.info("ℹ️ PNG contains EXIF-like chunks (less common but possible).")
            else:
                st.success("✅ Metadata status appears typical for the format.")

        # Payload Estimation
        max_theoretical_lsb_payload = estimate_lsb_payload_size(image_cv_bgr, bits_used_per_channel_pixel=1)
        estimated_payload_via_chi = estimate_payload_from_chi(chi_sq_stat, max_theoretical_lsb_payload)
        with st.expander("📦 Estimated Hidden Data Size (Heuristics)", expanded=False):
            st.markdown(f"- **Max Theoretical LSB Payload (1-bit/pixel/channel):** `{max_theoretical_lsb_payload:,} bytes` (~`{max_theoretical_lsb_payload/1024:.2f} KB`)")
            st.markdown(f"- **Estimated Embedded Data (via LSB Chi-Sq heuristic):** `{estimated_payload_via_chi:,} bytes` (~`{estimated_payload_via_chi/1024:.2f} KB`)")
            st.caption("Note: Payload estimation from Chi-Square is a very rough heuristic and may not be accurate.")

        # --- Ensemble Prediction ---
        st.subheader("🎯 Ensemble Classification Result")
        ensemble_probs = weighted_ensemble_prediction(
                    cnn_probabilities,
                    chi_sq_stat,
                    hist_diff_metric,
                    noise_residual_metric,
                    spam_probabilities,
                    rs_metric_value,
                    comp_diff_ratio_metric,
                    fourier_score_metric,
                    patch_std_dev_noise,
                    has_exif_data,
                    estimated_payload_via_chi
                )
        ensemble_class_labels = {0: "Cover Image", 1: "Stego Image", 2: "Original Image"}
        ensemble_predicted_idx = int(np.argmax(ensemble_probs))
        ensemble_confidence_score = float(np.max(ensemble_probs))
        ensemble_predicted_class_label = ensemble_class_labels[ensemble_predicted_idx]

        st.markdown(f"#### Final Verdict: **{ensemble_predicted_class_label}**")
        st.markdown(f"Confidence: **`{ensemble_confidence_score * 100:.2f}%`**")
        #st.markdown(f"**Confidence (Cover):** {ensemble_probs[0]:.2%}")
        #st.markdown(f"**Confidence (Stego):** {ensemble_probs[1]:.2%}")
        #st.markdown(f"**Confidence (Original):** {ensemble_probs[2]:.2%}")

        if ensemble_predicted_idx == 1: # Stego
            if ensemble_confidence_score > 0.80:
                st.error(f"🚨 Conclusion: Strong indication of **Stego Image**.")
            else:
                # Determine if Cover or Original is more likely
                if ensemble_probs[0] > ensemble_probs[2]:
                    st.success(f"✅ Conclusion: Likely a **Cover Image** (processed, no hidden data detected).")
                else:
                    st.success(f"✅ Conclusion: Likely an **Original Image** (untouched).")
            
        elif ensemble_predicted_idx == 0: # Cover
             st.success(f"✅ Conclusion: Likely a **Cover Image** (processed, no hidden data detected).")
        else: # Original
             st.success(f"✅ Conclusion: Likely an **Original Image** (untouched).")
        
        st.markdown(f"**Note: Consider image as Stego only when confidence is greater than 80%**")
        
        # Confidence Breakdown Plot
        try:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.barplot(x=list(ensemble_class_labels.values()), y=ensemble_probs, ax=ax, palette=["green", "red", "blue"])
            ax.set_ylabel("Confidence Score")
            ax.set_ylim(0, 1)
            ax.set_title("Ensemble Prediction Confidence")
            st.pyplot(fig)
            plt.close(fig) # Important to close plot
        except Exception as e:
            st.warning(f"Could not generate confidence plot: {e}")

        # --- Populate data for summary reports ---
        is_classified_as_stego_by_ensemble = (ensemble_predicted_idx == 1)
        simple_classification_for_report = "Stego" if is_classified_as_stego_by_ensemble else "Clean"

        if is_classified_as_stego_by_ensemble:
            standard_report_entries.append({
                "filename": uploaded_file_obj.name,
                "estimated_payload_bytes": estimated_payload_via_chi,
                "ensemble_confidence_stego": ensemble_probs[1] # Store stego-specific confidence
            })
        
        # Detailed report entry for every file
        detailed_report_entries.append({
            "filename": uploaded_file_obj.name,
            "final_classification_simple": simple_classification_for_report,
            "ensemble_predicted_class_verbose": ensemble_predicted_class_label,
            "ensemble_confidence_overall": f"{ensemble_confidence_score:.4f}",
            "ensemble_prob_cover": f"{ensemble_probs[0]:.4f}",
            "ensemble_prob_stego": f"{ensemble_probs[1]:.4f}",
            "ensemble_prob_original": f"{ensemble_probs[2]:.4f}",
            "estimated_payload_bytes_chi_heuristic": estimated_payload_via_chi,
            "max_theoretical_lsb_payload_bytes": max_theoretical_lsb_payload,
            "cnn_predicted_class": cnn_predicted_label,
            "cnn_prob_cover": f"{cnn_probabilities[0]:.4f}",
            "cnn_prob_stego": f"{cnn_probabilities[1]:.4f}",
            "cnn_prob_original": f"{cnn_probabilities[2]:.4f}",
            "spam_predicted_class": spam_predicted_label,
            "spam_prob_cover": f"{spam_probabilities[0]:.4f}",
            "spam_prob_stego": f"{spam_probabilities[1]:.4f}",
            "spam_prob_original": f"{spam_probabilities[2]:.4f}",
            "lsb_chi_square_value": f"{chi_sq_stat:.4f}",
            "lsb_zeros_count": lsb_zeros,
            "lsb_ones_count": lsb_ones,
            "lsb_ratio_1s_to_total": f"{lsb_ratio_ones:.4f}",
            "histogram_difference_from_uniform": f"{hist_diff_metric:.4f}",
            "noise_residual_mean_score": f"{noise_residual_metric:.4f}",
            "rs_score_simplified_proxy": f"{rs_metric_value:.4f}",
            "jpeg_compression_difference_ratio": f"{comp_diff_ratio_metric:.4f}",
            "fourier_relative_high_freq_score": f"{fourier_score_metric:.4f}",
            "patch_based_noise_mean": f"{patch_mean_noise:.4f}",
            "patch_based_noise_std_dev": f"{patch_std_dev_noise:.4f}",
            "file_format_detected": effective_file_format,
            "has_exif_data": "Yes" if has_exif_data else "No"
        })
        # --- End of analysis for a single image ---

    # --- After processing all files, display summary reports in the sidebar ---
    if detailed_report_entries: # Check if any file was processed successfully
        st.sidebar.header("📊 Summary Reports")
        st.sidebar.markdown("---")

        # Standard Report (Files classified as Stego)
        st.sidebar.subheader("📄 Standard Report: Stego Detections")
        if standard_report_entries:
            st.sidebar.markdown("The following files were classified as **Stego** by the ensemble:")
            for entry in standard_report_entries:
                payload_kb = entry['estimated_payload_bytes'] / 1024
                st.sidebar.markdown(
                    f"- **{entry['filename']}**: \n"
                    f"  - Est. Payload (heuristic): `{entry['estimated_payload_bytes']:,} bytes` (~`{payload_kb:.2f} KB`)\n"
                    f"  - Ensemble Stego Confidence: `{entry['ensemble_confidence_stego']:.2%}`"
                )
        else:
            st.sidebar.info("✅ No files were classified as Stego in this batch.")
        
        st.sidebar.markdown("---")
        
        # Detailed Report (CSV Download)
        st.sidebar.subheader("📋 Detailed CSV Report (All Files)")
        df_detailed = pd.DataFrame(detailed_report_entries)
        
        # Display a preview of the detailed report dataframe in the sidebar
        st.sidebar.write("Preview of Detailed Report:")
        st.sidebar.dataframe(df_detailed.head(min(5, len(df_detailed)))) # Show up to 5 rows

        csv_data = df_detailed.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Download Detailed Report as CSV",
            data=csv_data,
            file_name="steganalysis_detailed_report.csv",
            mime="text/csv",
            help="Downloads a CSV file with all analysis metrics for each uploaded image."
        )
    elif uploaded_files: # Files were uploaded but none processed (e.g., all failed to decode)
        st.sidebar.header("📊 Summary Reports")
        st.sidebar.warning("⚠️ No images were successfully processed to generate reports. Please check file integrity and format.")

else:
    st.info("✨ Upload one or more images to begin steganalysis.")

st.markdown("---")
st.caption("StegoAppDB Classifier v1.1 | For educational and research purposes. Interpret results with caution.")
st.caption("LSB Payload estimation is a rough heuristic. Model predictions are based on trained data patterns.")
