import numpy as np

def quantize_residuals(residuals, T=3):
    """
    Quantize residual values to range [-T, T].
    """
    residuals = np.clip(residuals, -T, T)
    residuals = residuals.astype(int)
    residuals += T  # shift to [0, 2T] range for indexing
    return residuals

def compute_transitions(residuals):
    """
    Compute transition probability matrix for residual pairs.
    """
    size = residuals.shape[0] - 1
    T = 7  # Quantization range [-3..3], so 2T+1 = 7 possible values
    matrix = np.zeros((2*T + 1, 2*T + 1), dtype=np.float32)
    
    for i in range(size):
        r1 = residuals[i]
        r2 = residuals[i+1]
        matrix[r1, r2] += 1

    # Normalize
    total = np.sum(matrix)
    if total > 0:
        matrix /= total
    return matrix.flatten()

def spam_features(image_gray):
    """
    Extract 2044-dimensional SPAM feature vector from grayscale image.
    """
    T = 3  # quantization threshold
    
    # Ensure image is 2D numpy array
    img = image_gray.astype(np.int32)

    features = []

    # Horizontal residuals (differences between adjacent pixels in rows)
    res_h = quantize_residuals(img[:, 1:] - img[:, :-1], T)
    # Vertical residuals
    res_v = quantize_residuals(img[1:, :] - img[:-1, :], T)
    # Diagonal residuals (top-left to bottom-right)
    res_d1 = quantize_residuals(img[1:, 1:] - img[:-1, :-1], T)
    # Anti-diagonal residuals (top-right to bottom-left)
    res_d2 = quantize_residuals(img[1:, :-1] - img[:-1, 1:], T)

    # Compute transition matrices along the residual sequences
    # Flatten along rows and columns as appropriate for transitions

    # Horizontal direction transitions (along rows)
    for row in res_h:
        features.extend(compute_transitions(row))
    # Vertical direction transitions (along columns)
    for col in res_v.T:
        features.extend(compute_transitions(col))
    # Diagonal directions transitions
    for diag in [res_d1[i,:] for i in range(res_d1.shape[0])]:
        features.extend(compute_transitions(diag))
    for diag in [res_d2[i,:] for i in range(res_d2.shape[0])]:
        features.extend(compute_transitions(diag))

    features = np.array(features, dtype=np.float32)

    # Trim or pad to 2044 if needed (some implementations vary)
    if features.size > 2044:
        features = features[:2044]
    elif features.size < 2044:
        features = np.pad(features, (0, 2044 - features.size), 'constant')

    return features
