"""
Mask Generator Module
Automatic damage detection using classical computer vision techniques.
Outputs binary mask: white = repair region, black = keep.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for damage detection.
    
    Args:
        image: Input image as numpy array (RGB or BGR)
    
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return gray


def detect_scratches(gray: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """
    Detect scratch-like damage using edge detection and morphological operations.
    
    Args:
        gray: Grayscale image
        sensitivity: Detection sensitivity (0.5-2.0, higher = more sensitive)
    
    Returns:
        Binary mask of detected scratches
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate adaptive thresholds based on image statistics
    median_val = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33 * sensitivity) * median_val))
    upper = int(min(255, (1.0 + 0.33 * sensitivity) * median_val))
    
    # Canny edge detection
    edges = cv2.Canny(blurred, lower, upper)
    
    # Use morphological operations to connect nearby edges (potential scratches)
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel_line, iterations=2)
    
    # Close small gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_line, iterations=1)
    
    return closed


def detect_missing_regions(
    gray: np.ndarray, 
    block_size: int = 35,
    c_value: int = 10
) -> np.ndarray:
    """
    Detect missing or corrupted regions using adaptive thresholding.
    
    Args:
        gray: Grayscale image
        block_size: Size of neighborhood for adaptive thresholding
        c_value: Constant subtracted from mean
    
    Returns:
        Binary mask of detected missing regions
    """
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Apply bilateral filter to preserve edges while smoothing
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive thresholding to find uniform/abnormal regions
    adaptive = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value
    )
    
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opened


def detect_bright_spots(gray: np.ndarray, threshold_factor: float = 0.95) -> np.ndarray:
    """
    Detect abnormally bright spots that may indicate damage or artifacts.
    
    Args:
        gray: Grayscale image
        threshold_factor: Percentile threshold (0.9-0.99)
    
    Returns:
        Binary mask of detected bright spots
    """
    # Calculate threshold based on image histogram
    threshold = np.percentile(gray, threshold_factor * 100)
    
    # Threshold bright regions
    _, bright_mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return cleaned


def detect_dark_spots(gray: np.ndarray, threshold_factor: float = 0.05) -> np.ndarray:
    """
    Detect abnormally dark spots that may indicate damage.
    
    Args:
        gray: Grayscale image
        threshold_factor: Percentile threshold (0.01-0.1)
    
    Returns:
        Binary mask of detected dark spots
    """
    # Calculate threshold based on image histogram
    threshold = np.percentile(gray, threshold_factor * 100)
    
    # Threshold dark regions
    _, dark_mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY_INV)
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return cleaned


def refine_mask(
    mask: np.ndarray, 
    min_area: int = 100,
    dilation_iterations: int = 3
) -> np.ndarray:
    """
    Refine the damage mask by removing small regions and dilating valid ones.
    
    Args:
        mask: Binary mask to refine
        min_area: Minimum contour area to keep
        dilation_iterations: Number of dilation iterations
    
    Returns:
        Refined binary mask
    """
    # Find contours and filter by area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create clean mask with only significant regions
    refined = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(refined, [contour], -1, 255, -1)
    
    # Dilate to ensure full coverage of damaged regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(refined, kernel, iterations=dilation_iterations)
    
    # Fill holes
    filled = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return filled


def generate_damage_mask(
    image: np.ndarray,
    scratch_sensitivity: float = 1.0,
    detect_bright: bool = True,
    detect_dark: bool = True,
    min_damage_area: int = 50,
    dilation_strength: int = 2
) -> Tuple[np.ndarray, dict]:
    """
    Generate a comprehensive damage mask combining multiple detection methods.
    
    Args:
        image: Input image as numpy array (RGB format)
        scratch_sensitivity: Sensitivity for scratch detection (0.5-2.0)
        detect_bright: Whether to detect bright spots
        detect_dark: Whether to detect dark spots
        min_damage_area: Minimum area for damage regions
        dilation_strength: Strength of mask dilation
    
    Returns:
        Tuple of (binary mask, detection statistics dict)
    """
    # Preprocess
    gray = preprocess_image(image)
    
    # Initialize combined mask
    height, width = gray.shape
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Detect different types of damage
    stats = {}
    
    # Scratch detection
    scratch_mask = detect_scratches(gray, scratch_sensitivity)
    stats['scratches'] = int(np.sum(scratch_mask > 0))
    combined_mask = cv2.bitwise_or(combined_mask, scratch_mask)
    
    # Missing region detection
    missing_mask = detect_missing_regions(gray)
    stats['missing_regions'] = int(np.sum(missing_mask > 0))
    combined_mask = cv2.bitwise_or(combined_mask, missing_mask)
    
    # Bright spot detection
    if detect_bright:
        bright_mask = detect_bright_spots(gray)
        stats['bright_spots'] = int(np.sum(bright_mask > 0))
        combined_mask = cv2.bitwise_or(combined_mask, bright_mask)
    
    # Dark spot detection
    if detect_dark:
        dark_mask = detect_dark_spots(gray)
        stats['dark_spots'] = int(np.sum(dark_mask > 0))
        combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
    
    # Refine the combined mask
    final_mask = refine_mask(
        combined_mask, 
        min_area=min_damage_area,
        dilation_iterations=dilation_strength
    )
    
    # Calculate coverage statistics
    total_pixels = height * width
    damage_pixels = np.sum(final_mask > 0)
    stats['total_damage_pixels'] = int(damage_pixels)
    stats['damage_percentage'] = float(round(100 * damage_pixels / total_pixels, 2))
    
    return final_mask, stats


def mask_from_pil(
    pil_image: Image.Image,
    **kwargs
) -> Tuple[Image.Image, dict]:
    """
    Generate damage mask from PIL Image.
    
    Args:
        pil_image: Input PIL Image
        **kwargs: Arguments passed to generate_damage_mask
    
    Returns:
        Tuple of (mask as PIL Image, detection statistics)
    """
    # Convert PIL to numpy
    image_array = np.array(pil_image)
    
    # Generate mask
    mask_array, stats = generate_damage_mask(image_array, **kwargs)
    
    # Convert back to PIL
    mask_pil = Image.fromarray(mask_array, mode='L')
    
    return mask_pil, stats


def create_visualization(
    original: np.ndarray, 
    mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create a visualization overlay of the mask on the original image.
    
    Args:
        original: Original image (RGB)
        mask: Binary mask
        alpha: Overlay transparency
    
    Returns:
        Visualization image with mask overlay
    """
    # Ensure original is RGB
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    elif original.shape[2] == 4:
        original = cv2.cvtColor(original, cv2.COLOR_RGBA2RGB)
    
    # Create colored mask overlay (red)
    overlay = original.copy()
    overlay[mask > 0] = [255, 0, 0]
    
    # Blend
    visualization = cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)
    
    return visualization
