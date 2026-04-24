"""
preprocessing.py
================
Retinal image preprocessing pipeline for Diabetic Retinopathy detection.
Implements all classical CV techniques used in the APTOS 2019 competition.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from skimage.filters import frangi, sato
from skimage.morphology import skeletonize
from scipy import ndimage
import os


# ─── ImageNet normalization constants ───────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image(path: str, size: int = 512) -> np.ndarray:
    """
    Load a retinal fundus image, crop the black border, and resize.

    Parameters
    ----------
    path : str   Path to image file (.png / .jpg)
    size : int   Target square size in pixels (default 512)

    Returns
    -------
    np.ndarray  uint8 RGB image of shape (size, size, 3)
    """
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image at: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = _crop_black_border(img)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
        return img.astype(np.uint8)
    except Exception as exc:
        raise RuntimeError(f"[load_image] Failed to load '{path}': {exc}") from exc


def _crop_black_border(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Remove the black circular border common in fundus photography.
    Crops rows/columns where the mean pixel value is below 'tol'.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return img
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return img[rmin:rmax + 1, cmin:cmax + 1]
    except Exception:
        return img  # Return unchanged on failure


def ben_graham_preprocess(image: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """
    Ben Graham preprocessing — competition-winning technique from APTOS 2019.

    Formula: addWeighted(img, 4, GaussianBlur(img, 0, sigmaX), -4, 128)

    This removes low-frequency illumination variation (background lighting
    differences across the fundus) and enhances fine structures like
    blood vessels and microaneurysms. The result is a high-pass filtered
    image where the mean intensity is 128 (neutral grey).

    Parameters
    ----------
    image  : np.ndarray  RGB uint8 image
    sigmaX : int         Gaussian blur sigma (larger = more aggressive)

    Returns
    -------
    np.ndarray  Processed RGB uint8 image
    """
    try:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
        result = cv2.addWeighted(image, 4, blurred, -4, 128)
        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception as exc:
        raise RuntimeError(f"[ben_graham_preprocess] Error: {exc}") from exc


def clahe_lab_preprocess(image: np.ndarray, clip_limit: float = 2.0,
                         tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE on the L-channel of LAB color space, then convert back to RGB.

    This is the best color-preserving contrast enhancement for retinal images:
    - Converts to LAB so luminance (L) is separated from color (A, B)
    - CLAHE is applied only to L, so hue/saturation are untouched
    - Prevents the color distortion of applying CLAHE directly to RGB

    Parameters
    ----------
    image      : np.ndarray  RGB uint8 image
    clip_limit : float       CLAHE clip limit (higher = more contrast)
    tile_grid  : tuple       CLAHE tile grid size

    Returns
    -------
    np.ndarray  Enhanced RGB uint8 image
    """
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l_enhanced = clahe.apply(l_ch)
        enhanced_lab = cv2.merge([l_enhanced, a_ch, b_ch])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB).astype(np.uint8)
    except Exception as exc:
        raise RuntimeError(f"[clahe_lab_preprocess] Error: {exc}") from exc


def full_pipeline(image_path: str, size: int = 512) -> torch.Tensor:
    """
    Complete preprocessing pipeline for model inference.

    Steps:
    1. Load & crop black border
    2. Resize to (size × size)
    3. Ben Graham preprocessing (vessel enhancement)
    4. CLAHE on LAB L-channel (color-preserving contrast boost)
    5. Non-local means denoising
    6. Normalize with ImageNet mean/std
    7. Convert to CHW float tensor

    Parameters
    ----------
    image_path : str   Path to retinal image file
    size       : int   Target square resolution

    Returns
    -------
    torch.Tensor  Float32 tensor of shape (3, size, size) normalized to ImageNet stats
    """
    try:
        img = load_image(image_path, size=size)
        img = ben_graham_preprocess(img, sigmaX=10)
        img = clahe_lab_preprocess(img, clip_limit=2.0)
        img = cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3,
                                              templateWindowSize=7, searchWindowSize=21)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        tensor = transform(Image.fromarray(img))
        return tensor
    except Exception as exc:
        raise RuntimeError(f"[full_pipeline] Error processing '{image_path}': {exc}") from exc


def full_pipeline_numpy(image_path: str, size: int = 512) -> np.ndarray:
    """
    Same as full_pipeline but returns a uint8 numpy array (for visualization / webapp).

    Returns
    -------
    np.ndarray  uint8 RGB preprocessed image
    """
    try:
        img = load_image(image_path, size=size)
        img = ben_graham_preprocess(img, sigmaX=10)
        img = clahe_lab_preprocess(img, clip_limit=2.0)
        img = cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3,
                                              templateWindowSize=7, searchWindowSize=21)
        return img
    except Exception as exc:
        raise RuntimeError(f"[full_pipeline_numpy] Error processing '{image_path}': {exc}") from exc


def get_frangi_vessels(image: np.ndarray) -> np.ndarray:
    """
    Apply Frangi vesselness filter to enhance blood vessels.

    The Frangi filter analyses eigenvalues of the image's Hessian matrix at
    multiple scales. Tubular structures (blood vessels) produce a specific
    eigenvalue ratio that the filter amplifies while suppressing background.
    This is the gold-standard traditional vessel enhancement in retinal imaging.

    Parameters
    ----------
    image : np.ndarray  RGB uint8 image

    Returns
    -------
    np.ndarray  float64 vessel-enhanced image in [0, 1]
    """
    try:
        green = image[:, :, 1].astype(np.float32) / 255.0
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green_8u = (green * 255).astype(np.uint8)
        green_clahe = clahe.apply(green_8u).astype(np.float64) / 255.0
        vessel_img = frangi(green_clahe, sigmas=range(1, 6), alpha=0.5,
                            beta=0.5, gamma=15, black_ridges=False)
        if vessel_img.max() > 0:
            vessel_img = vessel_img / vessel_img.max()
        return vessel_img.astype(np.float64)
    except Exception as exc:
        raise RuntimeError(f"[get_frangi_vessels] Error: {exc}") from exc


def segment_vessels(image: np.ndarray) -> np.ndarray:
    """
    Segment blood vessels using green channel → CLAHE → Frangi → morphological cleanup.

    Returns a binary mask where white pixels (255) represent vessel pixels.

    Parameters
    ----------
    image : np.ndarray  RGB uint8 image

    Returns
    -------
    np.ndarray  uint8 binary vessel mask (0 or 255)
    """
    try:
        vessel_float = get_frangi_vessels(image)
        thresh = (vessel_float > 0.05).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cleaned.astype(np.uint8)
    except Exception as exc:
        raise RuntimeError(f"[segment_vessels] Error: {exc}") from exc


def detect_lesions(image: np.ndarray) -> dict:
    """
    Detect pathological lesions in retinal images using classical morphological methods.

    Detects three types:
    - Exudates (bright hard deposits): White / yellowish regions — top-hat transform
    - Hemorrhages (dark red blobs): Dark irregular regions — black-hat transform
    - Microaneurysms: Small dark circular structures — morphological opening

    Parameters
    ----------
    image : np.ndarray  RGB uint8 image

    Returns
    -------
    dict with keys:
        'exudates_mask'      : uint8 binary mask (bright lesions)
        'hemorrhages_mask'   : uint8 binary mask (dark lesions)
        'microaneurysms_mask': uint8 binary mask (tiny dark dots)
    """
    try:
        green = image[:, :, 1]
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # ── Exudates (bright) ──────────────────────────────────────────────
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel_large)
        _, ex_mask = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ex_mask = cv2.morphologyEx(ex_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)

        # ── Hemorrhages (dark) ─────────────────────────────────────────────
        blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel_large)
        bh_blur  = cv2.GaussianBlur(blackhat, (5, 5), 0)
        hem_mask = cv2.adaptiveThreshold(bh_blur, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, -3)
        hem_mask = cv2.morphologyEx(hem_mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                    iterations=2)

        # ── Microaneurysms (tiny dark circles) ────────────────────────────
        kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened    = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel_ma)
        diff      = cv2.subtract(opened, green)
        _, ma_mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        kernel_ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ma_mask = cv2.morphologyEx(ma_mask, cv2.MORPH_OPEN, kernel_ref, iterations=1)

        return {
            'exudates_mask':       ex_mask.astype(np.uint8),
            'hemorrhages_mask':    hem_mask.astype(np.uint8),
            'microaneurysms_mask': ma_mask.astype(np.uint8),
        }
    except Exception as exc:
        raise RuntimeError(f"[detect_lesions] Error: {exc}") from exc


def create_lesion_overlay(image: np.ndarray, lesions: dict) -> np.ndarray:
    """
    Overlay detected lesions on original image using distinct colors.

    Color coding:
    - Exudates      : Yellow  (255, 255, 0)
    - Hemorrhages   : Red     (255, 0,   0)
    - Microaneurysms: Magenta (255, 0, 255)

    Parameters
    ----------
    image   : np.ndarray  Original RGB image
    lesions : dict        Output of detect_lesions()

    Returns
    -------
    np.ndarray  RGB image with colored lesion overlays
    """
    try:
        overlay = image.copy().astype(np.uint8)
        alpha = 0.6

        def _apply(mask, color):
            nonlocal overlay
            colored = np.zeros_like(image)
            colored[mask > 0] = color
            mask_bool = mask > 0
            overlay[mask_bool] = (
                alpha * np.array(color) + (1 - alpha) * overlay[mask_bool]
            ).astype(np.uint8)

        _apply(lesions['exudates_mask'],       [255, 255,   0])  # Yellow
        _apply(lesions['hemorrhages_mask'],    [255,   0,   0])  # Red
        _apply(lesions['microaneurysms_mask'], [255,   0, 255])  # Magenta
        return overlay
    except Exception as exc:
        raise RuntimeError(f"[create_lesion_overlay] Error: {exc}") from exc


def create_vessel_overlay(image: np.ndarray, vessel_mask: np.ndarray) -> np.ndarray:
    """
    Overlay vessel mask on original image in cyan/teal color.

    Parameters
    ----------
    image       : np.ndarray  Original RGB image
    vessel_mask : np.ndarray  Binary vessel mask from segment_vessels()

    Returns
    -------
    np.ndarray  RGB image with cyan vessel overlay
    """
    try:
        overlay = image.copy().astype(np.uint8)
        cyan_color = np.array([0, 212, 255], dtype=np.uint8)
        vessel_bool = vessel_mask > 0
        overlay[vessel_bool] = (
            0.7 * cyan_color + 0.3 * overlay[vessel_bool]
        ).astype(np.uint8)
        return overlay
    except Exception as exc:
        raise RuntimeError(f"[create_vessel_overlay] Error: {exc}") from exc
