"""
app.py
======
Flask backend for the RetinalAI web application.
Handles image upload → full preprocessing → vessel/lesion detection →
EfficientNet-B4 classification → Grad-CAM → JSON response with base64 images.
"""

import os
import sys
import uuid
import json
import traceback
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

# ── Path setup ────────────────────────────────────────────────────────────────
WEBAPP_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = WEBAPP_DIR.parent.resolve()
SRC_DIR = PROJECT_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

# ── Local imports ─────────────────────────────────────────────────────────────
from preprocessing import (
    load_image, full_pipeline_numpy, ben_graham_preprocess,
    clahe_lab_preprocess, segment_vessels, detect_lesions,
    get_frangi_vessels, create_lesion_overlay, create_vessel_overlay
)
from utils import numpy_to_base64, file_to_base64, logger
from model import (
    DRClassifier, CLASS_NAMES, RISK_LEVELS, BLINDNESS_RISK,
    RECOMMENDATIONS, URGENCY_BADGES, CLINICAL_FINDINGS_BY_GRADE
)

import torch
import torchvision.transforms as T
from PIL import Image as PILImage
from preprocessing import IMAGENET_MEAN, IMAGENET_STD

# ── App configuration ─────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(WEBAPP_DIR / 'templates'),
    static_folder=str(WEBAPP_DIR / 'static')
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
UPLOAD_FOLDER  = str(PROJECT_DIR / 'outputs' / 'uploads')
OUTPUT_FOLDER  = str(PROJECT_DIR / 'outputs')
MODEL_PATH     = str(PROJECT_DIR / 'models' / 'efficientnet_b4_dr_best.pth')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Lazy model loading ────────────────────────────────────────────────────────
_model = None
_model_loaded = False

def get_model():
    """Load model once and cache it."""
    global _model, _model_loaded
    if _model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _model = DRClassifier(num_classes=5, pretrained=False)
        if os.path.isfile(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location=device)
                if 'model_state_dict' in state:
                    state = state['model_state_dict']
                _model.load_state_dict(state)
                _model_loaded = True
                logger.info(f"Model loaded from {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Could not load model weights: {e}. Using demo mode.")
                _model_loaded = False
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Running in DEMO mode.")
            _model_loaded = False
        _model.to(device)
        _model.eval()
    return _model, _model_loaded


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _demo_result():
    """Return a plausible demo result when model is not trained yet."""
    import random
    grade = random.randint(0, 4)
    probs = np.random.dirichlet(alpha=[0.5] * 5).tolist()
    probs[grade] = max(probs[grade], 0.55)  # bias toward predicted class
    total = sum(probs)
    probs = [p / total for p in probs]
    return {
        'prediction':           CLASS_NAMES[grade],
        'predicted_class':      grade,
        'confidence':           round(probs[grade], 4),
        'risk_level':           RISK_LEVELS[grade],
        'blindness_risk_percent': BLINDNESS_RISK[grade],
        'probabilities':        {CLASS_NAMES[i]: round(probs[i], 4) for i in range(5)},
        'clinical_findings':    CLINICAL_FINDINGS_BY_GRADE[grade],
        'recommendations':      RECOMMENDATIONS[grade],
        'urgency':              URGENCY_BADGES[grade],
        'demo_mode':            True,
        'demo_warning':         "⚠️ Model not trained yet — showing DEMO output. Train the model for real predictions.",
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    model, loaded = get_model()
    return jsonify({
        'status':       'ok',
        'model_loaded': loaded,
        'model_path':   MODEL_PATH,
        'device':       str(next(model.parameters()).device),
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Full retinal analysis pipeline endpoint.

    Accepts: multipart/form-data with 'image' file field
    Returns: JSON with base64 images + full clinical analysis
    """
    try:
        # ── Validate request ────────────────────────────────────────────────
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided. Use field name "image".'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': f'Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400

        # ── Save uploaded image ─────────────────────────────────────────────
        img_id   = str(uuid.uuid4())[:8]
        filename = f"{img_id}_{file.filename}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        logger.info(f"Image saved: {save_path}")

        processing_steps = []

        # ── Step 1: Load original image ─────────────────────────────────────
        original_np = load_image(save_path, size=512)
        processing_steps.append("✓ Image loaded and black border removed")

        original_b64 = numpy_to_base64(original_np)

        # ── Step 2: Preprocessing pipeline ─────────────────────────────────
        preprocessed_np = full_pipeline_numpy(save_path, size=512)
        preprocessing_steps_msg = "✓ Ben Graham preprocessing + CLAHE-LAB enhancement + denoising"
        processing_steps.append(preprocessing_steps_msg)
        preprocessed_b64 = numpy_to_base64(preprocessed_np)

        # ── Step 3: Vessel segmentation ─────────────────────────────────────
        try:
            vessel_mask = segment_vessels(preprocessed_np)
            vessel_overlay_np = create_vessel_overlay(original_np, vessel_mask)
            processing_steps.append("✓ Blood vessel segmentation via Frangi vesselness filter")
        except Exception as e:
            logger.warning(f"Vessel segmentation failed: {e}")
            vessel_overlay_np = original_np.copy()
            vessel_mask = np.zeros(original_np.shape[:2], dtype=np.uint8)
            processing_steps.append("⚠ Vessel segmentation skipped (error)")

        vessel_mask_b64 = numpy_to_base64(vessel_overlay_np)

        # ── Step 4: Lesion detection ─────────────────────────────────────────
        try:
            lesions = detect_lesions(preprocessed_np)
            lesion_overlay_np = create_lesion_overlay(original_np, lesions)
            ex_count = int(np.sum(lesions['exudates_mask'] > 0)) // 200
            hm_count = int(np.sum(lesions['hemorrhages_mask'] > 0)) // 200
            ma_count = int(np.sum(lesions['microaneurysms_mask'] > 0)) // 200
            processing_steps.append(
                f"✓ Lesion detection: {ex_count} exudate, {hm_count} hemorrhage, "
                f"{ma_count} microaneurysm regions"
            )
        except Exception as e:
            logger.warning(f"Lesion detection failed: {e}")
            lesion_overlay_np = original_np.copy()
            lesions = {}
            ex_count = hm_count = ma_count = 0
            processing_steps.append("⚠ Lesion detection skipped (error)")

        lesion_overlay_b64 = numpy_to_base64(lesion_overlay_np)

        # ── Step 5: Classification ────────────────────────────────────────────
        model, model_loaded = get_model()
        device = next(model.parameters()).device

        if not model_loaded:
            result = _demo_result()
            gradcam_b64 = original_b64  # Use original as placeholder
            processing_steps.append("⚠ EfficientNet-B4 classification: DEMO mode (model not trained)")
        else:
            try:
                # Preprocess for model
                img_for_model = load_image(save_path, size=380)
                img_for_model = ben_graham_preprocess(img_for_model, sigmaX=10)
                img_for_model = clahe_lab_preprocess(img_for_model, clip_limit=2.0)

                transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
                input_tensor = transform(PILImage.fromarray(img_for_model)).unsqueeze(0).to(device)

                # Grad-CAM
                from gradcam import GradCAM
                gradcam_gen = GradCAM(model)
                logits, heatmap = gradcam_gen.generate(input_tensor, target_class=None)
                probs_arr = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                pred_class  = int(np.argmax(probs_arr))
                confidence  = float(probs_arr[pred_class])

                # Save and encode GradCAM
                gradcam_path = os.path.join(OUTPUT_FOLDER, f"gradcam_{img_id}.png")
                gradcam_gen.save_overlay(
                    original_img=img_for_model,
                    heatmap=heatmap,
                    save_path=gradcam_path,
                    pred_class=pred_class,
                    confidence=confidence,
                )
                gradcam_gen.remove_hooks()
                gradcam_b64 = file_to_base64(gradcam_path)
                processing_steps.append("✓ Grad-CAM heatmap generated")

                # Clinical findings
                clinical_findings = list(CLINICAL_FINDINGS_BY_GRADE[pred_class])
                if ex_count > 0:
                    clinical_findings.append(f"Exudate clusters detected (~{ex_count} regions)")
                if hm_count > 0:
                    clinical_findings.append(f"Hemorrhage regions detected (~{hm_count} regions)")
                if ma_count > 0:
                    clinical_findings.append(f"Microaneurysm candidates (~{ma_count} spots)")

                prob_dict = {CLASS_NAMES[i]: round(float(probs_arr[i]), 4) for i in range(5)}

                result = {
                    'prediction':           CLASS_NAMES[pred_class],
                    'predicted_class':      pred_class,
                    'confidence':           round(confidence, 4),
                    'risk_level':           RISK_LEVELS[pred_class],
                    'blindness_risk_percent': BLINDNESS_RISK[pred_class],
                    'probabilities':        prob_dict,
                    'clinical_findings':    clinical_findings,
                    'recommendations':      RECOMMENDATIONS[pred_class],
                    'urgency':              URGENCY_BADGES[pred_class],
                    'demo_mode':            False,
                }
                processing_steps.append(
                    f"✓ EfficientNet-B4: {CLASS_NAMES[pred_class]} "
                    f"({confidence*100:.1f}% confidence)"
                )

            except Exception as e:
                logger.error(f"Inference failed: {e}\n{traceback.format_exc()}")
                result = _demo_result()
                gradcam_b64 = original_b64
                processing_steps.append(f"⚠ Classification error: {str(e)[:100]}")

        # ── Vessel density metrics ─────────────────────────────────────────
        try:
            vessel_density = float(np.sum(vessel_mask > 0)) / vessel_mask.size * 100
        except Exception:
            vessel_density = 0.0

        # ── Build final response ───────────────────────────────────────────
        response = {
            **result,
            'original_b64':       original_b64,
            'preprocessed_b64':   preprocessed_b64,
            'vessel_mask_b64':    vessel_mask_b64,
            'lesion_overlay_b64': lesion_overlay_b64,
            'gradcam_b64':        gradcam_b64,
            'processing_steps':   processing_steps,
            'vessel_density':     round(vessel_density, 2),
            'exudate_count':      ex_count,
            'hemorrhage_count':   hm_count,
            'microaneurysm_count':ma_count,
            'image_id':           img_id,
        }

        logger.info(f"Prediction complete for {img_id}: {result.get('prediction')} "
                    f"({result.get('confidence', 0)*100:.1f}%)")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("  RetinalAI Flask Server Starting")
    logger.info(f"  Model path : {MODEL_PATH}")
    logger.info(f"  Output dir : {OUTPUT_FOLDER}")
    logger.info("=" * 60)

    # Pre-load model at startup
    get_model()

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
    )
