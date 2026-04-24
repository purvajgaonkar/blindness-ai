"""
model.py
========
EfficientNet-B4 based Diabetic Retinopathy severity classifier.
Provides full inference pipeline with clinical explanation and Grad-CAM.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Add project root to path for local imports
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import (
    load_image, ben_graham_preprocess, clahe_lab_preprocess,
    segment_vessels, detect_lesions, full_pipeline,
    IMAGENET_MEAN, IMAGENET_STD
)


# ─── Class metadata ─────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

RISK_LEVELS = {
    0: "Low",
    1: "Low-Moderate",
    2: "Moderate",
    3: "High",
    4: "Critical",
}

BLINDNESS_RISK = {
    0:  2.0,
    1: 15.0,
    2: 35.0,
    3: 60.0,
    4: 85.0,
}

RECOMMENDATIONS = {
    0: [
        "Annual dilated eye exam recommended.",
        "Maintain HbA1c < 7% for optimal glycemic control.",
        "Monitor blood pressure (target < 130/80 mmHg).",
        "Regular cholesterol checks and lipid management.",
        "Lifestyle modifications: healthy diet, regular exercise.",
    ],
    1: [
        "Follow-up dilated eye exam in 9–12 months.",
        "Optimize glycemic control — consult endocrinologist.",
        "Refer to ophthalmologist for baseline documentation.",
        "Blood pressure and lipid management intensification.",
        "Patient education on DR progression and monitoring.",
    ],
    2: [
        "Refer to ophthalmologist within 3–6 months.",
        "Laser photocoagulation treatment may be indicated.",
        "Intensive diabetes management (HbA1c target < 6.5%).",
        "Intravitreal anti-VEGF therapy evaluation.",
        "Monthly self-monitoring of vision changes.",
        "Consider continuous glucose monitoring (CGM) device.",
    ],
    3: [
        "URGENT ophthalmology referral within 1 month.",
        "High risk of significant vision loss — do not delay.",
        "Pan-retinal photocoagulation (PRP) likely required.",
        "Intravitreal anti-VEGF injections may be needed.",
        "Extremely tight glycemic control (HbA1c < 6%).",
        "Blood pressure target < 120/80 mmHg.",
        "Arrange transport assistance if driving is unsafe.",
    ],
    4: [
        "EMERGENCY referral to vitreoretinal specialist immediately.",
        "Risk of severe vision loss or total blindness without treatment.",
        "Anti-VEGF injections (ranibizumab, bevacizumab, aflibercept).",
        "Vitreoretinal surgery may be required for vitreous hemorrhage.",
        "Immediate hospitalization may be necessary.",
        "Notify primary care physician for systemic management.",
        "Cease driving immediately pending specialist assessment.",
    ],
}

URGENCY_BADGES = {
    0: "Routine",
    1: "Soon",
    2: "Soon",
    3: "Urgent",
    4: "EMERGENCY",
}

CLINICAL_FINDINGS_BY_GRADE = {
    0: ["No visible diabetic retinopathy lesions detected.",
        "Optic disc appears normal.",
        "Retinal vessels appear healthy."],
    1: ["Possible microaneurysms detected.",
        "Mild vascular changes present.",
        "No sight-threatening lesions at this stage."],
    2: ["Microaneurysms and dot hemorrhages detected.",
        "Hard exudates may be present.",
        "Possible cotton-wool spots (nerve fiber layer infarcts).",
        "Moderate vascular abnormalities."],
    3: ["Multiple hemorrhages detected in all quadrants.",
        "Venous beading observed.",
        "Intraretinal microvascular abnormalities (IRMA).",
        "Extensive hard exudates near macula."],
    4: ["Neovascularisation of disc (NVD) or elsewhere (NVE).",
        "Tractional retinal detachment risk.",
        "Pre-retinal or vitreous hemorrhage likely.",
        "Fibrovascular proliferation detected.",
        "Severe vision-threatening changes."],
}


# ─── Custom loss ─────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing for regularisation."""

    def __init__(self, smoothing: float = 0.1, num_classes: int = 5):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


# ─── Model ───────────────────────────────────────────────────────────────────

class DRClassifier(nn.Module):
    """
    EfficientNet-B4 fine-tuned for 5-class DR severity classification.

    Architecture:
    - Backbone: EfficientNet-B4 pretrained on ImageNet (feature extractor 1792-d)
    - Head: Dropout(0.4) → Linear(1792, 512) → ReLU → Dropout(0.3) → Linear(512, 5)
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model(
                'efficientnet_b4', pretrained=pretrained, num_classes=0
            )
            feature_dim = self.backbone.num_features  # 1792 for B4
        except ImportError:
            # Fallback to torchvision
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            base = efficientnet_b4(weights=weights)
            feature_dim = base.classifier[1].in_features  # 1792
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self._use_timm = False
        else:
            self._use_timm = True

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        self._feature_dim = feature_dim
        self._num_classes  = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_timm:
            feats = self.backbone(x)
        else:
            feats = self.backbone(x)
            feats = feats.flatten(1)
        return self.classifier(feats)

    def predict_with_explanation(
        self,
        image_path: str,
        model_path: str = None,
        output_dir: str = 'outputs',
        device: str = None,
    ) -> dict:
        """
        Full inference pipeline: preprocess → classify → GradCAM → clinical report.

        Parameters
        ----------
        image_path : str   Path to input retinal image
        model_path : str   Path to saved .pth weights (optional)
        output_dir : str   Directory to save GradCAM output
        device     : str   'cuda', 'cpu', or None (auto-detect)

        Returns
        -------
        dict with keys: predicted_class, class_name, confidence, probabilities,
                        risk_level, blindness_risk_percent, clinical_findings,
                        recommendations, urgency, gradcam_path
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dev = torch.device(device)

        self.to(dev)
        self.eval()

        # ── Load weights if provided ──────────────────────────────────────
        model_loaded = False
        if model_path and os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=dev)
                if 'model_state_dict' in state:
                    state = state['model_state_dict']
                self.load_state_dict(state)
                model_loaded = True
            except Exception as exc:
                print(f"[predict_with_explanation] Warning: could not load weights: {exc}")

        # ── Preprocess image ──────────────────────────────────────────────
        img_np = load_image(image_path, size=380)
        img_np = ben_graham_preprocess(img_np, sigmaX=10)
        img_np = clahe_lab_preprocess(img_np, clip_limit=2.0)

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        input_tensor = transform(Image.fromarray(img_np)).unsqueeze(0).to(dev)

        # ── Forward pass with GradCAM hook ────────────────────────────────
        from gradcam import GradCAM
        gradcam = GradCAM(self)
        with torch.no_grad() if not model_loaded else torch.enable_grad().__enter__():
            pass

        logits, heatmap = gradcam.generate(input_tensor, target_class=None)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence  = float(probs[pred_class])

        # ── Save GradCAM image ────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        img_id = os.path.splitext(os.path.basename(image_path))[0]
        gradcam_path = os.path.join(output_dir, f"gradcam_{img_id}.png")
        gradcam.save_overlay(
            original_img=img_np,
            heatmap=heatmap,
            save_path=gradcam_path,
            pred_class=pred_class,
            confidence=confidence,
        )

        # ── Lesion analysis ───────────────────────────────────────────────
        try:
            orig_np = load_image(image_path, size=512)
            lesions = detect_lesions(orig_np)
            ex_count = int(np.sum(lesions['exudates_mask'] > 0)) // 100
            hm_count = int(np.sum(lesions['hemorrhages_mask'] > 0)) // 100
            ma_count = int(np.sum(lesions['microaneurysms_mask'] > 0)) // 100
        except Exception:
            ex_count = hm_count = ma_count = 0

        # Augment clinical findings with actual lesion counts
        clinical_findings = list(CLINICAL_FINDINGS_BY_GRADE[pred_class])
        if ex_count > 0:
            clinical_findings.append(f"Exudate regions detected (~{ex_count} clusters).")
        if hm_count > 0:
            clinical_findings.append(f"Hemorrhage regions detected (~{hm_count} clusters).")
        if ma_count > 0:
            clinical_findings.append(f"Potential microaneurysm regions (~{ma_count} spots).")

        # ── Build result dict ─────────────────────────────────────────────
        prob_dict = {CLASS_NAMES[i]: float(f"{probs[i]:.4f}") for i in range(5)}

        return {
            'predicted_class':      pred_class,
            'class_name':           CLASS_NAMES[pred_class],
            'confidence':           round(confidence, 4),
            'probabilities':        prob_dict,
            'risk_level':           RISK_LEVELS[pred_class],
            'blindness_risk_percent': BLINDNESS_RISK[pred_class],
            'clinical_findings':    clinical_findings,
            'recommendations':      RECOMMENDATIONS[pred_class],
            'urgency':              URGENCY_BADGES[pred_class],
            'gradcam_path':         gradcam_path,
            'model_loaded':         model_loaded,
        }


def build_model(num_classes: int = 5, pretrained: bool = True) -> DRClassifier:
    """Factory function to create and return a DRClassifier model."""
    return DRClassifier(num_classes=num_classes, pretrained=pretrained)


def load_model(model_path: str, device: str = None) -> DRClassifier:
    """
    Load a saved DRClassifier from disk.

    Parameters
    ----------
    model_path : str   Path to .pth checkpoint
    device     : str   Target device ('cuda' / 'cpu' / None for auto)

    Returns
    -------
    DRClassifier  Model in eval mode
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    model = DRClassifier(num_classes=5, pretrained=False)
    checkpoint = torch.load(model_path, map_location=dev)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(dev)
    model.eval()
    return model
