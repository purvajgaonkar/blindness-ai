"""
gradcam.py
==========
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation
for EfficientNet-B4 in the DR classification pipeline.

Grad-CAM uses the gradients of the target class score flowing into the
last convolutional layer to produce a coarse localisation map highlighting
which regions of the retina were important for the prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization," ICCV 2017.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B4 (timm or torchvision).

    Hooks into the last convolutional block of EfficientNet to capture:
    1. Forward activations (feature maps from last conv block)
    2. Backward gradients (gradients of class score w.r.t. those feature maps)

    The class activation map is computed as:
        CAM = ReLU( Σ_k  α_k * A_k )
    where A_k is the k-th feature map and α_k is the global-average-pooled gradient.
    """

    def __init__(self, model: nn.Module):
        self.model   = model
        self.device  = next(model.parameters()).device

        # Storage for hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register hooks on last convolutional block
        self._hooks = []
        self._register_hooks()

    def _get_target_layer(self) -> nn.Module:
        """
        Find the last convolutional block of the backbone.
        Works for both timm and torchvision EfficientNet-B4.
        """
        backbone = self.model.backbone

        # timm EfficientNet: iterate named modules to find last Conv2d
        target = None
        for name, module in backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                target = module
        if target is None:
            raise RuntimeError("Could not find Conv2d layer in backbone for Grad-CAM hooks")
        return target

    def _register_hooks(self):
        """Attach forward and backward hooks to the target layer."""
        target_layer = self._get_target_layer()

        def _forward_hook(module, input, output):
            self._activations = output.detach().clone()

        def _backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach().clone()

        h1 = target_layer.register_forward_hook(_forward_hook)
        h2 = target_layer.register_full_backward_hook(_backward_hook)
        self._hooks.extend([h1, h2])

    def remove_hooks(self):
        """Remove all registered hooks (call after inference to free memory)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Run forward + backward pass and compute Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor  Batch of shape (1, 3, H, W), normalised
        target_class : int or None   Class index to explain; None = predicted class

        Returns
        -------
        logits   : torch.Tensor  Raw model output (1, num_classes)
        heatmap  : np.ndarray    Normalised heatmap in [0, 1], shape (H, W)
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=-1))

        # Backward pass for target class score
        score = logits[0, target_class]
        score.backward()

        # Compute Grad-CAM
        gradients   = self._gradients          # (1, C, H, W)
        activations = self._activations        # (1, C, H, W)

        if gradients is None or activations is None:
            raise RuntimeError("Grad-CAM: hooks did not capture gradients or activations")

        # Global-average pool the gradients across spatial dims → (1, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        # Weighted sum of activation maps → (1, H, W)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Upsample to input resolution and normalise
        cam_np = cam.squeeze().cpu().numpy()
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize to (380, 380) then caller can resize further
        h = input_tensor.shape[2]
        w = input_tensor.shape[3]
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

        return logits.detach(), cam_resized

    def save_overlay(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        save_path: str,
        pred_class: int,
        confidence: float,
    ):
        """
        Overlay jet-colormap heatmap on original image at 50% opacity.
        Annotates with predicted class, confidence, and risk level.

        Parameters
        ----------
        original_img : np.ndarray  uint8 RGB retinal image
        heatmap      : np.ndarray  normalised float heatmap [0, 1]
        save_path    : str         Output file path (.png)
        pred_class   : int         Predicted class index
        confidence   : float       Prediction confidence [0, 1]
        """
        from model import CLASS_NAMES, RISK_LEVELS

        img_h, img_w = original_img.shape[:2]

        # Resize heatmap to match image
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap_uint8, (img_w, img_h))

        # Apply jet colormap
        heatmap_jet = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

        # Blend with original
        img_float    = original_img.astype(np.float32)
        heatmap_float = heatmap_jet.astype(np.float32)
        blended = (0.5 * img_float + 0.5 * heatmap_float).clip(0, 255).astype(np.uint8)

        # Convert to BGR for cv2 drawing
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        # ── Annotations ───────────────────────────────────────────────────
        # Black banner at top
        banner_h = 80
        banner = np.zeros((banner_h, img_w, 3), dtype=np.uint8)
        blended_bgr = np.vstack([banner, blended_bgr])

        class_name  = CLASS_NAMES.get(pred_class, f"Class {pred_class}")
        risk        = RISK_LEVELS.get(pred_class, "Unknown")
        conf_pct    = f"{confidence * 100:.1f}%"

        severity_colors = {0: (0, 200, 0), 1: (0, 200, 200), 2: (0, 140, 255),
                           3: (0, 0, 255), 4: (0, 0, 180)}
        color_bgr = severity_colors.get(pred_class, (255, 255, 255))

        font      = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(blended_bgr, f"Grad-CAM: {class_name}",
                    (10, 30), font, 0.9, color_bgr, 2, cv2.LINE_AA)
        cv2.putText(blended_bgr, f"Confidence: {conf_pct}  |  Risk: {risk}",
                    (10, 65), font, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        # Colorbar strip on right
        cbar_w = 20
        cbar = np.linspace(255, 0, blended_bgr.shape[0], dtype=np.uint8)
        cbar_img = np.tile(cbar[:, None], (1, cbar_w))
        cbar_colored = cv2.applyColorMap(cbar_img, cv2.COLORMAP_JET)
        blended_bgr = np.hstack([blended_bgr, cbar_colored])

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, blended_bgr)
        print(f"[GradCAM] Saved to: {save_path}")


def generate_gradcam(
    model: nn.Module,
    image_path: str,
    output_dir: str = 'outputs',
    device: str = None,
) -> Tuple[int, float, str]:
    """
    Convenience function: run Grad-CAM for a single image.

    Parameters
    ----------
    model      : nn.Module   Trained DRClassifier
    image_path : str         Path to retinal image
    output_dir : str         Directory to save heatmap
    device     : str         'cuda' / 'cpu' / None

    Returns
    -------
    (predicted_class, confidence, gradcam_save_path)
    """
    import torchvision.transforms as T
    from PIL import Image as PILImage
    from preprocessing import (load_image, ben_graham_preprocess,
                               clahe_lab_preprocess, IMAGENET_MEAN, IMAGENET_STD)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    model.to(dev)
    model.eval()

    img_np = load_image(image_path, size=380)
    img_np = ben_graham_preprocess(img_np, sigmaX=10)
    img_np = clahe_lab_preprocess(img_np, clip_limit=2.0)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    input_tensor = transform(PILImage.fromarray(img_np)).unsqueeze(0)

    gradcam_gen = GradCAM(model)
    logits, heatmap = gradcam_gen.generate(input_tensor, target_class=None)
    probs       = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred_class  = int(np.argmax(probs))
    confidence  = float(probs[pred_class])

    os.makedirs(output_dir, exist_ok=True)
    img_id      = os.path.splitext(os.path.basename(image_path))[0]
    gradcam_path = os.path.join(output_dir, f"gradcam_{img_id}.png")

    gradcam_gen.save_overlay(
        original_img=img_np,
        heatmap=heatmap,
        save_path=gradcam_path,
        pred_class=pred_class,
        confidence=confidence,
    )
    gradcam_gen.remove_hooks()

    return pred_class, confidence, gradcam_path
