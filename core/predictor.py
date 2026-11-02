"""
Shared predictor: loads the trained model once and gives every app the same output.

Usage:
    from core.predictor import TumorPredictor
    predictor = TumorPredictor("model/best_model.pth")
    result = predictor.predict("some_mri.jpg")
    print(result.label, result.confidence, result.refer, result.reasons)
    result.heatmap   # PIL image with the Grad-CAM overlay
"""
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

# ---------------- Safety rules (change these numbers here) ----------------
# Refer to a doctor if the model is not confident enough in its top answer.
MIN_CONFIDENCE = 0.80
# Refer if the top-2 classes are too close (the model can't decide between them).
MIN_MARGIN = 0.20
# A "no tumor" answer is only trusted if P(no tumor) is at least this high.
# Missing a tumor is the worst mistake, so this rule is stricter than the others.
NOTUMOR_TRUST = 0.95
# Triage priority uses P(tumor) = 1 - P(no tumor).
PRIORITY_HIGH = 0.50
PRIORITY_MEDIUM = 0.10

NO_TUMOR = "notumor"

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}


@dataclass
class PredictionResult:
    label: str                      # e.g. "meningioma"
    display_name: str               # e.g. "Meningioma"
    confidence: float               # probability of the top class (0-1)
    probs: dict                     # {"glioma": 0.05, ...}
    p_tumor: float                  # 1 - P(no tumor)
    priority: str                   # "High" / "Medium" / "Low" (for triage)
    refer: bool                     # True -> "refer to a specialist"
    reasons: list = field(default_factory=list)  # why refer is True
    heatmap: Image.Image = None     # Grad-CAM overlay (None if not requested)
    image: Image.Image = None       # the resized input image (RGB, 224x224)


class TumorPredictor:
    def __init__(self, checkpoint_path="model/best_model.pth", device=None):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}. Put best_model.pth inside the model/ folder."
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.class_names = ckpt["class_names"]
        self.img_size = ckpt["img_size"]
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.test_accuracy = ckpt.get("test_accuracy")

        self.model = timm.create_model(
            ckpt["model_name"], pretrained=False, num_classes=len(self.class_names)
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        # Same preprocessing as eval_tf in the training notebook.
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(ckpt["mean"], ckpt["std"]),
        ])
        # Heatmap layer: saved in the checkpoint by the comparison notebook.
        # Old checkpoints (first EfficientNet-B0) don't have it, so default to conv_head.
        self.model_name = ckpt["model_name"]
        self.cam_layer = ckpt.get("cam_layer", "conv_head")
        self.cam = GradCAM(model=self.model, target_layers=[self.model.get_submodule(self.cam_layer)])
        # Grad-CAM keeps hooks and saved gradients on this one object, so two requests at the
        # same time would overwrite each other. The lock makes predictions run one at a time.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @staticmethod
    def _load(image):
        """Accepts a file path, a PIL image, or a numpy array (what Gradio gives)."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.convert("RGB")

    def _safety_check(self, label, confidence, probs):
        reasons = []
        sorted_p = sorted(probs.values(), reverse=True)
        if confidence < MIN_CONFIDENCE:
            reasons.append(f"Low confidence ({confidence:.0%} < {MIN_CONFIDENCE:.0%}).")
        if sorted_p[0] - sorted_p[1] < MIN_MARGIN:
            reasons.append("The top two answers are very close.")
        if label == NO_TUMOR and probs[NO_TUMOR] < NOTUMOR_TRUST:
            reasons.append(
                f"'No tumor' is not certain enough (P = {probs[NO_TUMOR]:.0%}, needs {NOTUMOR_TRUST:.0%}); "
                "a missed tumor is the most dangerous error."
            )
        return reasons

    @staticmethod
    def _priority(p_tumor):
        if p_tumor >= PRIORITY_HIGH:
            return "High"
        if p_tumor >= PRIORITY_MEDIUM:
            return "Medium"
        return "Low"

    # ------------------------------------------------------------------
    def predict(self, image, with_heatmap=True) -> PredictionResult:
        with self._lock:
            return self._predict(image, with_heatmap)

    def _predict(self, image, with_heatmap) -> PredictionResult:
        img = self._load(image)
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x).float()
        p = F.softmax(logits / self.temperature, dim=1)[0].cpu().numpy()
        probs = {c: float(v) for c, v in zip(self.class_names, p)}

        idx = int(p.argmax())
        label = self.class_names[idx]
        confidence = float(p[idx])
        p_tumor = 1.0 - probs.get(NO_TUMOR, 0.0)
        reasons = self._safety_check(label, confidence, probs)

        small = img.resize((self.img_size, self.img_size))
        heatmap = None
        if with_heatmap:
            rgb = np.asarray(small, dtype=np.float32) / 255.0
            cam = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(idx)])[0]
            heatmap = Image.fromarray(show_cam_on_image(rgb, cam, use_rgb=True))

        return PredictionResult(
            label=label,
            display_name=DISPLAY_NAMES.get(label, label),
            confidence=confidence,
            probs=probs,
            p_tumor=p_tumor,
            priority=self._priority(p_tumor),
            refer=bool(reasons),
            reasons=reasons,
            heatmap=heatmap,
            image=small,
        )
