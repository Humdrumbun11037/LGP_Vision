"""Vision encoder for extracting feature vectors from game frames using pretrained MobileNetV2."""

from __future__ import annotations

import numpy as np
from typing import Optional

try:
    import torch
    import torchvision
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class MobileNetV2FeatureExtractor:
    """Extract feature vectors from images using frozen pretrained MobileNetV2.
    
    This class loads a pretrained MobileNetV2 model, removes the classification
    layer, and optionally reduces the feature dimension via a linear projection.
    All weights are frozen (no gradient updates).
    
    Attributes:
        feature_size: Output feature vector dimension (default: 256)
        device: PyTorch device (CPU or CUDA)
        model: The MobileNetV2 model with frozen weights
        projection: Optional linear layer to reduce feature dimension
    """
    
    def __init__(
        self,
        feature_size: int = 256,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the MobileNetV2 feature extractor.
        
        Args:
            feature_size: Output feature vector dimension. Default is 256.
                          If set to 1280, no projection is applied (raw MobileNetV2 features).
            device: PyTorch device string ('cpu', 'cuda', etc.). If None, auto-detects.
        
        Raises:
            ImportError: If torch or torchvision are not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and torchvision are required for MobileNetV2FeatureExtractor. "
                "Install with: pip install torch torchvision"
            )
        
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for image preprocessing. "
                "Install with: pip install opencv-python"
            )
        
        self.feature_size = feature_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained MobileNetV2 using modern API
        # Use DEFAULT weights to get the most up-to-date pretrained model
        self.model = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.DEFAULT
        )
        
        # Remove the classifier (final layer) to get 1280D features
        # The classifier is at model.classifier, we keep everything before it
        self.model.classifier = torch.nn.Identity()
        
        # Add optional projection layer to reduce feature dimension
        if feature_size != 1280:
            self.projection = torch.nn.Linear(1280, feature_size)
            # Initialize projection with small random weights
            torch.nn.init.xavier_uniform_(self.projection.weight)
            torch.nn.init.zeros_(self.projection.bias)
        else:
            self.projection = None
        
        # Freeze all weights (no gradient updates)
        for param in self.model.parameters():
            param.requires_grad = False
        
        if self.projection is not None:
            # Also freeze projection layer
            for param in self.projection.parameters():
                param.requires_grad = False
        
        # Move model to device
        self.model = self.model.to(self.device)
        if self.projection is not None:
            self.projection = self.projection.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        if self.projection is not None:
            self.projection.eval()
        
        # ImageNet normalization constants
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def _preprocess_image(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess image for MobileNetV2.
        
        Steps:
        1. Resize to fit 224x224 while maintaining aspect ratio (letterboxing)
        2. Pad with black bars to exactly 224x224 (preserves all content, no distortion)
        3. Convert to RGB if needed
        4. Normalize using ImageNet mean/std
        5. Convert to tensor and add batch dimension
        
        Args:
            frame: RGB image array of shape (H, W, 3) with values 0-255
        
        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224) ready for model input
            Note: Original aspect ratio is preserved via letterboxing (black padding)
        """
        # Ensure frame is RGB (H, W, 3)
        if len(frame.shape) == 2:
            # Grayscale, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            # RGBA, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Get current dimensions
        h, w = frame.shape[:2]
        
        # Resize to 224x224 using letterboxing to preserve aspect ratio and all content
        # Step 1: Calculate scale to fit image into 224x224 while maintaining aspect ratio
        scale = min(224 / h, 224 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Step 2: Resize using INTER_AREA (best for downsampling)
        resized = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )
        
        # Step 3: Pad to exactly 224x224 (letterboxing - adds black bars if needed)
        # This preserves all original content without distortion
        top = (224 - new_h) // 2
        bottom = 224 - new_h - top
        left = (224 - new_w) // 2
        right = 224 - new_w - left
        
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        )
        
        # Convert to float32 and normalize to [0, 1]
        resized = resized.astype(np.float32) / 255.0
        
        # Normalize using ImageNet statistics
        resized = (resized - self.imagenet_mean) / self.imagenet_std
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        # Use torch.tensor() as fallback if torch.from_numpy() fails
        try:
            tensor = torch.from_numpy(resized).permute(2, 0, 1)
        except (RuntimeError, TypeError) as e:
            # Fallback: convert numpy array to list then to tensor
            # This works even if PyTorch's numpy integration has issues
            tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1)
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        tensor = tensor.unsqueeze(0)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return tensor
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract feature vector from a single frame.
        
        Args:
            frame: RGB image array of shape (H, W, 3) with values 0-255
        
        Returns:
            1D numpy array of shape (feature_size,) with dtype float32
        """
        # Preprocess image
        input_tensor = self._preprocess_image(frame)
        
        # Extract features (no gradient computation)
        with torch.no_grad():
            # Get 1280D features from MobileNetV2
            features_1280 = self.model(input_tensor)
            
            # Apply projection if needed
            if self.projection is not None:
                features = self.projection(features_1280)
            else:
                features = features_1280
        
        # Convert to numpy and remove batch dimension
        features_np = features.cpu().numpy().flatten().astype(np.float32)
        
        return features_np

