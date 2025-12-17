# SHARP Network Architecture

This document provides a detailed explanation of the SHARP (SHARP Monocular View Synthesis) network pipeline.

## Pipeline Overview

The SHARP pipeline takes a single RGB image as input and predicts a set of 3D Gaussians that represent the scene. The process involves monocular depth estimation, feature extraction, and delta prediction on top of a base Gaussian initialization.

The core logic is implemented in `src/sharp/models/predictor.py` within the `RGBGaussianPredictor` class.

### 1. Preprocessing

**Input:** Single RGB Image.

- **Resizing:** The input image is resized to the internal resolution of the network (default: 1536x1536).
- **Normalization:** The image is normalized to [0, 1] range.
- **Disparity Factor:** A disparity factor is calculated based on the focal length and image width to facilitate depth-to-disparity conversion.

### 2. Monocular Depth Estimation

**Module:** `MonodepthDensePredictionTransformer` (`src/sharp/models/monodepth.py`)

The first stage estimates the geometry of the scene using a monocular depth estimation network.

- **Encoder (`SlidingPyramidNetwork`):**
  - Extracts multi-scale features from the image using a Vision Transformer (ViT) backbone (e.g., DINOv2).
  - It processes the image at multiple scales (original, 0.5x, 0.25x) using a sliding window approach to handle high-resolution inputs efficiently.
- **Decoder (`MultiresConvDecoder`):**
  - Fuses the multi-scale features from the encoder.
- **Head:**
  - Predicts disparity (inverse depth). The model can predict multiple layers of depth (e.g., foreground and background surfaces) if configured (default `num_monodepth_layers=2`).
- **Output:**
  - `disparity`: The estimated disparity map.
  - `encodings`: Intermediate features passed to the Gaussian predictor.

### 3. Gaussian Prediction

**Module:** `RGBGaussianPredictor` (`src/sharp/models/predictor.py`)

This stage predicts the parameters of the 3D Gaussians. It uses a "residual" approach: initializing base Gaussians from the estimated depth and then predicting "deltas" (offsets) to refine them.

#### A. Depth Alignment & Initialization
- **Depth Alignment:** The estimated monodepth is optionally aligned to a ground truth depth (if available during training) or normalized.
- **Initialization (`MultiLayerInitializer`):**
  - Converts the aligned depth into "Base Gaussians".
  - **Position:** A grid of points in Normalized Device Coordinates (NDC) initialized at the estimated depth.
  - **Scale:** Initialized based on local disparity.
  - **Color:** Sampled from the input image (using average pooling).
  - **Opacity:** Initialized to a constant value to ensure uniform initial transmittance.
  - **Rotation:** Initialized to identity (no rotation).

#### B. Feature Extraction
**Module:** `GaussianDensePredictionTransformer` (`src/sharp/models/gaussian_decoder.py`)

- **Input:**
  - `feature_input`: A concatenation of the RGB image and the normalized disparity map.
  - `encodings`: Features extracted from the Monodepth model.
- **Architecture:**
  - **Image Encoder:** Encodes the `feature_input`.
  - **Fusion:** Fuses the features from the Monodepth decoder with the encoded `feature_input`.
  - **Heads:** Produces two sets of feature maps: `texture_features` and `geometry_features`.

#### C. Delta Prediction
**Module:** `DirectPredictionHead` (`src/sharp/models/heads.py`)

- Convolutional layers process the `texture_features` and `geometry_features`.
- Predicts **deltas** for all Gaussian parameters:
  - Position offsets (x, y, z)
  - Scale offsets
  - Rotation offsets (Quaternion)
  - Color offsets
  - Opacity offsets

#### D. Composition
**Module:** `GaussianComposer` (`src/sharp/models/composer.py`)

- Combines the **Base Gaussians** (from Initialization) and the **Deltas** (from Prediction).
- Applies activation functions to ensure valid ranges:
  - **Sigmoid** for color and opacity.
  - **Exponential/Sigmoid** based scaling for scales.
  - **Normalization** for quaternions (implicit in rendering).
- **Result:** A set of 3D Gaussians in NDC space.

### 4. Post-processing

- **Unprojection:** The Gaussians are unprojected from NDC space back to Metric space using the camera intrinsics.
- **Output:** The final `.ply` file containing 3D Gaussians ready for rendering.

## Summary Diagram

```mermaid
graph TD
    Input[RGB Image] --> Pre[Preprocessing]
    Pre --> MonoDepth[Monodepth Estimator]

    subgraph Monodepth
        MonoDepth --> Features[Image Features]
        MonoDepth --> Disparity[Disparity Map]
    end

    Disparity --> Init[Initializer]
    Input --> Init
    Init --> BaseG[Base Gaussians]
    Init --> FeatIn[Image + Depth Input]

    FeatIn --> GaussNet[Gaussian Predictor Network]
    Features --> GaussNet

    GaussNet --> Deltas[Parameter Deltas]

    BaseG --> Composer[Gaussian Composer]
    Deltas --> Composer

    Composer --> GaussNDC[Gaussians (NDC)]
    GaussNDC --> Unproj[Unprojection]
    Unproj --> Output[Metric 3D Gaussians]
```
