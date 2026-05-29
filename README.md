# VividTones - Intelligent Image Colorization System

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Architecture & Model Details](#architecture--model-details)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Technical Details](#technical-details)
8. [Model Information](#model-information)
9. [Example Use Cases](#example-use-cases)
10. [Troubleshooting](#troubleshooting)
11. [Dependencies](#dependencies)
12. [License & Attribution](#license--attribution)

---

## Project Overview

**VividTones** is an advanced image colorization web application that automatically converts grayscale (black and white) images into vibrant, realistic color images using state-of-the-art deep learning models. The application leverages two powerful pre-trained neural network architectures (ECCV16 and SIGGRAPH17) to intelligently predict and apply colors to monochrome images.

### Key Capabilities:
- **Automatic Color Prediction**: Intelligently predicts realistic colors for grayscale images
- **Multiple Model Support**: Offers two different colorization algorithms for comparison
- **Interactive Web Interface**: User-friendly Streamlit-based web application
- **Real-time Processing**: GPU-accelerated inference for fast results
- **Download Support**: Export colorized results in high-quality formats

### Target Applications:
- Historical photograph restoration
- Archival digitization and enhancement
- Film and media colorization
- Artistic and creative projects
- Educational demonstrations of deep learning capabilities

---

## Features

### Core Features:
1. **Dual Colorization Models**
   - ECCV16: Automatic colorization using ECCV 2016 architecture
   - SIGGRAPH17: Advanced colorization using SIGGRAPH 2017 architecture

2. **Web Application Interface**
   - Streamlit-based responsive UI
   - Drag-and-drop image upload
   - Real-time image processing
   - Side-by-side comparison of results

3. **GPU Acceleration**
   - Automatic CUDA GPU detection
   - Optimized tensor operations with PyTorch
   - Fast inference times

4. **Image Format Support**
   - JPG/JPEG format
   - PNG format
   - RGB and grayscale input handling

5. **Export Capabilities**
   - Download colorized outputs in JPEG format
   - High-quality image processing

---

## Project Structure

```
VividTones/
├── README.md                          # Project documentation (this file)
├── app.py                            # Main Streamlit web application
├── requirements.txt                  # Python dependencies
├── Colorization.ipynb               # Jupyter notebook with examples and experiments
├── Project Documentation.docx        # Additional project documentation
├── vividtones.pptx                  # Project presentation
├── Images/                          # Sample images directory
│   ├── ILSVRC2012_val_00041580.JPEG
│   ├── ILSVRC2012_val_00046524.JPEG
│   ├── ILSVRC2012_val_00046834.JPEG
│   ├── apple.jpg
│   ├── lion.jpg
│   ├── mangos.jpg
│   ├── person1.jpg
│   ├── place1.jpg
│   ├── place2.jpg
│   ├── place3.jpg
│   ├── railways.jpg
│   ├── sunflower.jpg
│   └── tulips.jpg
└── .git/                            # Git repository metadata
```

### File Descriptions:

- **app.py**: The main application file containing:
  - Neural network model definitions (BaseColor, ECCVGenerator, SIGGRAPHGenerator)
  - Image preprocessing and postprocessing functions
  - Streamlit web interface and user interaction logic
  - Model loading and inference pipeline

- **requirements.txt**: Lists all Python package dependencies with specific versions

- **Colorization.ipynb**: Jupyter notebook containing experimental code and examples

- **Images/**: Sample dataset of test images for demonstration purposes

---

## Architecture & Model Details

### System Architecture Overview

The application follows a classic deep learning pipeline:

1. **Image Input** → Uploaded grayscale or color image
2. **Preprocessing** → Convert to LAB color space, normalize, resize
3. **Neural Network** → Two parallel inference paths (ECCV16 and SIGGRAPH17)
4. **Colorization** → Predict a* and b* channels
5. **Postprocessing** → Combine L channel with predicted ab channels, convert back to RGB
6. **Output Display** → Show results and provide download options

### Color Space Conversion

The application uses the **LAB color space** for all processing:
- **L channel**: Luminance (brightness) - preserved from input
- **a channel**: Green-Red color component
- **b channel**: Blue-Yellow color component

This separation allows the networks to focus on predicting colors while maintaining original brightness information.

### BaseColor Class

A foundational class that handles color space normalization:

```
normalize_l(input_l): Centers and scales L channel to [-0.5, 0.5]
unnormalize_l(input_l): Reverses L channel normalization
normalize_ab(input_ab): Scales ab channels by factor of 110
unnormalize_ab(input_ab): Reverses ab channel scaling
```

### ECCVGenerator Architecture

A fully convolutional neural network inspired by ECCV 2016 colorization research:

**Key Components:**
- **Input**: Single L channel (luminance)
- **Encoder Blocks**: 8 sequential convolutional blocks with increasing receptive fields
  - Conv1-Conv2: Strided convolutions for downsampling
  - Conv3-Conv4: Deep feature extraction
  - Conv5-Conv7: Dilated convolutions for large receptive fields
- **Decoder**: Transposed convolution for upsampling
- **Output**: 313-class softmax distribution over ab color quantization
- **Final Layer**: 4x upsampling to restore original resolution

**Architecture Details:**
```
Input (1, H, W)
  ↓ Conv1-2 (stride 1, stride 2) → 64 channels
  ↓ Conv2 (stride 2) → 128 channels  
  ↓ Conv3 (stride 2) → 256 channels
  ↓ Conv4 (stride 1) → 512 channels
  ↓ Conv5-7 (dilated, stride 1) → 512 channels
  ↓ Conv8 (transpose, stride 2) → 256 channels
  ↓ Output: 313 classes → 2 (ab) channels
  ↓ 4x Upsample
Output (2, H, W)
```

### SIGGRAPHGenerator Architecture

An advanced architecture from SIGGRAPH 2017 with skip connections:

**Key Components:**
- **Input**: L channel (luminance) + optional ab hints + optional mask (4 channels)
- **Multi-scale Encoder**: Conv1-7 with progressive downsampling
- **Skip Connections**: Feature connections from encoder to decoder levels
- **Multi-scale Decoder**: Conv8-10 with upsampling and concatenation
- **Dual Output**:
  - Classification branch: 529-class color predictions
  - Regression branch: Direct ab channel prediction with Tanh activation
- **Output**: ab color channels

**Architecture Details:**
```
Input (4, H, W) [L, a, b, mask]
  ↓ Conv1: (stride 1) → 64 channels
  ↓ Conv2: (stride 1 on H/2) → 128 channels
  ↓ Conv3: (stride 1 on H/4) → 256 channels
  ↓ Conv4: (stride 1 on H/8) → 512 channels
  ↓ Conv5-7: Dilated convolutions → 512 channels
  ↓ Conv8up (transpose) + skip from Conv3 → 256 channels
  ↓ Conv9up (transpose) + skip from Conv2 → 128 channels
  ↓ Conv10up (transpose) + skip from Conv1 → 128 channels
  ↓ Output: Direct regression → 2 (ab) channels
Output (2, H, W)
```

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA support for faster processing
- Approximately 500MB of disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/Chakrapani2122/VividTones.git
cd VividTones
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **streamlit**: Web framework for the application
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computing library
- **Pillow**: Image processing library
- **scikit-image**: Advanced image processing
- **matplotlib**: Visualization library

### Step 4: (Optional) Install GPU Support

For CUDA GPU acceleration (NVIDIA GPUs only):

```bash
# Replace with your CUDA version (11.8, 12.1, etc.)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## Usage Guide

### Basic Workflow

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

2. **Upload an Image**
   - Click "Choose an image..." button
   - Select a JPG, JPEG, or PNG file
   - The app accepts both color and grayscale images

3. **Wait for Processing**
   - The application will:
     - Load the pre-trained models
     - Preprocess your image
     - Run inference on both ECCV16 and SIGGRAPH17 models
     - Display results (processing time: typically 10-30 seconds)

4. **View Results**
   - Original image and grayscale version shown on the left
   - ECCV16 colorization shown on top right
   - SIGGRAPH17 colorization shown on bottom right

5. **Download Outputs**
   - Click "Download ECCV16 Output" to save ECCV16 result
   - Click "Download SIGGRAPH17 Output" to save SIGGRAPH17 result

### Interface Elements

- **File Uploader**: Supports JPG, JPEG, PNG formats
- **Results Section**: 2-column layout showing before/after comparisons
- **Download Buttons**: Export high-quality JPEG images

### Tips for Best Results

- **Image Quality**: Higher resolution input images produce better results
- **Content Type**: Works best with:
  - Historical photographs
  - Portraits and faces
  - Natural scenes
  - Objects and still life
- **Avoid**: 
  - Very small images (< 100x100 pixels)
  - Highly stylized artwork
  - Abstract images

---

## Technical Details

### Image Processing Pipeline

#### Preprocessing (`preprocess_img` function):

1. **Resize to Standard Size**: 256x256 pixels (enables batch processing)
2. **RGB to LAB Conversion**: Using scikit-image's `color.rgb2lab()`
3. **Extract L Channel**: Preserve luminance information
4. **Normalization**:
   - Center L channel around 50 (neutral gray)
   - Scale to range approximately [-0.5, 0.5]
5. **Tensor Conversion**: Convert to PyTorch tensors with batch dimension
6. **Output**: Two tensors
   - `tens_orig_l`: Original resolution L channel
   - `tens_rs_l`: Resized L channel (256x256)

#### Model Inference:

1. **Model Selection**: Load either ECCV16 or SIGGRAPH17
2. **Forward Pass**: Feed resized L channel tensor through network
3. **Output**: Predicted ab channels (H/4 × W/4 resolution)

#### Postprocessing (`postprocess_tens` function):

1. **Resize ab Channels**: Interpolate to original image resolution
2. **Combine Channels**: Concatenate L (original) + ab (predicted)
3. **LAB to RGB Conversion**: Convert back to standard RGB color space
4. **Denormalization**: Scale values to [0, 1] range
5. **Output**: RGB image array ready for display

### Tensor Operations

- **Input Tensors**: Shape `(batch, channels, height, width)` where batch=1
- **L Channel**: Shape `(1, 1, H, W)` - single luminance channel
- **ab Channels**: Shape `(1, 2, H, W)` - two color channels

### GPU/CPU Handling

```python
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()
    tensor.cuda()  # Move tensors to GPU for processing
```

---

## Model Information

### ECCV16 Model

**Source**: ECCV 2016 Colorization Research

**Characteristics**:
- Single-input architecture (L channel only)
- 8 convolutional/deconvolutional blocks
- Achieves good colorization with faster inference
- Better for natural scenes and straightforward images
- Pre-trained weight source: https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth

**Model Size**: ~130MB
**Inference Time**: ~1-2 seconds on GPU

### SIGGRAPH17 Model

**Source**: SIGGRAPH 2017 Colorization Research

**Characteristics**:
- Multi-input architecture (supports hints and masks)
- Skip connections between encoder and decoder
- More sophisticated feature extraction
- Produces more detailed and sometimes more vibrant colorization
- Better for complex scenes with varied objects
- Pre-trained weight source: https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth

**Model Size**: ~140MB
**Inference Time**: ~2-3 seconds on GPU

### Pre-trained Weights

Both models use weights pre-trained on ImageNet-scale datasets:
- Automatically downloaded on first use (requires internet connection)
- Cached locally for subsequent runs
- Verified using SHA256 hash checks

### Quantization Strategy

Both models use **color quantization**:
- ECCV16: 313 color classes
- SIGGRAPH17: 529 color classes
- Maps continuous color space to discrete classes
- Reduces computational complexity
- Models predict probability distribution over color classes

---

## Example Use Cases

### 1. Historical Photo Restoration
- Input: Black and white photograph from 1950s
- Output: Color version showing likely historical colors
- Use Case: Family archives, historical research

### 2. Film/Video Colorization
- Input: Frame from vintage black and white film
- Output: Color version for distribution to modern audiences
- Use Case: Archival digitization, media production

### 3. Educational Demonstration
- Input: Historical images for students
- Output: Color versions for history/art education
- Use Case: Classroom engagement, visual learning

### 4. Artistic Projects
- Input: Artistic black and white photographs
- Output: Automatic color interpretation
- Use Case: Creative exploration, art direction testing

### 5. News and Media Enhancement
- Input: Archival photographs for articles
- Output: Enhanced color versions for publications
- Use Case: Digital journalism, historical reporting

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: 
```bash
pip install streamlit==1.25.0
```

#### Issue: "CUDA out of memory" error
**Solution**: 
- The application requires ~2GB GPU memory
- Restart Streamlit or restart your system
- Use CPU instead: Comment out GPU detection in app.py
- Reduce image resolution or process smaller batches

#### Issue: Model download fails (internet connectivity)
**Solution**:
- Ensure stable internet connection on first run
- Models are cached after first successful download
- Check firewall settings blocking S3 access
- Manually download model weights and modify app.py paths

#### Issue: Application runs slowly
**Solution**:
- Verify GPU usage: `nvidia-smi` (for NVIDIA GPUs)
- GPU significantly faster than CPU (5-10x speedup)
- First run slower due to model download and compilation
- Subsequent runs are faster with cached models

#### Issue: Downloaded image shows blank/corrupted
**Solution**:
- Try different image format (PNG instead of JPEG or vice versa)
- Verify input image is valid before upload
- Check file size (very large images may have memory issues)

#### Issue: Colorization results look unnatural
**Solution**:
- This is expected for some image types:
  - Try the other model (ECCV16 vs SIGGRAPH17)
  - Results vary based on ImageNet training distribution
  - Complex or unusual content may produce unexpected colors
  - Compare side-by-side with reference images

#### Issue: Streamlit app won't start
**Solution**:
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Run with verbose logging
streamlit run app.py --logger.level=debug
```

---

## Dependencies

### Python Version
- Python 3.8 - 3.11 (recommended 3.9+)

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.25.0 | Web application framework |
| torch | 2.0.1 | Deep learning framework |
| torchvision | 0.15.2 | Computer vision utilities |
| numpy | 1.24.4 | Numerical computing |
| Pillow | 9.5.0 | Image I/O and manipulation |
| scikit-image | 0.21.0 | Image processing algorithms |
| matplotlib | 3.7.2 | Visualization (optional) |

### Installation Command
```bash
pip install streamlit==1.25.0 torch==2.0.1 torchvision==0.15.2 numpy==1.24.4 Pillow==9.5.0 scikit-image==0.21.0 matplotlib==3.7.2
```

### Optional Dependencies

- **NVIDIA CUDA Toolkit**: For GPU acceleration (NVIDIA GPUs only)
- **cuDNN**: CUDA Deep Neural Network library (included in PyTorch)

---

## License & Attribution

### Project Credits

**VividTones** implements research and models from:

1. **ECCV 2016 Colorization**
   - Paper: "Colorful Image Colorization" by Richard Zhang, Phillip Isola, Alexei A. Efros
   - Reference: ECCV 2016
   - Model weights provided by Colorizers project

2. **SIGGRAPH 2017 Colorization**
   - Paper: "Real-Time User-Guided Image Colorization with Learned Deep Priors"
   - Reference: SIGGRAPH 2017
   - Model weights provided by Colorizers project

3. **Model Weights Source**
   - Hosted by Colorizers AWS S3 Bucket
   - URLs: https://colorizers.s3.us-east-2.amazonaws.com/

### Attribution

When using VividTones in projects or publications, please consider citing:
- The original ECCV 2016 and SIGGRAPH 2017 papers
- The Colorizers project

### License

This implementation is provided for educational and research purposes. Please respect the original researchers' work and citations.

### Disclaimer

- Results are best-effort predictions and may not accurately represent original colors
- Not suitable for applications requiring perfect color accuracy
- Intended for enhancement and visualization purposes
- Users assume responsibility for appropriate usage

---

## Contact & Support

For issues, questions, or contributions:
- Repository: https://github.com/Chakrapani2122/VividTones
- Project Structure: See Project Documentation.docx
- Presentation: See vividtones.pptx
- Notebook: See Colorization.ipynb for experimental examples

---

## Changelog

### Version 1.0 (Current)
- Initial release with ECCV16 and SIGGRAPH17 support
- Streamlit web interface
- GPU acceleration support
- Download functionality
- Dual model comparison

---

**Last Updated**: 2026
**Project Status**: Active

