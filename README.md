# stego-images-detection
Machine learning project for detecting steganography in images. Features automated preprocessing, statistical feature extraction from RGB channels, PCA visualization, and balanced dataset handling. Uses the Kaggle Stego Images Dataset. Perfect for cybersecurity and image forensics applications. Built with Python, OpenCV, and scikit-learn.

# 🔍 Stego Images Detection

A machine learning project for detecting steganography in images using feature extraction and classification techniques.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Dataset](https://img.shields.io/badge/dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project focuses on detecting hidden information in images using steganography detection techniques. Steganography is the practice of concealing messages or information within other non-secret data. This tool helps identify whether an image contains hidden data (stego) or is a clean cover image.

**Key Highlights:**
- Automated feature extraction from RGB channels
- Statistical analysis and visualization
- Balanced dataset handling with class weights
- PCA-based dimensionality reduction
- Ready for ML model integration

## 📊 Dataset

The project uses the [Stego Images Dataset](https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset) from Kaggle, which contains:
- Cover images (clean images without hidden data)
- Stego images (images with embedded hidden information)

### Dataset Structure
```
stegoimagesdataset/
├── cover/          # Original images
└── stego/          # Images with hidden data
```

## ✨ Features

### Extracted Features (per image)
For each RGB channel, the following statistics are computed:
- Mean intensity
- Standard deviation
- Minimum value
- Maximum value

**Total: 12 features per image**

### Analysis Capabilities
- Pixel intensity distribution visualization
- Feature correlation heatmap
- PCA visualization for dimensionality reduction
- Class imbalance handling with computed weights

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Google Colab (optional, for cloud execution)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stego-images-detection.git
cd stego-images-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**

The script automatically downloads the dataset using `kagglehub`. No API key configuration needed!

## 💻 Usage

### Quick Start

1. **Run the preprocessing script**
```bash
python preprocessing.py
```

This will:
- Download the dataset
- Load and preprocess images
- Extract features
- Generate visualizations
- Save train/test splits

2. **Google Colab**

Upload the notebook to Google Colab and run all cells. The dataset will be automatically downloaded.

### Output Files

After running the preprocessing script, you'll get:
- `X_train.npy` - Training features
- `X_test.npy` - Testing features
- `y_train.npy` - Training labels
- `y_test.npy` - Testing labels

### Loading Preprocessed Data

```python
import numpy as np

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
```

## 📁 Project Structure

```
stego-images-detection/
│
├── preprocessing.py           # Main preprocessing script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
│
├── notebooks/
│   └── stego_analysis.ipynb  # Jupyter notebook version
│
├── data/                     # Generated after running (gitignored)
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
│
├── visualizations/           # Sample outputs (optional)
│   ├── pixel_distribution.png
│   ├── correlation_heatmap.png
│   └── pca_visualization.png
│
└── models/                   # For future ML models
    └── README.md
```

## 📈 Results

### Sample Visualizations

**Pixel Intensity Distribution**
- Shows RGB channel intensity patterns across the dataset

**Correlation Heatmap**
- Reveals relationships between extracted features

**PCA Visualization**
- 2D projection showing class separability

### Performance Metrics
- Dataset size: Varies based on download
- Image dimensions: 256×256 pixels
- Train/Test split: 80/20
- Stratified sampling ensures balanced classes

## 🔧 Customization

### Adjust Image Size
```python
IMG_SIZE = (512, 512)  # Change from default (256, 256)
```

### Modify Feature Extraction
Add new features in the `extract_features()` function:
```python
def extract_features(img):
    feats = []
    # Add your custom features here
    return feats
```

### Change Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42  # 70/30 split
)
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contribution
- Add CNN-based detection models
- Implement additional feature extraction methods
- Create a web interface for image upload
- Add support for other steganography techniques
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- [Marco Zuppelli](https://www.kaggle.com/marcozuppelli) for providing the dataset
- Kaggle for hosting the dataset
- The open-source community for the amazing tools

## 📧 Contact

For questions or feedback, please open an issue or reach out at: your.email@example.com

## 🔗 Useful Links

- [Dataset on Kaggle](https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset)
- [Steganography Overview](https://en.wikipedia.org/wiki/Steganography)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐