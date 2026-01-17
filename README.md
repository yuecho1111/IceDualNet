# IceDualNet: Sea Ice Classification

A deep learning framework for sea ice classification using ICESat-2 satellite data with dual-path neural network architecture.

[Quick Start](#-quick-start) • [Documentation](#-documentation)

## 📋 Overview

- **Data Preprocessing**: Robust handling of ICESat-2 datasets
- **Model Training**: Dual-path neural network with no data leakage
- **Inference**: Fast prediction on new ICESat-2 data

## ✨ Key Features

- No Data Leakage: Files treated as atomic units
- Dual-Path Architecture: Local + contextual waveform patterns
- Production Ready: Type hints, error handling, CI/CD
- Easy to Use: Simple API, minimal dependencies

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yuecho1111/IceDualNet.git
cd IceDualNet

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage

```python
from icesat import ICeSatClassifierNoLeak

clf = ICeSatClassifierNoLeak(epochs=20, hidden_dim=128)
clf.fit(['data/20190101.csv', 'data/20190102.csv'])
clf.save('model.pth')
predictions = clf.predict(['data/test.csv'])
```

## 📊 Project Structure

```
├── src/icesat/              # Main package
│   ├── models.py           # DualPathNetwork
│   ├── classifier.py       # ICeSatClassifierNoLeak API
│   └── data.py             # Data loading & preprocessing
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── examples/                # Example scripts
└── setup.py, requirements.txt
```

## 🏗️ Model Architecture

Dual-path neural network:
- **Local Path**: Dense layers on center point (7 features)
- **Context Path**: Conv1D layers on full sequence (7×7)
- **Fusion**: Concatenation + dense layers for prediction

## 📈 Input Features

7 features from ICESat-2/ATL07 data (background rate masked in May-August):

| # | Feature | Range | Unit |
|---|---------|-------|------|
| 0 | Photon Density | 0-100 | photons/m |
| 1 | Photon Rate | 0-100 | photons |
| 2 | Segment Height | -10 to 10 | m |
| 3 | Gaussian Width | 0-1 | - |
| 4 | Reflectance (ASR) | 0-10 | - |
| 5 | Histogram Width | 0-10 | - |
| 6 | Background Rate | 0-10⁸ | photons* |

## 🔧 Configuration

```python
ICeSatClassifierNoLeak(
    input_dim=7, sequence_length=7, hidden_dim=128,
    dropout_rate=0.3, learning_rate=0.001, batch_size=64,
    epochs=20, use_season_mask=True
)
```

## 📊 Data Format

CSV with required columns: `photon_density_per_m`, `photon_rate`, `height_segment_height`, `height_segment_w_gaussian`, `asr_25`, `hist_w`, `background_r_norm`, `classify`

Label: 0 (open water) or 1 (sea ice)

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📚 Documentation

- [QUICKSTART.md](docs/QUICKSTART.md) - Quick start guide
- [API.md](docs/API.md) - API reference
- [INSTALL.md](docs/INSTALL.md) - Installation
- [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) - Architecture

## 📊 Data Sources

- **ICESat-2/ATL07**: https://nsidc.org/data/atl07/versions/6
- **ASF**: https://asf.alaska.edu
- **Google Earth Engine**: https://earthengine.google.com/

## 🌱 Seasonal Masking

Background rate is masked during May-August to handle high solar background noise.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Push and open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
