# Installation

## Requirements
- Python 3.8+
- pip

## From Source

```bash
git clone https://github.com/yuecho1111/IceDualNet.git
cd IceDualNet
pip install -e .
```

## Verify

```python
from icesat import ICeSatClassifierNoLeak
print('Installation successful')
```

## Dependencies

See [requirements.txt](../requirements.txt):
- numpy, pandas, scikit-learn, torch, matplotlib, seaborn, tqdm

## GPU Support (Optional)

For CUDA acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
