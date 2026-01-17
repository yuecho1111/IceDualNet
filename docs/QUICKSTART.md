# Quick Start

## Install

```bash
git clone https://github.com/yuecho1111/IceDualNet.git
cd IceDualNet
pip install -e .
```

## Train

```python
from icesat import ICeSatClassifierNoLeak

clf = ICeSatClassifierNoLeak(epochs=20, hidden_dim=128)
clf.fit(['data/20190101.csv', 'data/20190102.csv'])
clf.save('model.pth')
```

## Predict

```python
clf = ICeSatClassifierNoLeak()
clf.load('model.pth')
predictions = clf.predict(['data/test.csv'])
probabilities = clf.predict_proba(['data/test.csv'])
```

## Evaluate

```python
from sklearn.metrics import f1_score
y_pred = clf.predict(test_files)
f1 = f1_score(y_true, y_pred)
```

## Data Format

CSV with columns: `photon_density_per_m`, `photon_rate`, `height_segment_height`, `height_segment_w_gaussian`, `asr_25`, `hist_w`, `background_r_norm`, `classify`

Label: 0 (open water) or 1 (sea ice)
