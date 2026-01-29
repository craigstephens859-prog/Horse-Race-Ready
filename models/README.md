# Models Directory

This directory stores trained PyTorch models from the retraining process.

## Model Files

After retraining, models are saved here with timestamped filenames:
```
ranking_model_20241216_143522.pt
ranking_model_20241217_091045.pt
...
```

## Model Architecture

- **Input**: 60+ features per horse
- **Hidden Layer 1**: 128 neurons (ReLU, Dropout 0.3)
- **Hidden Layer 2**: 128 neurons (ReLU, Dropout 0.3)
- **Output**: 1 score per horse

## Model Size

~500KB per model checkpoint (lightweight and fast)

## Usage

Models are loaded automatically by the system. Manual loading:

```python
import torch
from retrain_model import RankingNN

# Load model
model = RankingNN(input_dim=60)
model.load_state_dict(torch.load('models/ranking_model_YYYYMMDD_HHMMSS.pt'))
model.eval()

# Predict
with torch.no_grad():
    scores = model(features_tensor)
```

## Retention Policy

Keep last 10 models for performance comparison. Older models can be safely deleted.
