# DeepTimeSeries

Deep learning library for time series forecasting based on PyTorch and PyTorch Lightning.

## Overview

DeepTimeSeries is a comprehensive library designed for time series forecasting using deep learning models. It provides a logical framework for designing and implementing various deep learning architectures specifically tailored for time series data.

The library is built on PyTorch and PyTorch Lightning, offering both high-level APIs for beginners and flexible low-level components for intermediate to advanced users who need to customize their forecasting models.

## Key Features

- **Modular Architecture**: Clean separation between encoding, decoding, and prediction components
- **Multiple Model Support**: Pre-implemented models including MLP, Dilated CNN, RNN variants (LSTM, GRU), and Transformer
- **Flexible Data Handling**: Pandas DataFrame-based data processing with chunk-based extraction
- **Data Preprocessing**: Built-in `ColumnTransformer` for feature scaling and transformation
- **Probabilistic Forecasting**: Support for both deterministic and probabilistic predictions
- **PyTorch Lightning Integration**: Seamless integration with Lightning for training, validation, and testing

## Installation

The project uses `pyproject.toml` for dependency management. You can install it using:

```bash
# Using pip
pip install .

# Using uv (recommended)
uv pip install .

# For development with dev dependencies
uv pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10, < 4.0
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- NumPy >= 1.24.2
- Pandas >= 1.5.3
- XArray >= 2023.2.0

See `pyproject.toml` for the complete list of dependencies.

## Quick Start

```python
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import deep_time_series as dts
from deep_time_series.model import MLP
from sklearn.preprocessing import StandardScaler

# Prepare data
data = pd.DataFrame({
    'target': np.sin(np.arange(100)),
    'feature': np.cos(np.arange(100))
})

# Preprocess data
transformer = dts.ColumnTransformer(
    transformer_tuples=[
        (StandardScaler(), ['target', 'feature'])
    ]
)
data = transformer.fit_transform(data)

# Create model
model = MLP(
    hidden_size=64,
    encoding_length=10,
    decoding_length=5,
    target_names=['target'],
    nontarget_names=['feature'],
    n_hidden_layers=2,
)

# Create dataset and dataloader
dataset = dts.TimeSeriesDataset(
    data_frames=data,
    chunk_specs=model.make_chunk_specs()
)
dataloader = DataLoader(dataset, batch_size=32)

# Train model
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
```

## Supported Models

| Model | Target Features | Non-target Features | Deterministic | Probabilistic |
|-------|----------------|---------------------|---------------|---------------|
| MLP | ✓ | ✓ | ✓ | ✓ |
| Dilated CNN | ✓ | ✓ | ✓ | ✓ |
| Vanilla RNN | ✓ | ✓ | ✓ | ✓ |
| LSTM | ✓ | ✓ | ✓ | ✓ |
| GRU | ✓ | ✓ | ✓ | ✓ |
| Transformer | ✓ | ✓ | ✓ | ✓ |

## Project Structure

```
deep_time_series/
├── core.py          # Core modules: ForecastingModule, Head, BaseHead, etc.
├── chunk.py         # Chunk specification and extraction utilities
├── dataset.py       # TimeSeriesDataset implementation
├── transform.py     # ColumnTransformer for data preprocessing
├── plotting.py      # Visualization utilities
├── layer.py         # Custom neural network layers
├── util.py          # Utility functions
└── model/           # Pre-implemented forecasting models
    ├── mlp.py
    ├── dilated_cnn.py
    ├── rnn.py
    └── single_shot_transformer.py
```

## Core Concepts

### Chunk Specification

The library uses a chunk-based approach for handling time series data:
- **EncodingChunkSpec**: Defines the input window for the encoder
- **DecodingChunkSpec**: Defines the input window for the decoder
- **LabelChunkSpec**: Defines the target window for prediction

### Forecasting Module

All models inherit from `ForecastingModule`, which provides:
- Automatic training/validation/test step implementations
- Metric tracking and logging
- Loss calculation with multiple heads
- Chunk specification generation

### Data Flow

1. Load data as pandas DataFrame
2. Apply preprocessing with `ColumnTransformer`
3. Create `TimeSeriesDataset` with chunk specifications
4. Train model using PyTorch Lightning Trainer
5. Use `ChunkInverter` to convert model outputs back to DataFrame format

## Documentation

Full documentation is available at: https://bet-lab.github.io/DeepTimeSeries/

The documentation includes:
- User Guide: Design concepts and usage patterns
- Tutorials: Step-by-step examples
- API Reference: Complete API documentation

## Design Philosophy

DeepTimeSeries is designed with the following principles:

**We present logical guidelines for designing various deep learning models for time series forecasting**

The library targets intermediate-level users who need to develop deep learning models for time series prediction. It provides solutions to common problems that arise from the unique characteristics of time series data, such as:

- Handling variable-length sequences
- Managing encoding and decoding windows
- Supporting both target and non-target features
- Enabling probabilistic forecasting

Additionally, the high-level API allows beginners to use pre-implemented models with minimal configuration.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License

## Authors

- Sangwon Lee 
- Wonjun Choi