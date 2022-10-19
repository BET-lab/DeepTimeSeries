# DeepTimeSeries
Last update Oct.19, 2022

Deep learning library for time series forecasting based on PyTorch.
It is under development and the first released version will be announced soon.

## Why DeepTimeSeries?

DeepTimeSeries is inspired by libraries such as ``Darts`` and
``PyTorch Forecasting``. So why was DeepTimeSeries developed?


The design philosophy of DeepTimeSeries is as follows:

**We present logical guidelines for designing various deep learning models for
time series forecasting**

Our main target users are intermediate-level users who need to develop
deep learning models for time series prediction. 
We provide solutions to many problems that deep learning modelers face 
because of the uniqueness of time series data.

We additionally implement a high-level API, which allows comparatively beginners
to use models that have already been implemented.

## Supported Models

| Model         | Target features | Non-target features | Deterministic | Probabilistic |
|---------------|-----------------|---------------------|---------------|---------------|
| MLP           | o               | o                   | o             | o             |
| Vanilla RNN   | o               | o                   | o             | o             |
| LSTM          | o               | o                   | o             | o             |
| GRU           | o               | o                   | o             | o             |
| Transformer   | o               | o                   | o             | o             |

## Documentation

https://bet-lab.github.io/DeepTimeSeries/
