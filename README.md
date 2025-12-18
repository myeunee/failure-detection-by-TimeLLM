# Predicting Resource Usage in Cloud Environments Using Trace Data

> Time-LLM + LSTM/MLP Hybrid Model for Cloud Resource Forecasting and Failure Detection

## 📌 Overview

A hybrid model combining Time-LLM with LSTM and MLP to predict cloud resource usage and detect failures. Enhances short-term pattern recognition while maintaining LLM's long-term dependency modeling capabilities.

## 🎯 Key Features

- **Hybrid Architecture**: Time-LLM + LSTM + MLP
- **Multi-Task Learning**: Memory usage prediction (regression) + failure detection (classification)
- **Datasets**: Google Cluster Trace v3, ETTh1 benchmark

## 🏗️ Model Architecture
```
Input → Patching → LSTM → Skip Connection → MLP → LLM → Output
```

- LSTM captures short-term temporal patterns
- Skip connection preserves information
- MLP aligns features with LLM embedding space

## 📚 References

- [Time-LLM (ICLR 2024)](https://openreview.net/forum?id=Unb5CVPtae)
- [Google Cluster Trace v3](https://github.com/google/cluster-data)

## 👤 Author

**Yoonji Heo**  
Computer Engineering, Kyung Hee University
