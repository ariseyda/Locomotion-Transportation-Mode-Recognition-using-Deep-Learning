# 🚶‍♂️ Locomotion & Transportation Mode Recognition using Deep Learning

This project was developed as part of the **2021 SHL Recognition Challenge**, focused on predicting human locomotion modes using only radio frequency signals (GPS, WiFi, cell tower data). The solution leverages feature engineering and LSTM networks for time-series classification.

---

## 🧠 Project Summary

- 📌 **Goal:** Classify 8 different transportation modes (e.g., walking, biking, subway) using user-independent data sources.
- 🛰️ **Input Data:** GPS location, GPS satellite info, Wi-Fi signals, cellular tower data.
- 🔍 **Approach:** Time-series deep learning using a stacked Bi-LSTM neural network.
- 🎯 **Accuracy:** Achieved **89% average validation accuracy** across all transportation classes.

---

## 📊 Dataset

- Source: [SHL Challenge 2021 Dataset](http://www.shl-dataset.org/activity-recognition-challenge/)
- Contains time-aligned data from:
  - GPS (position, altitude, accuracy)
  - GPS satellites (azimuth, signal strength, elevation)
  - WiFi and cellular tower signals
- Labels: Still, Walking, Running, Bike, Car, Bus, Train, Subway

---

## 🛠️ Technologies Used

- Python, Pandas, NumPy
- TensorFlow, Keras (LSTM & BiLSTM models)
- Scikit-learn (evaluation & preprocessing)
- StandardScaler, Dropout, Batch Normalization
- Pittsburgh Supercomputing Center (Bridges-2) for training

---

## 🚀 Model Architecture

- 4 stacked **Bidirectional LSTM** layers
- Dropout, max-pooling & batch normalization
- Dense layer with softmax activation
- Trained with **early stopping** (patience=10)

---

## 📈 Results

- 📌 **Validation Accuracy:** 89%
- 🚲 Best classified mode: **Bike (94%)**
- 🚶‍♀️ Most challenging: **Walking (83%)**

---

## 📜 Citation

If you use this work, please cite:

> Şeyda Arı, Gulustan Dogan, Jonathan Sturdivant, Evan Kurpiewski.  
> **Locomotion-Transportation Recognition via LSTM and GPS Derived Feature Engineering from Cell Phone Data**.  
> *UbiComp-ISWC '21 Adjunct*, DOI: [10.1145/3460418.3479379](https://doi.org/10.1145/3460418.3479379)

---

## 👩‍💻 Author

**Şeyda Arı**  
AI Researcher & ML Engineer  
[LinkedIn](https://www.linkedin.com/in/seydaari) • [Medium](https://medium.com/@seydaari)
