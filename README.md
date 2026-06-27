# 🧠 DermAI – Multimodal Skin Lesion Classification

> **A Vision Transformer–based multimodal deep learning system for automated skin lesion classification using dermoscopic images and patient metadata.**

<p align="center">

## 🚀 Live Demo

### **https://chandu06-dermai.hf.space**

[🌐 Launch DermAI](https://chandu06-dermai.hf.space)

---

📓 **Research Notebook** • 🤗 **Hugging Face Model** • 💻 **GitHub Repository**

</p>

<p align="center">

[🚀 Live Demo](https://chandu06-dermai.hf.space) •
[📓 Research Notebook](https://www.kaggle.com/) •
[🤗 Hugging Face](https://huggingface.co/spaces/Chandu06/dermai)

</p>

---

## Dashboard

<p align="center">
<img src="assets/dashboard.png" width="100%">
</p>

---

## Overview

Early identification of skin cancer can significantly improve treatment outcomes. DermAI explores how modern Vision Transformers can assist clinicians by combining **dermoscopic images** with **patient metadata** to classify common skin lesions.

Unlike conventional image-only classifiers, DermAI integrates both visual and clinical information to provide richer representations of skin lesions through a clean, intuitive interface. The project demonstrates a complete AI workflow—from model development and evaluation to cloud deployment—making the research accessible through an interactive web application.

---

## Features

* 🧠 Vision Transformer (ViT-B16) backbone
* 📷 Dermoscopic image classification
* 👤 Multimodal prediction using patient metadata
* 📊 Interactive probability visualization
* 🎯 Seven-class skin lesion classification
* ☁️ Live cloud deployment with Streamlit
* 🤗 Hugging Face hosted demo

---

## Supported Lesion Classes

| Class    | Description                                   |
| -------- | --------------------------------------------- |
| AKIEC    | Actinic Keratoses / Intraepithelial Carcinoma |
| BCC      | Basal Cell Carcinoma                          |
| BKL      | Benign Keratosis-like Lesions                 |
| DF       | Dermatofibroma                                |
| Melanoma | Malignant Melanoma                            |
| Nevus    | Melanocytic Nevus                             |
| VASC     | Vascular Lesions                              |

---

## Model Architecture

* **Backbone:** Vision Transformer (ViT-B16)
* **Pretraining:** ImageNet-21k
* **Input Resolution:** 224 × 224
* **Framework:** TensorFlow / Keras
* **Learning Strategy:** Transfer Learning + Fine-Tuning
* **Inference:** Multimodal (Image + Metadata)

---

## Patient Metadata

Alongside dermoscopic images, the model utilizes basic clinical information:

* Age
* Sex
* Lesion Localization

This multimodal approach enables the network to incorporate contextual patient information in addition to visual features.

---

## Performance

| Metric              |      Score |
| ------------------- | ---------: |
| Validation Accuracy | **83.04%** |
| Weighted F1 Score   |   **0.83** |
| Macro F1 Score      |   **0.70** |
| Melanoma Precision  |   **0.56** |
| Melanoma Recall     |   **0.64** |
| Melanoma F1 Score   |   **0.60** |

---

## Technology Stack

* Python
* TensorFlow
* Keras
* KerasHub
* Vision Transformer (ViT-B16)
* Streamlit
* Hugging Face Spaces
* NumPy
* Pandas
* Scikit-learn
* Pillow

---

## Project Structure

```text
DermAI/
│
├── app.py
├── style.css
├── requirements.txt
│
├── configs/
│   └── class_mapping.json
│
├── utils/
│   ├── image_preprocessing.py
│   ├── inference.py
│   └── metadata_encoder.py
│
├── assets/
│   └── dashboard.png
│
└── README.md
```

---

## Running Locally

Clone the repository

```bash
git clone https://github.com/chandu2006-git/cancer-detection-model.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Live Demo

🚀 **Try DermAI online**

**https://chandu06-dermai.hf.space**

---

## Research Impact

DermAI demonstrates how multimodal deep learning can be applied to dermatological image analysis by combining computer vision with patient metadata in a unified inference pipeline.

While intended as a research and educational project, it showcases a complete machine learning workflow including data preparation, model development, evaluation, deployment, and interactive visualization.

---

## Disclaimer

This application is provided **for research and educational purposes only**.

It is **not** a medical device and should **not** be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

---

## Author

**Chandu**

Artificial Intelligence • Deep Learning • Computer Vision • Medical AI

If you found this project interesting, consider giving the repository a ⭐.
