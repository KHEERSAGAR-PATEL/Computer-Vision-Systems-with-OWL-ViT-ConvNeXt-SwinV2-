# 🧠 Vision-Based AI Projects — Applied Scientist Portfolio  
**By Kheer Sagar Patel | M.Tech (AI & ML), IIITDM Jabalpur**

Welcome to a curated showcase of high-impact computer vision projects leveraging state-of-the-art transformer architectures and deep convolutional networks. These implementations target real-world challenges spanning medical imaging, scene classification, and zero-shot object detection — showcasing domain diversity, model innovation, and performance-driven ML engineering.

> 📌 All models were trained on Kaggle’s high-performance GPU infrastructure, and trained checkpoints are saved for reproducibility and future inference.

---

## 🔍 Projects Overview

### 1. 🖼️ ConvNeXt for Scene Classification
**Objective:** Classify real-world natural scenes from the Intel Image Classification dataset using ConvNeXt — a modernized ResNet with Swin-Transformer-inspired improvements.

- **Model:** ConvNeXt-Tiny with DropPath regularization and custom training loop.  
- **Dataset:** Intel Natural Scenes (6 classes: buildings, forests, mountains, etc.).  
- **Techniques:** Advanced augmentation (RandAugment, RandomErasing), Cosine Learning Rate Decay, mixed precision training.  
- **Performance:**  
  - ✅ **Achieved 58.7% Top-1 Accuracy**  
  - 🧠 DropPath enhanced robustness to occlusion and image noise.  
- **Applications:** Autonomous vehicles, drone surveillance, AR/VR scene understanding.  

---

### 2. 🧠 SwinV2 Transformer for Brain MRI Segmentation
**Objective:** Segment tumor regions from Brain MRI scans using a hierarchical vision transformer — SwinV2 — optimized for pixel-level segmentation.

- **Model:** SwinV2 U-Net hybrid with custom decoder blocks and patch merging.  
- **Dataset:** BraTS Brain MRI Dataset (FLAIR modality).  
- **Techniques:** Multi-head self-attention, skip connections, and patch shifting.  
- **Performance:**  
  - 🎯 **Achieved IoU ≈ 0.85**  
  - 🚀 Inference-ready model with TensorRT optimizations.  
- **Applications:** Clinical diagnostics, radiology automation, precision healthcare.

---

### 3. 🧭 Open World Localization using OWL-ViT
**Objective:** Build a real-time, zero-shot object detection pipeline using OWL-ViT from Google DeepMind.

- **Model:** OWL-ViT (Open-World Localization Vision Transformer)  
- **Use Case:** Detect arbitrary objects from user text prompts without additional training.  
- **Techniques:** Vision-text contrastive pretraining, CLIP-style embeddings, prompt engineering.  
- **Performance:**  
  - 🔍 Strong generalization to unseen object categories using language prompts  
  - 🕒 **Real-time inference (30+ FPS)** using ONNX optimization  
- **Applications:** Robotics, autonomous navigation, industrial automation.

---

## 💼 Why These Projects Matter

- ✅ **Generalization Focus:** Covers medical AI, zero-shot reasoning, and real-world scene understanding.  
- 🧪 **SOTA Architectures:** Uses ConvNeXt, SwinV2, OWL-ViT — latest in transformer vision.  
- 📊 **Metrics-Driven:** Evaluated with IoU, Accuracy, FPS, and Dice score.  
- ⚙️ **Production Ready:** Models trained on Kaggle GPU, saved for reproducibility and future deployment.

---

## 🛠️ Tech Stack

- **Languages:** Python, PyTorch, TensorFlow  
- **Frameworks:** HuggingFace, timm, torchvision, MONAI  
- **Tools:** ONNX, TensorRT, JupyterLab  
- **Hardware:** NVIDIA T4/V100 on Kaggle  

---

## 📁 Repository Structure

```
├── ConvNeXt_SceneClassification.ipynb
├── SwinV2_BrainMRI_Segmentation.ipynb
├── OWLViT_RealTime_ObjectDetection.ipynb
├── assets/
│   └── sample_outputs/
├── README.md
└── requirements.txt
```
---
## 🚀 How to Run

```bash
# Create environment and install dependencies
pip install -r requirements.txt

# Or run each notebook individually in Colab (recommended for GPUs)
```

---

## 📣 Let's Connect

These projects reflect my domain versatility, applied research capabilities, and end-to-end ML engineering skills. I’m actively exploring opportunities in:

- 🧠 Applied ML Research  
- 🧬 Medical AI and Radiology Tech  
- 🤖 Vision & Robotics Engineering  
- 🔍 LLM + Vision Integrations (Multimodal AI)

Feel free to connect via [LinkedIn](https://linkedin.com/in/kheer-sagar-patel-7b7431187) or explore more on my [GitHub](https://github.com/KHEERSAGAR-PATEL).



---

## ❤️ Made with love by Kheer Sagar Patel
## Mtech CSE (AI & ML) IIITDM Jabalpur
