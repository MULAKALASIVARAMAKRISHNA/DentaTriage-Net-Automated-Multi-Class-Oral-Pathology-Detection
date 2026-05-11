# DentaTriage-Net-Automated-Multi-Class-Oral-Pathology-Detection

**Domain:** Medical Imaging / Computer Vision / Healthcare AI

**Tech Stack:** Python, TensorFlow, Keras, EfficientNetB0, OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib

---

## 1. Executive Summary: Scalable Medical Inference

**DentaTriage-Net** is an end-to-end computer vision solution engineered for **High-Throughput Screening** and **Automated Triage** of oral pathologies. By leveraging **State-of-the-Art (SOTA)** Deep Learning architectures, the system classifies six clinically significant conditions: Caries, Calculus, Gingivitis, Ulcers, Tooth Discoloration, and Hypodontia. The project addresses the "Data Velocity" problem in healthcare—where the volume of intraoral imagery far exceeds the human capacity for real-time review. Designed for population-scale health agencies, DentaTriage-Net identifies urgent cases from datasets of 10,000+ images, reducing clinical overhead and minimizing **Diagnostic Variability**.

## 2. Problem Statement: Mitigating Clinical Decision Fatigue

In large-scale screening programs, the primary challenge is not the complexity of a single image, but the **Long-Tail Distribution** of disease and the resulting **Decision Fatigue**. When a clinician manually reviews 10,000 images, their sensitivity (Recall) naturally drops due to repetitive stress, leading to **False Negatives** in critical early-stage pathologies. DentaTriage-Net serves as a **Decision Support System (DSS)** that ensures a consistent, mathematically objective baseline across any dataset volume, effectively "funneling" only the most statistically significant cases to the practitioner.

## 3. The Technical Engine: Architectural Selection & Optimization

### EfficientNetB0 Backbone & Compound Scaling

The model utilizes **EfficientNetB0**, selected for its superior **Floating Point Operations (FLOPs)** efficiency. Unlike standard CNNs (like ResNet or VGG), EfficientNet uses **Compound Scaling** to balance depth, width, and resolution. This ensures the model is lightweight enough for **Edge Deployment** (on mobile dental cameras) while maintaining high **Representational Power** for subtle textures like enamel pitting or mucosal inflammation.

### Strategic Transfer Learning & Fine-Tuning

A sophisticated **Two-Phase Transfer Learning** strategy was implemented to leverage **Knowledge Distillation** from ImageNet:

* **Phase 1: Feature Extraction:** The base model was frozen to preserve general-purpose spatial hierarchies, while a custom **Global Average Pooling** and **Dropout-regulated** dense head were trained on the dental domain.
* **Phase 2: Layer-wise Fine-Tuning:** The top 30 layers were unfrozen for specialized training using a **Differential Learning Rate** ($1 \times 10^{-5}$), allowing the model to adapt its filters to the high-frequency details of intraoral pathologies.

## 4. The Data Pipeline: Engineering for Medical Reliability

### Automated Data Wrangling & Deduplication

The project involved a complex **Recursive Image Parsing** engine to handle unstructured, nested Kaggle directories. Key skills applied:

* **Path-based Metadata Extraction:** Mapping inconsistent folder names to a unified **Clinical Ontology**.
* **Data Deduplication:** Implementing a **Hashed-Content Check** to remove redundant images across "train/val/done" folders that often plague Kaggle datasets.
* **Stratified Splitting:** Ensuring that the 70/15/15 split maintained identical class distributions to prevent **Bias-Variance** issues in rare classes like Hypodontia.

### Advanced Image Augmentation

To improve **Model Generalization** against varying camera angles and lighting conditions (common in intraoral photography), a robust **Augmentation Pipeline** was built using:

* **Geometric Transformations:** Random rotations, zooms, and horizontal flips.
* **Photometric Adjustments:** Brightness and contrast shifts to simulate different dental chair lighting environments.

## 5. Evaluation Metrics: The Reliability Standard

### Why Macro F1-Score is the Primary KPI

In medical imaging, **Accuracy is a Trap** due to **Class Imbalance**. If 80% of images are "Normal," a broken model that only predicts "Normal" would still have 80% accuracy. DentaTriage-Net is evaluated using the **Weighted Macro F1-Score**.

* **Precision vs. Recall:** While **Precision** ensures dentists don't get "spammed" with False Alarms, **Recall (Sensitivity)** is prioritized to ensure no genuine pathology is missed.
* **F1-Score** provides the harmonic mean, proving the model is a reliable tool for both common diseases (Caries) and rare ones (Ulcers).

## 6. Real-World Impact: Population Health Triage

For an agency managing 10,000 images, the deployment of DentaTriage-Net provides:

1. **Computational Triage:** Reducing the "Review Pool" from 10,000 to the ~1,000 most suspicious images.
2. **Resource Allocation:** Mapping disease prevalence across geographic sectors using AI-generated metadata.
3. **Patient Trust:** Providing a "Visual Second Opinion" with **Confidence Intervals**, which has been shown to increase patient compliance for expensive procedures.


## 🔍 Model Predictions & Results

Below are sample outputs from **DentaTriage-Net**, demonstrating its ability to detect and classify various oral pathologies with high confidence.

### Dental Caries
![Caries Prediction](images/Caries.PNG)
> **Prediction:** Caries | **Confidence:** 99.42%

### Tooth Discoloration
![Tooth Discoloration Example 1](images/Discoloration.PNG)
> **Prediction:** Tooth_Discoloration | **Confidence:** 76.23%

![Tooth Discoloration Example 2](images/tooth_discoloration.PNG)
> **Prediction:** Tooth_Discoloration | **Confidence:** 97.60%

### Hypodontia
![Hypodontia Prediction](images/hypondotia.PNG)
> **Prediction:** Hypodontia | **Confidence:** 99.87%

### Mouth Ulcer
![Ulcer Prediction](images/Ulcer.PNG)
> **Prediction:** Ulcer | **Confidence:** 72.94%
## 7. Conclusion: The Future of Dental AI

DentaTriage-Net isn't just a classifier; it’s a **Scalable Inference Pipeline**. By utilizing **Global Pooling, Dropout Regularization, and SOTA Transfer Learning**, the project demonstrates the capability to handle real-world clinical data complexity. It stands as a blueprint for how AI can transform dental healthcare from reactive treatment to **Proactive, Population-Scale Screening**.

---
