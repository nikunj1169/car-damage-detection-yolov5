# Car Damage Detection — YOLOv5

Automated vehicle damage detection using a custom-trained YOLOv5s model, developed during an internship at AddInn Group (Tunis, 2024–2025).

The model detects and localises damage regions in vehicle images using bounding boxes. The intended use case is a mobile application where a car owner photographs their vehicle and receives an instant assessment, reducing the time and cost of the insurance claim process.

---

## Results

| Metric | Score |
|--------|-------|
| mAP @ IoU 0.50 | 0.987 |
| mAP @ IoU 0.50:0.95 | ~0.80 |
| Precision (best threshold) | 1.00 |
| Confusion Matrix Accuracy | 0.99 |

---

## Repository Structure

```
car-damage-detection/
├── car_damage_detection.ipynb   # Full training and analysis pipeline
├── README.md
└── .gitignore
```

Model weights (`best.pt`, `last.pt`) and the dataset are not included in this repository. See the sections below for instructions on reproducing them.

---

## Requirements

- Python 3.8+
- Google Colab (recommended) or a local machine with a CUDA-capable GPU
- A Roboflow account for dataset access

All Python dependencies are handled by YOLOv5's `requirements.txt` and installed automatically in the notebook.

---

## How to Run

**1. Open the notebook in Google Colab**

Upload `car_damage_detection.ipynb` to [colab.research.google.com](https://colab.research.google.com) and set the runtime to GPU (Runtime > Change runtime type > A100 or T4).

**2. Add your Roboflow API key**

In the dataset cell, replace the placeholder with your own key:

```python
ROBOFLOW_API_KEY = "YOUR_API_KEY"
```

Your API key is available at roboflow.com under Settings > API Keys.

**3. Run all cells in order**

The notebook will handle the rest: cloning YOLOv5, downloading the dataset, training the model, evaluating it, and running inference on the test set.

---

## Dataset

- **Source:** [Roboflow Universe — car-damage-rlogo](https://universe.roboflow.com/ayhan-gul-hgudf/car-damage-rlogo/dataset/1)
- **Total images:** 9,900
- **Split:** 70% train / 20% validation / 10% test
- **Class:** `Car-Damage` (single class)
- **License:** CC BY 4.0

An initial dataset provided by AddInn was evaluated and discarded. Its validation and test splits together accounted for roughly 1% of the total images, making honest evaluation impossible and producing unstable metrics. The replacement dataset above uses a standard split and resolved the issue entirely.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv5s |
| Image size | 640 |
| Batch size | 16 |
| Epochs | 100 |
| Pretrained weights | yolov5s.pt (COCO) |
| GPU | NVIDIA A100 (Colab Pro) |
| Training time | ~1.5 hours |

---

## Notebook Overview

The notebook is structured as a complete pipeline:

1. **Environment setup** — clones YOLOv5, installs dependencies, verifies GPU
2. **Exploratory analysis** — dataset split counts, bounding box dimension distributions, annotated image samples
3. **Training** — configures and launches the YOLOv5 training script with documented parameters
4. **Results analysis** — loss curves, precision/recall/mAP over epochs, confusion matrix, PR curve
5. **Inference** — runs the best checkpoint on the test set, side-by-side ground truth vs prediction, confidence score distribution
6. **Conclusions** — summary of findings and potential next steps

---

## Author

Ilyes Khayati  
Engineering Program — MedTech, Mediterranean Institute of Technology  
Internship at AddInn Group, Tunis  
Academic Supervisor: Dr. Walid Ben Haj Othmen  
Institution Supervisor: Dr. Nivine Attoue
