# 🚗 Driver Action Recognition with Drive&Act (Inner Mirror View)

This project implements a deep learning pipeline to recognize **driver actions** using the [Drive&Act dataset](https://www.driveandact.com/), specifically leveraging the **inner mirror camera** perspective. We train a lightweight ResNet-based model to classify fine-grained actions like reaching, retracting, interacting, etc., using short video clips.

---

## 🎥 Demo

[![Driver Action Recognition Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)]([(https://www.youtube.com/shorts/r7v8o_iFcyA])
> 🔍 Inference demo: Annotated driver clips with ground truth and top-1 predicted action.

---

## 🧠 Project Highlights

- 🧾 Uses **ResNet18** on 32-frame clips
- 📼 Works on raw frames or preprocessed `.pt` tensors
- ⚖️ Handles class imbalance via `WeightedRandomSampler`
- 🧪 Inference with sliding window or per-clip prediction
- 🎞️ Annotated output video with prediction confidence

---

## 🗂 Dataset Structure

We use the *inner_mirror* camera view from the Drive&Act dataset.

```
DAD/
├── processed_frames/
│   └── interacting/vp1_0/00001.jpg ...
├── pt_clips/
│   └── reaching_for/vp2_1.pt
├── clip_index.csv  # clip_path,label
```

Supported Classes:
```
['closing', 'interacting', 'opening', 'placing_moving_to', 'reaching_for', 'retracting_from']
```

---

## ⚙️ Training Setup

```bash
# Install dependencies
pip install torch torchvision opencv-python pandas tqdm


Model: ResNet18 with `[C, T, H, W]` inputs  
Clip size: 32 frames (112x112)  
Optimizer: Adam | Loss: CrossEntropy | Balanced sampling

---

## 🎬 Inference


- Predicts clip-level or video-level driver actions  
- Annotates video with GT and predicted labels  

---

## 📊 Results

| Metric     | Value       |
|------------|-------------|
| Train Acc  | ~93%        |
| Val Acc    | ~70–75%     |
| Best Class | Interacting |
| Clip Size  | 32 frames   |

---


## 🧠 Applications

- Advanced driver-assistance systems (ADAS)
- In-cabin monitoring
- Behavior-based safety alerts
- Infotainment gesture tracking

---

## 📬 Contact

This project was built for academic + practical exploration of driver behavior.  
Questions, ideas, or collaboration? Reach out!

```diff
+ Made with 💡 using PyTorch and OpenCV
```

