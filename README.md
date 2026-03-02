# CODECRAFT_GA_04 — Image-to-Image Translation with cGAN (pix2pix)

> A conditional generative adversarial network (cGAN) that learns to translate architectural segmentation label maps into realistic building photographs, implemented using PyTorch in Google Colab.

---

## Overview

This project implements **pix2pix** — a paired image-to-image translation framework built on a conditional GAN (cGAN). Unlike vanilla GANs that generate images from random noise, pix2pix conditions both the Generator and Discriminator on an input image, enabling precise structured translations.

The key idea: the Generator learns a mapping from input domain X → output domain Y, while the Discriminator judges whether the output is real *given* the input condition — this is the **conditional** part.

---

## How It Works
```
Label Map → U-Net Generator → Fake Photo → PatchGAN Discriminator → Real / Fake
                                    ↕  L1 loss vs real photo
```

| Step | Description |
|------|-------------|
| **Input** | 256×256 segmentation label map (right half of paired image) |
| **Generator** | U-Net encodes input to bottleneck, decodes back with skip connections |
| **Discriminator** | PatchGAN judges 70×70 patches as real or fake (conditioned on input) |
| **Loss** | Adversarial BCE loss + L1 pixel loss (λ=100) |
| **Output** | 256×256 realistic building photograph |

### Loss Function
```
L_total = L_adversarial + λ × L_L1

L_adversarial = BCE(D(input, fake), 1)          # fool the discriminator
L_L1          = ||G(input) - target||₁ × 100   # pixel-level fidelity
```

| Component | Role |
|---|---|
| Adversarial loss | Drives realism — forces G to produce sharp, plausible images |
| L1 loss (λ=100) | Drives accuracy — keeps generated image close to ground truth |

---

## Project Structure
```
CODECRAFT_GA_04/
│
├── GA_04.ipynb    # Main Colab notebook (9 cells, fully self-contained)
└── README.md      # This file
```

---

## Getting Started

### Run in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Open `GA_04.ipynb` in Google Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells (`Runtime → Run All`)
4. Dataset downloads automatically — no manual setup needed

---

## Notebook Cells

| Cell | Description |
|------|-------------|
| **Cell 1** | Imports and GPU check |
| **Cell 2** | Auto-download facades dataset (~30 MB) with sample preview |
| **Cell 3** | Dataset class with random jitter + flip augmentation |
| **Cell 4** | U-Net Generator + PatchGAN Discriminator + weight init |
| **Cell 5** | Hyperparameters, loss functions, optimizers |
| **Cell 6** | Training loop with live inline previews every 5 epochs |
| **Cell 7** | Loss curve plots |
| **Cell 8** | Inference on full validation set (100 images) |
| **Cell 9** | Zip and download checkpoints + results |

---

## Architecture

### Generator — U-Net with Skip Connections
```
Input (3ch, 256×256)
  → e1(64) → e2(128) → e3(256) → e4(512) → e5(512) → e6(512) → e7(512) → e8(512)
                                                                                 ↓
Output (3ch, 256×256)
  ← d8(64) ← d7(128) ← d6(256) ← d5(512) ← d4(512) ← d3(512) ← d2(512) ← d1(512)
       ↑__________↑__________↑__________↑__________↑__________↑__________↑
                              skip connections (concat)
```

- 8 encoder blocks with strided convolutions (downsampling)
- 8 decoder blocks with transposed convolutions (upsampling)
- Skip connections concatenate encoder features to decoder — preserves edges and layout
- Dropout (p=0.5) on first 3 decoder blocks for regularisation
- Output activation: Tanh → pixel values in [-1, 1]

### Discriminator — 70×70 PatchGAN
```
(Input + Real/Fake concatenated, 6ch) → Conv blocks → patch score map (≈30×30 for 256×256 inputs)
```

- Each value in the 30×30 output scores one 70×70 receptive field patch
- Patch-level discrimination encourages sharper, higher-frequency texture details
- Label tensors use `torch.ones_like` / `torch.zeros_like` — robust to architecture changes

---

## Dataset

**Facades** — paired architectural images, auto-downloaded in Cell 2.

| Split | Images |
|-------|--------|
| Train | 400    |
| Val   | 100    |

Each raw image is a 512×256 side-by-side pair:
```
+-----------------+-----------------+
|   Real Photo    |   Label Map     |
|   (TARGET)      |   (INPUT)       |
+-----------------+-----------------+
     Left half         Right half
```

> **Note:** Left = real photo (target), Right = label map (input). The loader correctly handles this so the model learns label map → photo.

**Other available datasets** — change `DATASET_NAME` in Cell 2:

| Name | Translation |
|------|-------------|
| `maps` | Satellite → Road map |
| `edges2shoes` | Edge sketch → Shoe photo |
| `cityscapes` | Segmentation → Street scene |
| `night2day` | Night photo → Day photo |

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_SIZE` | 256 | Input/output resolution |
| `BATCH_SIZE` | 4 | Increase to 8 if VRAM > 8 GB |
| `EPOCHS` | 100 | Paper recommends 200 for best quality |
| `LR` | 2e-4 | Adam learning rate |
| `LAMBDA_L1` | 100 | Weight for pixel-level L1 loss |
| Adam β1 | 0.5 | Lower than default for GAN stability |
| Dropout | 0.5 | First 3 decoder blocks only |
| Augmentation | Jitter + H-flip | Random 256→286→256 crop + horizontal flip |

---

## Training Results

Trained for **100 epochs** on **Tesla T4 GPU** in ~32 minutes:

| Epoch | D Loss | G Loss | G L1 |
|-------|--------|--------|------|
| 1     | ~0.6   | ~37    | ~35  |
| 50    | ~0.3   | ~31    | ~29  |
| 100   | ~0.4   | ~29    | ~27  |

> Visualization format: Input (label map) | Generated (fake photo) | Target (real photo)

### Loss Curve Observations

- **Generator (total)** declines steadily — indicates consistent learning without collapse
- **Discriminator loss** remains low and stable — no strong oscillations observed
- **G L1 ×100** dominates early and declines gradually — reconstruction fidelity improves
- **G adversarial** stays bounded — suggests stable GAN dynamics

L1 still declining at epoch 100 indicates further improvement is possible by training to 200 epochs.

---

## Design Decisions

**Why U-Net over a plain encoder-decoder?**
Skip connections pass encoder feature maps directly to the decoder. This preserves fine spatial structure (building edges, window positions) that would otherwise be destroyed by bottleneck compression.

**Why PatchGAN over a full-image discriminator?**
Classifying patches rather than the whole image lets the discriminator focus on local texture realism. Full-image discriminators tend to produce blurry outputs because they can be fooled by globally plausible but locally smooth images.

**Why L1 loss alongside adversarial loss?**
Adversarial loss alone leads to hallucinated but structurally incorrect details. L1 anchors the output to the correct layout, while adversarial sharpens textures. λ=100 balances the two effectively.

**Dynamic label tensors:**
`torch.ones_like(pred_real)` instead of hardcoded `torch.ones(b,1,30,30)` keeps the code correct if image size or discriminator depth is changed later.

---

## Key Concepts

- **Conditional GAN** — both G and D are conditioned on an input image, not just noise
- **U-Net skip connections** — preserve spatial information across the bottleneck
- **PatchGAN** — discriminates at the patch level for sharper local textures
- **L1 + adversarial** — L1 ensures structural accuracy, adversarial ensures visual realism
- **Paired training** — requires aligned input/target pairs (unlike CycleGAN which is unpaired)

---

## References

- [Isola et al. — *Image-to-Image Translation with Conditional Adversarial Networks*](https://arxiv.org/abs/1611.07004)
- [Conditional GAN explanation — GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/conditional-generative-adversarial-network/)
- [cGAN practical guide — Medium via Scribe](https://scribe.rip/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8)

---
