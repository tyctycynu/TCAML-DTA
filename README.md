# TCAML-DTA

**Task-Characteristic-Aware Geometric Meta-Learning for Few-Shot Drug-Target Affinity Prediction**

---

## 📋 Overview

TCAML-DTA is a geometric meta-learning framework for few-shot drug-target affinity prediction. It integrates 3D structural information of drugs and protein pockets with task-adaptive inner-loop learning rate modulation based on three task characteristics: Structural-Affinity Alignment (SAA), Distributional Novelty (DN), and Supervisory Signal Dispersion (SSD).

---

## 🖥️ System Requirements

### Recommended Configuration (Tested)

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA A800 80GB |
| CPU | Intel Xeon Gold 6348 (14 cores) |
| Memory | 100GB RAM |
| Storage | 50GB available space |
| CUDA | 12.1 |
| OS | Linux (Ubuntu 20.04/CentOS 7+) |

---

## ⚙️ Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/ljatynu/TCAML-DTA.git
cd TCAML-DTA

# Create and activate conda environment
conda create -n kdbnet python=3.10
conda activate kdbnet

# Install PyTorch
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=$PWD:$PYTHONPATH