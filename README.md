```
# S4Mamba
## U3PNet: Enhancing Image Fusion through Spatial-Spectral Network with Prompt Tuning for Super-Resolution

This repository presents the official implementation of **S4Mamba**, a novel approach for enhancing image fusion through a Spatial-Spectral Network with Prompt Tuning, specifically designed for Super-Resolution tasks. Our framework, **S4Mamba**, leverages advanced techniques to achieve superior performance in hyperspectral image processing.

---

## 🚀 Getting Started

### 📦 Environment Setup

To set up the required environment, we highly recommend using `conda`.

1.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n MambaHSI_env python=3.9
    conda activate MambaHSI_env
    ```

2.  **Install PyTorch and Torchvision:**
    *   **Option A (As provided in original setup commands):**
        ```bash
        conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
        ```
    *   **Option B (Recommended for compatibility with listed environment):**
        If you aim for the exact PyTorch version listed in our tested environment (`2.1.1+cu121`), please ensure your CUDA installation is compatible (e.g., CUDA 12.1).
        ```bash
        conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
        ```
        **Note:** There is a discrepancy between the PyTorch version used in the installation commands (`1.13.1`) and the one listed in our tested environment (`2.1.1`). We recommend trying `Option B` first for optimal compatibility with the project's tested setup. If you encounter issues, `Option A` might be a fallback, but ensure `mamba-ssm` compatibility.

3.  **Install Other Dependencies:**
    ```bash
    pip install packaging==24.0
    pip install triton==2.2.0
    pip install mamba-ssm==1.2.0
    pip install spectral
    pip install scikit-learn==1.4.1.post1
    pip install calflops
    ```

### ⚙️ Required Libraries

The project was developed and tested with the following key dependencies:

*   **Python:** 3.9.18
*   **PyTorch:** 2.1.1+cu121
*   **NumPy:** 1.25.2

### 📊 Datasets

To replicate our results, please download the following hyperspectral datasets:

*   **PaviaU:** [http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)
*   **Houston:** [https://hyperspectral.ee.uh.edu/?page_id=459](https://hyperspectral.ee.uh.edu/?page_id=459)
*   **IndianPines:** [https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)

Please place the downloaded datasets in the appropriate directory as expected by the `main.py` script (e.g., `./data/`).

### 🚀 Usage

Once the environment is set up and datasets are prepared, you can run the main training/evaluation script using:

```bash
python main.py
```

---

