# Integrating PET/CT, Radiomics and Clinical Data: An Advanced Multi-Modal Approach for Lymph Node Metastasis Prediction in Prostate Cancer
![Main Figure](./GraphicalAbstract.png)

## Introduction
MM-ProLN is an artificial intelligence model that integrates PET/CT fusion images, clinical data, and radiomics features. It was developed using a combination of 3D slice-wise and 2D spatial feature extraction to provide an advanced multimodal approach for lymph node metastasis prediction.

## Installation

Tested on:
- Ubuntu 22.04
- Nvidia GeForce RTX 4090
- Python 3.10
- Python 3.9
- PyTorch 1.13

Clone the repository and go to the directory.

```bash
git clone https://github.com/ezgiogulmus/MM-ProLN.git
cd MM-ProLN
```

Create a conda environment and install required packages.

```bash
conda env create -n mmproln python=3.9 -y
conda activate mmproln
pip install -e .
```

## Repository Structure
- `src/`: Contains the main source code.
- `data/`: Contains CSV files with clinical and radiomics data for training and testing. PET and CT images will be available after submission at [Google Drive](https://drive.google.com/). Download the images and extract them into this directory.

## Usage
Use the following command to train the model:

```bash
python train.py 
```

For a comprehensive list of all command-line arguments and their descriptions, please refer to the [train.py](./train.py) file.

## License
This project is licensed under the [MIT License](./LICENSE).

## Contact
For questions and feedback, please reach out to [ezgiogulmus@gmail.com](mailto:ezgiogulmus@gmail.com).
