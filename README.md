# Integrating PET/CT, Radiomics and Clinical Data: An Advanced Multi-Modal Approach for Lymph Node Metastasis Prediction
![Main Figure](./MainFig.png)

## Introduction
MM-ProLN is an artificial intelligence model that integrates PET/CT fusion images, clinical data, and radiomics features. It was developed using a combination of 3D slice-wise and 2D spatial feature extraction to provide an advanced multimodal approach for lymph node metastasis prediction.

## System Requirements
- **Python Version**: Python 3.9+ is needed to run the code.
- **Dependencies**: The necessary libraries are specified in `env.yaml`.

## Repository Structure
- `src/`: Contains the main source code.
- `data/`: Contains CSV files with clinical and radiomics data for training and testing. PET and CT images will be available after submission at [Google Drive](https://drive.google.com/). Download the images and extract them into this directory.

## Usage
Navigate to the `src` directory and use the following command to train the model with the configuration of the top-performing model:

```bash
python train.py --run_config_file ../data/config.json
```
## Additional Usage Options
For a comprehensive list of all command-line arguments and their descriptions, please refer to the [train.py](./src/train.py) file.

### Example Commands
- **Hyperparameter Search on WandB**:
  ```bash
  python train.py --wandb --project_name MMProLN-HPSearch
  ```
- **Training Without Image Data**:
  ```bash
  python train.py --image_data None --clinical_group clinical+radiomics --run_name TabularOnly --cross_validation 5 --epochs 100
  ```

## Contribution
Contributions are welcome! If you have suggestions or want to contribute to the project, please feel free to open an [issue](https://github.com/ezgiogulmus/MM-ProLN/issues) or a [pull request](https://github.com/ezgiogulmus/MM-ProLN/pulls).

## License
This project is licensed under the [MIT License](./LICENSE).

## Contact
For questions and feedback, please reach out to [ezgiogulmus@gmail.com](mailto:ezgiogulmus@gmail.com).
