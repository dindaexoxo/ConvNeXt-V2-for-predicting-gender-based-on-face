# Gender Classification Using ConvNeXt V2
Welcome to the repository for Gender Classification using the **ConvNeXt V2** model! My name is **Adinda Ayu Lestari**, you can call me **Dinda**. In this project, we use a deep learning model to classify images into two categories: **female** and **male**. The ConvNeXt V2 architecture provides high accuracy while being efficient for practical use, making it suitable for tasks like gender classification.
## Overview
This project demonstrates the complete pipeline for training, validating, and evaluating a deep learning model for binary image classification. The dataset used is sourced from Kaggle, and the ConvNeXt V2 model is utilized for its efficient learning and impressive performance.
## Project Structure
```
ConvNeXt-V2-for-predicting-gender-based-on-face/
|
├── README.md                  						# Description of the project
├── LICENSE                    						# Licensing information
├── requirements.txt           						# Python dependencies
└── ConvNeXt V2 for predicting gender based on face.ipynb          	# Code script
```
## Getting Started

### Prerequisites

To run this project, you need to have the following installed:

- Python 3.7+
- PyTorch
- Torchvision
- timm library
- Other dependencies: `scikit-learn`, `matplotlib`, `seaborn`, `Pillow`


You can install all the required libraries by running:

```bash
pip install -r requirements.txt
```
### Cloning the Repository
Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/dindaexoxo/ConvNeXt-V2-for-predicting-gender-based-on-face.git
```
Navigate to the project directory:

```bash
cd ConvNeXt-V2-for-predicting-gender-based-on-face
```
## Dataset
The dataset used for this project is the **Gender Classification Dataset** from Kaggle, which contains labeled images of male and female faces.
## Model Architecture
We use **ConvNeXt V2**, a modern deep learning architecture designed for high efficiency and performance. We utilize the `timm` library to load the model and fine-tune it for our classification task.
The model has been modified by replacing the final classifier layer to accommodate two classes (‘female’ and ‘male’). The rest of the pre-trained weights are retained to benefit from transfer learning.
## Training Process
The training process involves fine-tuning the ConvNeXt V2 model on our dataset. The dataset is split into **80% training** and **20% validation**. We use **Cross Entropy Loss** as the loss function and **Adam** as the optimizer with a learning rate of `1e-4`.
## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
## Acknowledgments
- The **Gender Classification Dataset** is provided by Kaggle. Please check the dataset page for appropriate attribution.
- The **timm library** by Ross Wightman for providing easy access to pre-trained models.
- **OpenAI's ChatGPT** for helping structure this project.

Feel free to reach out for any questions or suggestions regarding this project!

## Notes
Please download the .ipynb file if you want to see the full output, because when I tried to open it on Github, it couldn't show.

