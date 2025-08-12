# Indian-Sign-Language-ISL-Recognition
Indian Sign Language (ISL) Recognition using Python, TensorFlow, OpenCV, Keras, Machine Learning.

# Downloaded Dataset from:
https://www.kaggle.com/datasets/soumyakushwaha/indian-sign-language-dataset 


## Overview
This project implements an end-to-end deep learning pipeline to recognize 30+ Indian Sign Language (ISL) gestures. It aims to improve accessibility for hearing and speech-impaired users by translating ISL gestures into text or voice commands.
## Key Features
- Achieved **95% training accuracy** and **73% validation accuracy** across 30+ ISL gesture classes.
- Benchmarked and fine-tuned **5 Convolutional Neural Network (CNN) architectures**: Xception, InceptionV3, MobileNet, VGG16, and a combined Inception+Xception model.
- Utilized transfer learning, regularization techniques, and hyperparameter optimization (HPO) for optimal model performance.
- Implemented robust data preprocessing and augmentation to handle real-world variations such as different angles, lighting conditions, and hand positions.
- Developed a Python inference pipeline compatible with Spark/Hive environments for scalable deployment.
## Technologies Used
- Python  
- TensorFlow  
- Keras  
- OpenCV  
- Spark/Hive (for scalable inference pipeline)  
## Project Structure
- `data/` — Dataset containing ISL gesture images/videos.
- `models/` — Trained CNN model architectures and checkpoints.
- `notebooks/` — Jupyter notebooks for exploratory data analysis and model training.
- `scripts/` — Python scripts for data preprocessing, augmentation, and inference.
- `inference_pipeline/` — Spark/Hive-ready Python inference code for scalable production deployment.
## Usage
1. **Data Preparation:**  
   Run the preprocessing script to clean and augment the dataset to improve model generalization.
   ```bash
   python scripts/preprocess_data.py --input data/raw --output data/processed

2. **Training:**  
   Train models using transfer learning with the command:  
   `python scripts/train_model.py --model xception --epochs 50`

3. **Evaluation:**  
   Evaluate trained models on validation data:  
   `python scripts/evaluate.py --model models/xception.h5 --data data/processed/val`

4. **Inference:**  
   Use the scalable inference pipeline with Spark/Hive:  
   `spark-submit inference_pipeline/inference_spark.py --model models/xception.h5 --input data/test`

## Results

| Model               | Training Accuracy | Validation Accuracy |
|---------------------|-------------------|---------------------|
| Xception            | 95%               | 73%                 |
| InceptionV3         | 93%               | 70%                 |
| MobileNet           | 90%               | 68%                 |
| VGG16               | 89%               | 66%                 |
| Inception+Xception  | 96%               | 74%                 |

## Future Work

- Improve validation accuracy by collecting more diverse ISL samples.
- Deploy a real-time mobile app using the trained model.
- Extend gesture vocabulary beyond 30 signs.

## Author

**Shahaan Kaushik**

