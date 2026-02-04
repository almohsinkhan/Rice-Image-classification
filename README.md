# 🍚 Rice Type Classification using CNN

This project focuses on classifying rice grain images into different varieties using a Convolutional Neural Network (CNN). The model is trained on the Kaggle Rice Image Dataset and demonstrates strong performance in distinguishing visually similar rice types.

📌 Dataset

Source: Kaggle – Rice Image Dataset

The dataset contains five different rice varieties:

🌾 Arborio

🌾 Basmati

🌾 Ipsala

🌾 Jasmine

🌾 Karacadag

Each class consists of high-quality rice grain images captured under controlled conditions, making it suitable for image classification tasks.

## 🧠 Model Architecture

A custom CNN-based classifier was designed and trained from scratch.

Architecture Overview:

Multiple Convolutional layers with ReLU activation

Batch Normalization for stable training

MaxPooling layers for spatial downsampling

Adaptive Average Pooling

Fully Connected (Linear) layer for final classification

Model Summary:

Total Parameters: 94,341

Trainable Parameters: 94,341

Output Classes: 5

The final output layer uses Softmax (via CrossEntropyLoss) for multi-class classification.

## ⚙️ Training Details

Framework: PyTorch

Loss Function: CrossEntropyLoss

Optimizer: Adam

Batch Size: 32

Epochs: 5

Input Image Size: 224 × 224

## 📊 Results & Performance
Loss Curve

Training loss steadily decreased across epochs

Test loss showed initial fluctuation but stabilized with training

Accuracy Curve

Training Accuracy: ~98%

Test Accuracy: ~93–95%

These results indicate good generalization with minimal overfitting.

## 📈 Visualizations

<img width="1294" height="497" alt="image" src="https://github.com/user-attachments/assets/c4ea8f0d-d618-44cd-b984-d867aedac47f" />


Plots clearly show model convergence and improved performance over epochs.

## 🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/rice-classification-cnn.git
cd rice-classification-cnn


Install dependencies:

pip install torch torchvision matplotlib torchinfo


Train the model:

python train.py


Evaluate / Load trained model:

model.load_state_dict(torch.load("models/model_0.pth"))
model.eval()

📂 Project Structure
.
├── dataset/
│   ├── Arborio/
│   ├── Basmati/
│   ├── Ipsala/
│   ├── Jasmine/
│   └── Karacadag/
├── models/
│   └── model_0.pth
├── train.py
├── evaluate.py
├── utils.py
├── README.md

## 🧪 Future Improvements

Data augmentation for better generalization

Try deeper architectures (ResNet, EfficientNet)

Hyperparameter tuning

Deployment as a web application (Django / FastAPI)

🏁 Conclusion

This project demonstrates how CNNs can effectively classify rice varieties based on visual features. Despite the similarity between rice grains, the model achieves high accuracy, proving the effectiveness of deep learning in agricultural image analysis
