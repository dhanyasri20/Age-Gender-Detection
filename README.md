# 🧑 Age and Gender Detection using Deep Learning

This project applies **Computer Vision** and **Deep Learning** to automatically predict a person’s **Age** and **Gender** from facial images.  
Two models were developed and compared:  
- **Custom CNN** (Convolutional Neural Network)  
- **VGG16 Transfer Learning**  

The work includes training, evaluation, visualization (accuracy/loss plots, confusion matrices), and a **research report**.

---

## 📌 Key Features
- Implementation of **CNN and VGG16** for classification
- Accuracy/Loss plots for both models
- Confusion matrices for performance analysis
- Flowchart of the methodology
- Detailed **research paper report**
- Well-documented code in Jupyter notebooks

---

## 📂 Project Structure
```plaintext
Age-Gender-Detection/
│
├── models/                  
│   ├── cnn_model.ipynb
│   ├── vgg16_model.ipynb
│
├── results/                 
│   ├── cnn_accuracy_plot.png
│   ├── cnn_loss_plot.png
│   ├── vgg16_accuracy_plot.png
│   ├── vgg16_loss_plot.png
│   ├── cnn_confusion_matrix.png
│   ├── vgg16_confusion_matrix.png
│   ├── flowchart.png
│
├── report/
│   └── Research_Paper_Report.pdf
│
├── requirements.txt         
├── README.md                
└── .gitignore               
📊 Results Summary

CNN Model Accuracy: 92.6%

VGG16 Model Accuracy: 89%
```
##  ✅ CNN outperformed the VGG16 in terms of accuracy and generalization.
📈 See full plots and confusion matrices in the results/ folder.

🚀 How to Run the Project1️⃣ Clone the Repository
git clone https://github.com/your-username/Age-Gender-Detection.git
cd Age-Gender-Detection

## 📂 Dataset
The models were trained on the [UTKFace Dataset](https://susanqq.github.io/UTKFace/)  
(contains over 20,000 face images labeled with age, gender, and ethnicity).  

👉 Download the dataset and place it inside a folder named `dataset/` before running the notebooks.

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Notebooks

Open models/cnn_model.ipynb or models/vgg16_model.ipynb in Jupyter Notebook or Google Colab

Run the cells to train/evaluate the models

📜 Research Paper

The complete research paper, including methodology, experiments, and analysis, is available in:
👉 report/Research_Paper_Report.pdf

🛠️ Tech Stack
```plaintext
Python 
TensorFlow / Keras
NumPy, Pandas
Matplotlib, Seaborn
OpenCV
Scikit-learn 
```
📌 Future Work

Extend dataset for more robust predictions
Experiment with ResNet, EfficientNet, or Vision Transformers
Deploy as a Web App or Mobile Application

🤝 Contributing

Contributions are welcome! If you’d like to improve this project:

1.Fork the repo
2.Create a new branch
3.Commit your changes
4.Open a Pull Request
```
