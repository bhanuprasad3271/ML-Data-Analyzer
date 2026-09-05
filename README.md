ML Data Analyzer

Upload • Analyze • Train • Compare • Predict

ML Data Analyzer is a Python-based automated machine learning application built with Streamlit. It allows users to upload a dataset, select a target column, automatically detect the ML problem type, train multiple models, compare their performance, and make predictions without writing ML code.

✨ Features

- 📂 Supports CSV, Excel and JSON datasets
- 🎯 Target column selection
- 🤖 Automatic Classification / Regression detection
- 🧹 Handles missing values and numeric values stored as text
- 🔤 Supports numerical and categorical features
- ⚙️ Multiple algorithm selection
- 📊 Model performance comparison
- 🏆 Automatic best-model selection
- 🔮 Prediction using the trained model
- 📈 Classification and regression evaluation metrics

🤖 Algorithms

Classification

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

Regression

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression (SVR)

📊 Evaluation Metrics

Classification

- Accuracy
- Precision
- Recall
- F1 Score

Regression

- R² Score
- RMSE

🔄 Workflow

Upload Dataset
      ↓
Select Target
      ↓
Automatic Problem Detection
      ↓
Select Algorithms
      ↓
Train & Test
      ↓
Compare Models
      ↓
Best Model
      ↓
Make Prediction

🛠️ Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- OpenPyXL

📁 Project Structure

ML-Data-Analyzer/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

🚀 Run Locally

pip install -r requirements.txt
streamlit run app.py

🌐 Deployment

The application can be deployed using Streamlit Community Cloud directly from GitHub.

🔮 Future Enhancements

- Cross-validation
- Confusion matrix
- Feature importance
- Hyperparameter tuning
- Model download
- Automated reports
- More ML algorithms

👨‍💻 Author

Bhanu Prasad
