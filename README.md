Project Overview: DiabetesGuard is a diagnostic intelligence project focused on early diabetes detection using patient health metrics. It explores the effectiveness of various supervised learning models on a clinical dataset containing 100,000 records.

Technical Workflow & Implementation:

Exploratory Data Analysis (EDA): Leveraged Seaborn and Matplotlib to analyze the distribution of key features such as BMI, HbA1c levels, and Blood Glucose.

Data Engineering: Implemented categorical encoding for features like gender and smoking history, and split data for robust training and testing.

Model Benchmarking: Implemented and compared six core algorithms:

SVM, KNN, Random Forest

Gradient Boosting, Gaussian Naive Bayes, and Decision Trees

Key Results & Performance:

Top Performer: Gradient Boosting achieved an impressive accuracy of 97.26%, showing superior precision in identifying high-risk patients.

Robust Metrics: Evaluated models using detailed Classification Reports (Precision, Recall, F1-score) and Confusion Matrices to ensure reliability in medical contexts.

Predictive Power: Demonstrated high recall for critical features like HbA1c and glucose levels, essential for early intervention.

Technical Stack:

Languages: Python.

Libraries: Pandas, Scikit-learn, Seaborn, Matplotlib.

Platform: Developed using Jupyter Notebooks for transparent research and documentation.
