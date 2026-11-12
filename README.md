Diabetes Readmission Prediction (Healthcare Data Science)

📌 Project Overview
This project focuses on predicting whether a patient with diabetes will be readmitted to the hospital within 30 days of discharge. 
Hospital readmissions are a major cost burden for healthcare systems. By identifying high-risk patients early,
hospitals can provide targeted interventions and improve patient care.

This project demonstrates an end-to-end Machine Learning pipeline, from data cleaning and exploratory data analysis (EDA) to model training and evaluation.

📊 Dataset
*   **Source:** UCI Machine Learning Repository (Diabetes 130-US Hospitals for Years 1999-2008).
*   **Size:** 101,766 patient encounters, 50 features.
*   **Target:** Readmission status (Simplified to Binary: 1 for readmitted, 0 for not readmitted).

🛠️ Technologies & Tools Used
*   **Language:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Handling Imbalanced Data:** Imbalanced-learn (SMOTE)
*   **Data Visualization:** Matplotlib, Seaborn

🚀 Methodology
1.  **Data Cleaning:** Handled missing values (replaced '?' with NaN), dropped irrelevant features, and encoded categorical variables.
2.  **Feature Engineering:** Converted medical string categories (e.g., '>200') into numerical values.
3.  **Handling Imbalanced Data:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training dataset, ensuring the model learns the minority class (readmitted patients) effectively.
4.  **Model Training:** Trained an **XGBoost Classifier**, a powerful gradient boosting algorithm known for its performance on tabular data.
5.  **Evaluation:** Evaluated the model using F1-Score, ROC-AUC, and a Confusion Matrix.

📈 Results
*   **ROC-AUC Score:** ~0.70
*   **Accuracy:** 64%
*   **Key Insight:** The Feature Importance plot revealed that the number of previous inpatient visits, diabetes medication status, and changes in medication are the strongest predictors of readmission.

💡 Key Takeaways
*   Successfully addressed the challenge of highly imbalanced healthcare data.
*   Demonstrated the ability to build interpretable ML models that provide actionable insights for clinical workflows.

*(Note: This project is for educational and portfolio purposes. The dataset is publicly available under the CC BY 4.0 license.)*
