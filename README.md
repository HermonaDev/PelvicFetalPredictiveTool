Pelvic Fetal Predictive Interactive Tool
Project Overview
The Pelvic Fetal Predictive Interactive Tool is a web-based application designed to assist clinicians in predicting delivery outcomes based on maternal pelvic and fetal biometric data. Using machine learning, the tool predicts the likelihood of successful vaginal delivery or the need for cesarean section, focusing on key metrics like pelvic inlet diameter and fetal head circumference. The tool provides interpretable results through visualizations and explainable AI, ensuring clinician trust and usability. Built with Python and Streamlit, this project demonstrates end-to-end AI development skills, from data processing to model deployment, with an emphasis on ethical AI practices for healthcare applications.
This project was developed as a portfolio piece to showcase AI and software engineering skills for a health AI lab role, emphasizing practical, clinician-facing solutions.
Objectives

Clinical Utility: Provide a user-friendly tool for obstetricians to assess delivery risks based on pelvic and fetal measurements.
Ethical AI: Incorporate responsible AI practices, such as using synthetic data to avoid privacy concerns and addressing model bias.

Functional Requirements

Input Interface: Allow users (e.g., clinicians) to input maternal and fetal data via a web form, including:
Pelvic measurements (e.g., inlet/outlet diameter in cm).
Fetal biometrics (e.g., head circumference, estimated weight).
Additional factors (e.g., maternal age, parity).


Prediction Engine: Use a machine learning model (e.g., logistic regression or random forest) to predict:
Probability of successful vaginal delivery.
Risk score for complications (e.g., cesarean need, shoulder dystocia).


Explainability: Provide feature importance (e.g., via SHAP) to explain why predictions were made (e.g., "Narrow pelvic outlet contributed 40% to cesarean risk").
Visualization: Display results as:
Risk probability bar chart or heatmap.
Optional 2D/3D schematic of pelvic-fetal interaction (if time permits).


Output: Generate a concise report summarizing predictions and key factors for clinicians.

Non-Functional Requirements

Performance: Deliver predictions in <5 seconds for real-time use.
Accuracy: Achieve >80% AUC-ROC on validation data, benchmarked against medical standards.
Usability: Intuitive Streamlit-based web interface, mobile-responsive, with clear instructions.
Privacy: Use synthetic or public datasets to avoid real patient data, ensuring HIPAA-like compliance in design.
Scalability: Modular code structure for future enhancements (e.g., adding genetic or real-time sensor data).
Portability: Runnable locally or deployable via Docker for easy sharing.

Technical Stack

Language: Python 3.8+
Machine Learning: Scikit-learn for classification, SHAP for explainability.
Data Processing: Pandas, NumPy for data handling; Faker or NumPy for synthetic data generation.
Visualization: Plotly or Matplotlib for charts; (optional) VTK or Plotly for 3D visuals.
Web Framework: Streamlit for rapid UI development.
Version Control: Git/GitHub for repository management.
Environment: Virtualenv or Conda for dependency management; Docker for deployment.

Data Sources

Primary: Synthetic dataset generated using NumPy/Faker, simulating pelvic measurements (e.g., inlet: 10-14 cm, outlet: 8-12 cm), fetal biometrics (e.g., head circumference: 30-36 cm), and outcomes (e.g., vaginal vs. cesarean).
Secondary: Public dataset (e.g., Fetal Health Classification from Kaggle) for model training if time permits.
Ethical Note: No real patient data is used to ensure privacy and compliance with medical ethics.


Ethical Considerations

Data Privacy: Uses synthetic data to avoid handling sensitive medical records, aligning with HIPAA principles.
Bias Mitigation: Ensures diverse synthetic data (e.g., varied maternal ages, pelvic shapes) to reduce model bias.
Transparency: Includes explainable AI (SHAP) to build clinician trust and avoid black-box predictions.
Scope: Designed for educational/portfolio purposes, not clinical use, to prevent misapplication.

How to Run

Clone the repository: git clone https://github.com/HermonaDev/PelvicFetalPredictiveTool.git
Install dependencies: pip install -r requirements.txt
Run the app: streamlit run app.py

Future Enhancements

Integrate physics-based simulation for fetal descent (e.g., using PyBullet) to model biomechanical interactions.
Add support for real-time ultrasound data integration.
Expand dataset with generative AI (e.g., GANs) to simulate rare pelvic/fetal scenarios.

Author
Hermona
