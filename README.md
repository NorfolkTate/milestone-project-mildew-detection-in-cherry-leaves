# Mildew Detection in Cherry Leaves

## Contents
- [Introduction](#introduction)
- [Business Requirements](#business-requirements)
- [Machine Learning Business Case](#machine-learning-business-case)
- [Hypothesis](#hypothesis)
- [Mapping Business Requirements to Project Tasks](#mapping-business-requirements-to-project-tasks)
- [Dashboard Design](#dashboard-design)
- [Dataset Content](#dataset-content)
- [App Features](#app-features)
- [Testing](#testing)
- [Known Bugs and Limitations](#known-bugs-and-limitations)
- [Further Improvements](#further-improvements)
- [Deployment](#deployment)
- [Main Libraries Used](#main-libraries-used)
- [Credits](#credits)

## Introduction
This project aims to deliver a dashboard that visually differentiates healthy cherry leaves from those affected by powdery mildew and provides predictions on new leaf images. Using computer vision and machine learning, I have built a convolutional neural network (CNN) to automate the classification process and present the results through an interactive Streamlit dashboard.

## Business Requirements
The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Machine Learning Business Case
Given the image-based nature of the problem, a binary classification model was deemed suitable. A Convolutional Neural Network (CNN) was selected for its proven effectiveness in image classification tasks.

### Performance Goal:
- A target accuracy of 97% was initially discussed. However, due to dataset limitations and model size considerations, this became an aspirational target rather than a strict requirement.

### Business Impact:
- Accurate classification ensures the client avoids supplying compromised products, protecting both reputation and market value.

## Hypothesis

* Hypothesis 1:
The visual characteristics of healthy cherry leaves differ significantly from those affected by powdery mildew and can be quantitatively analyzed.
* Validation: I will use image analysis techniques, such as average images, variability images, and difference plots, to highlight distinguishable features. Image montages will be used for side-by-side comparisons.

Hypothesis 2:
* A convolutional neural network trained on labeled leaf images can classify new leaf samples with up to 97% accuracy.
* Validation: I will evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix. I will compare test results to the target threshold of 97%.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

Conduct a visual study to differentiate between healthy and mildew-infected cherry leaves.

Build an automated system that can predict whether a cherry leaf is healthy or contains powdery mildew.

These requirements stem from a real business challenge faced by Farmy & Foods. Their current inspection process is manual and time consuming, taking up to 30 minutes per tree. By developing an automated image classification system, the company hopes to save time, reduce costs, and maintain product quality across its extensive network of cherry farms.

## Dashboard Design
The dashboard was designed with clarity and business relevance in mind. Pages were structured to address each client requirement and showcase the project workflow.

Key design considerations:
- Simple navigation using Streamlit multipage functionality.
- Clear separation of project overview, findings, predictions, and technical details.
- Allowing non-technical users to interact with the model through file uploads and reports.

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## App Features

### Project Summary Page
- Overview of the project goals, dataset summary, and business requirements.
- Highlights the binary classification problem and the approach taken.

### Visual Study Page
- Displays visual differentiators between healthy and powdery mildew leaves.
- Includes:
  - Average images per class.
  - Image montage grids.
  - Difference visualisations.

### Prediction Page
- File uploader for users to upload cherry leaf images.
- For each image:
  - Displays the image.
  - Shows a prediction statement (Healthy / Powdery Mildew) with probability.
- A table summarising all predictions with a download button for CSV export.

### Model Performance Page
- Displays training and validation accuracy and loss curves.
- Shows final evaluation metrics on test data.
- Discusses observed overfitting and potential reasons.

## Testing
- All notebooks were re-run sequentially to ensure reproducibility.
- Streamlit app tested locally via streamlit run app.py.
- Manual testing of image uploads and predictions.

## Known Bugs and Limitations

## Further Improvements

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- The project was deployed to Heroku using the following steps

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- **TensorFlow & Keras:** Model development and training.
- **Matplotlib & Seaborn:** Data visualisation.
- **Pandas & NumPy:** Data manipulation.
- **Streamlit:** Dashboard development.
- **Scikit-learn:** Evaluation metrics.
- **OS & Glob:** File handling.

## Credits

### Credits in Code

### Content

- Project brief and provided by Code Institute.
- CNN architecture inspired by Keras tutorials.
- Streamlit multipage functionality adapted from Code Institute Malaria walkthrough

### Media
- All images used are from the Kaggle cherry leaf dataset.




