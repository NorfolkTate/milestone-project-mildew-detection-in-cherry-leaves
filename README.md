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
indentation error
forgetting to cntrl s so commit messages were haphazard
Didn't understand whether to import libraries for each notebook so sometimes they're imported more than once
Various ways to split data and v confusing 
TensorFlow not downloading - windows error due to vs code Enable Long Paths on Windows
Reloading vscode and being back in my old repo
is there a difference between !pip and pip
spelling in normalize/normalise 
didn't import layers
saved at h5 changed to keras
removing large files added to gitignore

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
* Notebook 1
    - Kaggle
* Notebook 2
    - 1 - https://realpython.com/image-processing-with-the-python-pillow-library/
    - 2 - https://matplotlib.org/stable/tutorials/images.html
    - 3 - https://stackoverflow.com/questions/52214776/python-matplotlib-differences-between-subplot-and-subplots
    - 4 - https://stackoverflow.com/questions/72772666/get-the-average-color-of-an-image-before-it-loads-on-the-page
    - 5 - https://www.geeksforgeeks.org/matplotlib-pyplot-figure-in-python/
    - 5 - https://www.scaler.com/topics/matplotlib/introduction-to-figures-in-matplotlib/
    - 6 - https://www.mathworks.com/help//releases/R2021a/matlab/ref/imshow.html#:~:text=imshow(%20I%20)%20displays%20the%20grayscale,vector%2C%20%5Blow%20high%5D%20.
* Notebook 3 
    - 1 - https://stackoverflow.com/questions/60675066/why-is-the-raw-file-path-not-working-in-python
    - 2 - https://docs.python.org/3/library/os.path.html
    - 3 - https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/
* Notebook 4
    - 1 - https://stackoverflow.com/questions/41175401/what-is-a-batch-in-tensorflow
    - 2 - https://keras.io/api/data_loading/image/
    - 3 - https://www.reddit.com/r/learnmachinelearning/comments/11m0tyy/what_exactly_is_happening_when_i_dont_normalize/?rdt=59010
    - 3 - https://stackoverflow.com/questions/33610825/normalization-in-image-processing
    - 4 - https://forums.fast.ai/t/normalizing-images-with-a-lambda-instead-of-stats-latest-efficientnet-requiires-it/62441
    - 5 - https://www.reddit.com/r/deeplearning/comments/iov9qy/proper_use_of_tensorflow_dataset_prefetch_and/
    - 6 - https://colab.research.google.com/github/csoren66/Potato_disease_Detection/blob/main/Potato_disease_detection.ipynb#scrollTo=dTStMzu6wAhH
https://viso.ai/computer-vision/image-classification/#:~:text=Today%2C%20the%20use%20of%20convolutional,What%20Is%20Image%20Classification%3F
    - 7 - https://www.tensorflow.org/guide/keras/serialization_and_saving
    - 8 - https://docs.python.org/3/library/pickle.html
* Notebook 5 
    - 1 - https://stackoverflow.com/questions/59840289/model-evaluate-in-keras
    - 2 - https://discuss.pytorch.org/t/how-to-plot-train-and-validation-accuracy-graph/105524

### Content

- Project brief and provided by Code Institute.
- CNN architecture inspired by Keras tutorials.
- Streamlit multipage functionality adapted from Code Institute Malaria walkthrough
- TensorFlow tutorial
- Keras documentation for saving the model
- The general use of Shutil module in python was helpfully explained via geeks for geeks
- Stack overflow provided great information on spliting data into 3 groups
- The general idea of scikit-learns train_test_split function was helpfully explained via geeks for geeks

### Media
- All images used are from the Kaggle cherry leaf dataset.




