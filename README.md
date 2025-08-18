# Mildew Detection in Cherry Leaves

## Contents
- [Introduction](#introduction)
- [Business Requirements](#business-requirements)
- [User Stories](#user-stories)
- [Machine Learning Business Case](#machine-learning-business-case)
- [Hypothesis](#hypothesis)
- [Mapping Business Requirements to Project Tasks](#rationale-to-map-business-requirements-to-ml-tasks)
- [Dashboard Design](#dashboard-design)
- [Dataset Content](#dataset-content)
- [App Features](#app-features)
- [Testing](#testing)
- [Known Bugs and Limitations](#known-bugs-and-limitations)
- [Further Improvements](#further-improvements)
- [Conclusion](#conclusion)
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

## User Stories

The project was developed using Agile methodology. User stories were written to understand and atriculate business requirements and implemented through GitHub Issues and project board (linked to the repo).

### User Story 1: Simple Prediction
**As a farmer**, I want to upload a cherry leaf photo and see whether it is healthy or infected, so that I can make quick treatment decisions in the field  

**Acceptance Criteria:**  
- I can upload `.jpg/.jpeg/.png` images.  
- The image is resized and preprocessed automatically.  
- The model at `outputs/models/cherry_leaf_model.keras` is loaded.  
- A prediction (healthy or powdery_mildew) is displayed with class probabilities  

---

### User Story 2: Dataset View
**As a user**, I want to view how the dataset is structured and see examples of each class, so that I can understand the data quality and balance.  

**Acceptance Criteria:**  
- I can choose a dataset split (train/val/test) from `inputs/dataset/`.  
- A bar chart shows the number of images in each class.  
- I can view 4–20 random sample images from either class with captions.  
- A short explanation is provided about differences between healthy and mildew leaves.  

---

### User Story 3: Model Performance Check
**As a manager**, I want to review a simple summary of model performance, so that I can trust the system before deploying it.  

**Acceptance Criteria:**  
- If `outputs/metrics.json` exists, show accuracy, precision, recall, and F1.  
- If `outputs/y_true.npy` and `outputs/y_pred.npy` exist, show a confusion matrix.  
- A short explanation of the confusion matrix is provided.  
- If results are missing, a clear message is shown instead of an error.  

---

### User Story 4: Simple Project Overview
**As a stakeholder**, I want to see a clear project summary and system readiness check, so that I understand what the tool does and whether it is ready to use.  

**Acceptance Criteria:**  
- The Home page explains the app functions in plain English  
- Instructions are provided for how to use the dashboard


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

### Rationale to Map Business Requirements to ML Tasks  

The key business requirement is to provide cherry growers with a reliable way of identifying powdery mildew on leaves early, in order to protect yield and reduce unnecessary chemical use. To achieve this, the following mapping between business requirements, ML tasks, and visualisations has been established:  

- **Business Requirement:** Detect whether a cherry leaf is healthy or infected.  
  - **ML Task:** Binary image classification using a Convolutional Neural Network (CNN).  
  - **Dashboard Feature:** *Predict Leaf Infection* page, where users upload an image and receive a prediction with confidence score.  

- **Business Requirement:** Provide transparency into how the model performs.  
  - **ML Task:** Model evaluation through accuracy, precision, recall, and F1-score.  
  - **Dashboard Feature:** *Model Performance* page, which presents training/validation curves, metrics from `metrics.json`, and a confusion matrix with explanations.  

- **Business Requirement:** Build user trust by allowing exploration of the dataset.  
  - **ML Task:** Not directly ML, but supports ML explainability by showing dataset balance and sample images.  
  - **Dashboard Feature:** *Visualise Dataset* page, where users can explore class balance and random samples to understand what the model was trained on.  

This alignment ensures that machine learning is not only applied (via CNN-based classification) but also directly supports the stated business objectives through interactive dashboard components.  


## Dashboard Design
The dashboard was designed with clarity and business relevance in mind. Pages were structured to address each client requirement and showcase the project workflow. These are demonstrated by the user stories designed for this project and documented on a kanban board.

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

During development, a number of challenges and limitations were encountered:  

- **Container width**: Fixed by adding `use_container_width=True`.  
- **Undefined `probs`**: Solved by setting it to `None` (seems to be a Streamlit quirk).  
- **Image size mismatch**: Standardised uploads to (256, 256, 3).  
- **Indentation errors**: Python being picky — fixed as I tested.  
- **Forgot to save before commits**: A few messy commit messages because of this - see,s to be a VS code fave 
- **Library imports**: Sometimes repeated across notebooks. Didn’t break anything, just untidy.  
- **Data splitting**: Tried a few methods before sticking with a consistent approach.  
- **TensorFlow install**: Wouldn’t install on Windows until I enabled long paths in VS Code.  
- **Repo confusion**: VS Code sometimes reopened an old repo.  
- **Pip vs !pip**: Got caught out a couple of times switching between terminal and notebooks.  
- **Spelling mistakes**: e.g., *normalize* vs *normalise*.  
- **Model saving**: Started with `.h5`, moved to Keras’ native format for stability.  
- **Large files**: Accidentally committed some, later fixed with `.gitignore`

These challenges provided valuable (if sometimes frusrtating) learning opportunities and do not prevent the app from functioning 


## Further Improvements

There are several areas where this project could be extended or improved in the future:
- **Data Augmentation:** Apply techniques such as rotation, flipping, and zooming to increase dataset variability and reduce overfitting.
- **Larger Models:** Experiment with more advanced architectures such as ResNet or EfficientNet to potentially improve accuracy.
- **Additional Crops:** Extend the model to detect powdery mildew (or other diseases) in crops beyond cherries, making it more useful to Farmy & Foods.
- **Mobile Deployment:** Develop a lightweight mobile version of the model so farmers can take photos of leaves in the field and receive instant predictions.


## Conclusion
This project set out to build a system capable of distinguishing between healthy cherry leaves and those infected with powdery mildew. By training a Convolutional Neural Network (CNN) on the provided dataset, the model achieved a high level of accuracy, correctly classifying most test images.  

The results show that the model is effective at addressing the client’s main requirement: reducing the need for manual inspection and saving time across large cherry plantations. While the model is not perfect and may benefit from more training data or augmentation, it demonstrates that predictive analytics can be successfully applied to this agricultural challenge.  

In summary, the machine learning pipeline was successful in meeting the project goals, and it provides a strong foundation for further development and deployment

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

#### links from credits in code 
https://stackoverflow.com/questions/55378692/load-keras-model-and-cache-it-in-a-variable-without-having-to-reload - Keras model cache
https://stackoverflow.com/questions/36158469/what-is-meaning-of-using-probsrange6-y - understanding and solving probs issue
https://www.geeksforgeeks.org/deep-learning/tf-keras-models-load_model-in-tensorflow/ - keras model loading 
https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy - plotting the accuracy and loss graphs
https://www.geeksforgeeks.org/python/understanding-python-pickling-example/ - pickle!
https://docs.python.org/3/library/pickle.html - pickle
https://medium.com/@oladokedamilola7/a-beginners-guide-to-database-management-in-python-f77deb297389 - database and data management basics
https://stackoverflow.com/questions/15320052/what-are-all-the-differences-between-src-and-data-src-attributes -SRC
https://docs.streamlit.io/develop/api-reference/charts/st.pyplot - general understanding of plotting charts 
https://www.geeksforgeeks.org/python/plotting-multiple-bar-charts-using-matplotlib-in-python/ - understanding plotting graphs
https://rollbar.com/blog/what-is-except-exception-as-e-in-python/ - e as exception for try and accept
https://discuss.pytorch.org/t/confused-about-target-label-when-using-nn-crossentropyloss/63264 - explaining and understanding issue with plotting and loss
https://numpy.org/doc/2.2/reference/generated/numpy.argmax.html - learning and understanding argmax

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




