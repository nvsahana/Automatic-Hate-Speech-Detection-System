# Project: Automatic Hate Speech Detection using TF-IDF Vectorization and Ensemble Machine Learning Approach
A model that incorporates TF-IDF vectorization for feature extraction and employs an ensemble machine learning approach (an ensemble of Support Vector and Random Forest classifier) to accurately classify hateful and offensive content. The end application is a user-friendly Web Application that seamlessly handles textual inputs in both English and Hindi-English code mix language. Through advanced classification techniques, the underlying model accurately classifies the input as hate speech, offensive content, or neutral, based on its content.
# System Design and Steps Involved
## 1. Dataset Collection
The dataset utilized for this project comprises publicly available data, encompassing a diverse range of tweets in both English and Hinglish (a hybrid of Hindi and English). 
## 2. Data Preprocessing
Following the dataset collection, a crucial step lies in data preprocessing. This involves the removal of undesirable elements like usernames, URL links, multiple spaces, emojis, numbers, and other extraneous characters,  refining the data for subsequent analysis. The data pre-processing techniques employed in this project include case-folding, data cleaning, tokenization and stop-word removal.
## 3. Feature Extraction
This step focuses on feature extraction, transforming textual data into a matrix or vector of features. To accomplish this, the widely adopted technique of Term Frequency-Inverse Document Frequency (TF-IDF) vectorization is employed. Through this process, the inherent characteristics of the data are effectively captured, facilitating further analysis.
## 4. Dataset Splitting
To ensure the proper training and validation of the system, the pre-processed data is split into two distinct sets: the training dataset and the testing dataset. This division involves allocating 80 percent of the data for training purposes while reserving the remaining 20 percent for testing and evaluating the system's performance.
## 5. Training Model
To maximize classification accuracy, an ensemble model is constructed by integrating multiple single classification techniques like KNN, Random-Forest Classifiers, RVM, SVM, and more. Through rigorous experimentation with various combinations of these classifiers using both training and testing data, the model with the highest accuracy is identified and selected as the final solution.
### Algorithm: Evaluating single classifiers and Ensemble model
#### Step 1: Load Dataset.
#### Step 2: Load Random Forest, Decision Tree, Logistic regression model, etc., or ensemble model.
#### Step 3: Train the developed model.
#### Step 4:Evaluate the developed model and display the classification report.
## 6. Evaluate Final Model
Different combinations are tested with the testing data and the combination with the highest accuracy is deemed as the final model. 
