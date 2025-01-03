# **Financial Sentiment Analysis**

## **Overview**
This project focuses on performing sentiment analysis on financial text data. It leverages Natural Language Processing (NLP) techniques to preprocess the text, visualize insights, and classify sentences into sentiment categories. The models used include Logistic Regression and Naive Bayes variants, with hyperparameter tuning to enhance performance.

---

## **Table of Contents**
1. [Dataset](#dataset)  
2. [Technologies Used](#technologies-used)  
3. [Project Workflow](#project-workflow)  
4. [Model Performance](#model-performance)  
5. [Installation and Usage](#installation-and-usage)  
6. [Future Improvements](#future-improvements)  
7. [Contributors](#contributors)  
8. [License](#license)

---

## **Dataset**
- **Source**: The dataset used for this project is `data.csv`.  
- **Columns**:  
  - `Sentence`: Text data containing financial sentences.  
  - `Sentiment`: Target variable representing sentiment labels (e.g., Positive, Negative, Neutral).  
- **Dataset Details**:  
  - Shape: _(number of rows × number of columns)_  
  - Missing Values: Checked and handled appropriately.

---

## **Technologies Used**
- **Languages & Libraries**:  
  - Python  
  - NumPy, Pandas, Matplotlib, Seaborn  
  - NLTK, WordCloud  
  - Scikit-learn  
- **Tools**:  
  - Google Colab  
  - Jupyter Notebook  

---

## **Project Workflow**

### 1. **Data Cleaning**
- Removed HTML tags using regex.
- Tokenized, stemmed, and removed stopwords using **NLTK**.

### 2. **Exploratory Data Analysis (EDA)**
- Visualized sentiment distribution using bar plots.  
- Generated a WordCloud for high-frequency words.

### 3. **Feature Extraction**
- Converted text into numerical format using:
  - **CountVectorizer**  
  - **TfidfVectorizer**

### 4. **Model Training**
- Models used:  
  - **Logistic Regression**  
  - **Multinomial Naive Bayes**  
  - **Gaussian Naive Bayes**  
  - **Bernoulli Naive Bayes**
- Performed train-test split with an 80-20 ratio.

### 5. **Evaluation**
- Measured performance using metrics:
  - **Accuracy**  
  - **Classification Report**  
  - **Confusion Matrix**

### 6. **Hyperparameter Tuning**
- Conducted hyperparameter optimization using **GridSearchCV** for all models to enhance accuracy and generalization.

---

## **Model Performance**
| Model                  | Accuracy (Before Tuning) | Accuracy (After Tuning) |
|------------------------|--------------------------|--------------------------|
| Logistic Regression    | 69.2%                   | 69.2%                   |
| Multinomial Naive Bayes| 71.0%                   | 71.0%                   |
| Gaussian Naive Bayes   | 46.6%                   | 46.6%                   |
| Bernoulli Naive Bayes  | 70.6%                   | 70.6%                   |

Confusion matrices and classification reports were generated for each model.

---

