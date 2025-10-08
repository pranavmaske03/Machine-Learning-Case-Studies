# Iris Classification Case Study

## 1. Problem Statement
The **Iris Flower Classification** case study aims to build a machine learning model that can classify iris flowers into one of three species — **Setosa, Versicolor, or Virginica** — based on four key features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## 2. Dataset

We used the Iris dataset (stored in iris.csv) for this case study. It contains 150 records of iris flowers with the following structure:

- **Features:**
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width
- **Labels:**
    - Species of Iris flower (Setosa, Versicolor, Virginica)

This dataset is widely used in machine learning as a beginner-friendly classification problem.

**Feature Encoding:**  
- In this dataset, all features (sepal length, sepal width, petal length, petal width) are numeric.
- Therefore, no special feature encoding is required.

**Label Encoding:**  
- The Label column (species) is categorical:
    - Setosa
    - Versicolor
    - Virginica
- We apply Label Encoding to convert these species into numeric form:
    - Setosa → 0
    - Versicolor → 1
    - Virginica → 2

## 4. Approach
The following steps were followed to build the **Iris Flower Classification model:**

**1.Data Loading**
- Loaded the dataset (iris.csv) using Pandas.

**2. Data Preprocessing**
- Checked for missing values.
- Applied Label Encoding to convert species into numeric values.

**3. Feature Selection**
- Selected four features: sepal length, sepal width, petal length, petal width.

**4. Train-Test Split**
- Split the dataset into training (80%) and testing (20%) sets.

**5. Model Training**
- Used Decision Tree Classifier from scikit-learn to train the model.

**6. Evaluation**
- Evaluated the model using accuracy score on the test data.

## 5. How to Run

- Make sure **Python** and the **scikit-learn** library are installed.  
  You can install scikit-learn using:
  ```bash
  pip3 install scikit-learn
- Run the program using:
    ```bash
    python3 iris-classification.py

## 6. Conclusion
This case study demonstrates how a Decision Tree Classifier can be trained on the Iris dataset to classify flowers into three species i.e Setosa, Versicolor, and Virginica.

It highlights the basic machine learning workflow:

- Preparing the dataset
- Training the model
- Making predictions

Through this simple example, we see how ML models can learn patterns from data and apply them to make accurate predictions.