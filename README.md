# Clock Drawing Classification

This project demonstrates the classification of clock images using machine learning techniques in Python. The workflow includes data preprocessing, exploratory data visualization, and model training using scikit-learn.

## Dataset
The data is retrieved from the `brainCodeCamp2023` repository and consists of:
* **Images**: 60 grayscale image samples, each with a resolution of 48x48 pixels.
* **Labels**: Binary classifications (0 and 1) corresponding to the image samples.

## Prerequisites
To run the notebook successfully, you will need the following Python libraries:
* `numpy`
* `matplotlib`
* `scikit-learn`

## Project Workflow

1.  **Data Ingestion**: The notebook automatically downloads the `clock_images.pickle` and `labels.pickle` files via web requests and loads them into memory.
2.  **Exploratory Data Analysis**: Sample images from both classes are visualized to provide an understanding of the dataset's characteristics.
3.  **Data Preprocessing**: 
    * The data is split into training (36 samples) and testing (24 samples) sets, maintaining class stratification.
    * Features are standardized using `StandardScaler` to ensure optimal performance during model training.
    * A `StratifiedKFold` setup is defined for cross-validation.
4.  **Model Training and Evaluation**:
    * **Baseline Model**: A `LogisticRegression` classifier is trained. Its performance is evaluated and visualized using classification reports and confusion matrices for both the training and test sets.
    * **Support Vector Machine (SVM)**: A linear SVM (`SVC`) is implemented. The notebook includes a 2D visualization of the SVM's decision boundary, margins, and support vectors.

## Usage
Run the cells in the Jupyter Notebook sequentially. Ensure you have an active internet connection during the first run, as the notebook will execute `wget` commands to download the dataset into your working directory before training the models.
