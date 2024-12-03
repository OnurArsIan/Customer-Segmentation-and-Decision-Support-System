# Customer Segmentation and Decision Support System

## Overview

This project is a Customer Segmentation and Decision Support System developed using Python and Streamlit. It aims to classify customers into different segments based on their shopping habits and provide customized recommendations for financial products, consumer products, and services. The model is built using Random Forest Classifier, with preprocessing steps involving feature scaling and encoding of categorical variables.

## Features

- **Data Loading and Preprocessing**: The data is loaded from CSV files and preprocessed to handle missing values, add new features like Income-to-Spending Ratio, and apply One-Hot Encoding for categorical features.
- **Machine Learning Model**: A Random Forest Classifier is trained to classify customers into 8 different clusters.
- **Recommendations**: Based on the predicted cluster, personalized recommendations for credit offers, products, and services are generated.
- **Interactive User Interface**: Users can input customer information via a user-friendly Streamlit sidebar to get predictions and recommendations.

## Technologies Used

- **Python**: The core language for data processing and modeling.
- **Streamlit**: For building an interactive web application.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For feature scaling, encoding, and model training.

## Dataset

The dataset used for this project contains customer information such as:
- **Gender**: The gender of the customer (Male/Female).
- **Age Group**: The age group of the customer.
- **Marital Status**: The marital status of the customer (Single/Married).
- **Education Level**: The highest education level achieved by the customer.
- **Employment Status**: The employment status (Employed/Unemployed/Self-Employed).
- **Average Annual Income**: The average annual income of the customer.
- **City of Residence**: Type of city where the customer resides (Urban/Rural).
- **Product Interests**: Interests in specific product categories (e.g., Electronics, Fashion, Household).
- **Purchase and Order Data**: The number of purchases and orders made by the customer.
- **Education Continuation Status**: Whether the customer is continuing education or not.

The dataset is used to categorize customers based on their shopping habits, and it forms the basis for providing personalized financial and product recommendations.

## Installation

1. Clone the repository:
   ```sh
   git clone <repository_url>
   ```

2. Navigate to the project directory:
   ```sh
   cd customer-segmentation-system
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure you have the training and testing CSV files (`train.csv` and `test_x.csv`) in the project directory.
2. Run the Streamlit application:
   ```sh
   streamlit run updated_uygulama.py
   ```
3. Open the local URL provided by Streamlit to interact with the application.

## Usage

- Use the sidebar to enter customer information, such as annual income, average spending, gender, employment status, and more.
- The model will predict the customer segment and display personalized recommendations for credit, products, and services.

## Data Preprocessing

The data preprocessing steps include:
- **Missing Value Handling**: Missing values are filled using forward-fill (`ffill`) method.
- **Feature Engineering**: A new feature, "Income-to-Spending Ratio," is calculated.
- **One-Hot Encoding**: Categorical features such as `Gender`, `Education Status`, and `Employment Status` are converted into numerical form using One-Hot Encoding.
- **Feature Scaling**: Important features are scaled using `StandardScaler` to normalize the data.

## Model Details

The Random Forest Classifier used in this project is configured with the following hyperparameters:
- **Number of Estimators**: 300
- **Minimum Samples Split**: 5
- **Minimum Samples Leaf**: 2
- **Max Features**: Logarithmically selected features
- **Class Weight**: Balanced to handle imbalanced datasets

The model is trained to classify customers into 8 different clusters, each with distinct characteristics, which enables personalized recommendations.

## Recommendations System

The recommendation system uses the predicted customer segment to suggest:
- **Credit Offers**: Personalized based on income and spending patterns.
- **Products**: Tailored suggestions depending on the customer's interests and spending behavior.
- **Services**: Customized services such as financial planning, investment advice, and lifestyle services.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributions

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for discussion.


![Ekran görüntüsü 2024-12-03 053401](https://github.com/user-attachments/assets/7f793322-4ec8-494c-af08-95d2aae79150)

