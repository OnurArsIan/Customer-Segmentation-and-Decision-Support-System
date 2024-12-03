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

## Contact

For any questions or suggestions, please reach out at [your_email@example.com].

