# Customer Churn Prediction ML Project

## Overview
A production-ready machine learning system for predicting customer churn using classical ML techniques. This project provides a complete pipeline for data preprocessing, model training, evaluation, and inference.

## Project Structure
```
customer-churn-ml/
├── data/
│   ├── raw/              # Original, immutable data
│   └── processed/        # Cleaned and transformed data
├── models/               # Trained model artifacts (.joblib files)
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── reports/              # Generated analysis, figures, and reports
├── src/                  # Source code for the project
│   ├── config.py         # Configuration and constants
│   ├── preprocess.py     # Data preprocessing functions
│   ├── evaluate.py       # Model evaluation metrics
│   ├── predict.py        # Prediction logic
│   └── utils.py          # Utility functions
├── tests/                # Unit tests
├── requirements.txt      # Project dependencies
├── .gitignore           # Git ignore rules
└── run_inference.py     # Main inference script
```

## Features
- **Modular Architecture**: Clean separation of concerns with dedicated modules for preprocessing, evaluation, and prediction
- **Production-Ready**: Reproducible pipeline with serialized models and configurations
- **Extensible**: Easy to adapt for different classification tasks
- **Well-Documented**: Comprehensive docstrings and type hints

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd customer-churn-ml
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Inference
To predict customer churn on new data:

```bash
python run_inference.py --input data/raw/new_customers.csv --output predictions.csv
```

### Using the Prediction Module
```python
from src.predict import ChurnPredictor

# Initialize predictor with trained model
predictor = ChurnPredictor(model_path='models/churn_prediction_pipeline.joblib')

# Make predictions
predictions = predictor.predict(new_data)
probabilities = predictor.predict_proba(new_data)
```

### Data Preprocessing
```python
from src.preprocess import load_data, clean_data, prepare_features

# Load and preprocess data
df = load_data('data/raw/customers.csv')
df_clean = clean_data(df)
X, y = prepare_features(df_clean)
```

## Model Information
- **Model Type**: Classification (Binary)
- **Target Variable**: Churn (0 = No Churn, 1 = Churn)
- **Model Artifact**: `models/churn_prediction_pipeline.joblib`
- **Pipeline Includes**: Preprocessing steps + trained classifier

## Evaluation Metrics
The project tracks the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
This project follows PEP 8 style guidelines. Format code using:
```bash
black src/
flake8 src/
```

## Project Dependencies
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and utilities
- **xgboost**: Gradient boosting framework
- **joblib**: Model serialization
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License
This project is licensed under the MIT License.

## Contact
For questions or suggestions, please open an issue or contact the project maintainer.

## Acknowledgments
- Data source: [Specify your data source]
- Inspired by production ML best practices
