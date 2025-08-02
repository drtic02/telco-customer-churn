# Telco Customer Churn Prediction

Machine learning project to predict customer churn for telecommunications companies.

## Project Objective

Complete data science workflow to predict customer churn and provide actionable business insights for proactive customer retention strategies.

## Key Results

- Best Model: Gradient Boosting (Advanced Model)
- F1-Score: 0.628 (4% improvement over baseline)
- ROC AUC: 0.840
- Accuracy: 77.2%
- Precision: 55.3%
- Recall: 72.7%
- Cross-Validation F1-Score: 0.840 (±0.014)
- Cross-Validation ROC-AUC: 0.915 (±0.006)

## Project Structure

```
TELCO CHURN PROJECT/
├── 01_data_exploration_and_cleaning.ipynb    # Data exploration and preprocessing
├── 02_feature_engineering_and_modeling.ipynb # Feature engineering and model training
├── 03_model_evaluation_and_analysis.ipynb    # Model evaluation and business insights
├── 04_advanced_model_improvement.ipynb       # Advanced modeling and optimization
├── requirements.txt                          # Project dependencies
├── OutlineProject.txt                        # Project mission brief
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       └── churn_data_cleaned.csv
├── models/
│   ├── best_churn_model.pkl                 # Baseline trained model with preprocessor
│   └── best_advanced_churn_model.pkl        # Advanced optimized model
└── reports/
    └── figures/
        ├── confusion_matrix.png             # Model performance visualization
        ├── feature_importance.png           # Feature importance analysis
        ├── roc_pr_curves.png               # ROC and Precision-Recall curves
        ├── advanced_model_comparison.png    # Advanced models performance comparison
        └── feature_importance_*.png         # Individual model feature importance
```

## Methodology

### Phase 1: Data Exploration & Cleaning
- Input: Raw telco customer data (7,043 customers, 21 features)
- Fixed data type issues
- Handled missing values
- Removed non-predictive features
- Exploratory data analysis

### Phase 2: Feature Engineering & Modeling
- Preprocessing: StandardScaler, OneHotEncoder
- Models: Logistic Regression, Random Forest, XGBoost
- Evaluation Metric: F1-Score
- Data Split: 80% training, 20% testing (stratified)

### Phase 3: Model Evaluation & Analysis
- Performance metrics analysis
- Feature importance analysis
- Business insights and recommendations

### Phase 4: Advanced Model Improvement
- Advanced feature engineering (12 new features)
- Hyperparameter optimization (RandomizedSearchCV)
- Class imbalance handling (SMOTE)
- Ensemble methods
- 5-fold cross-validation
- Advanced algorithms: Gradient Boosting, LightGBM
- 4% F1-Score improvement over baseline

## Key Predictive Features

1. TotalCharges (15.97%) - Customer's total spend
2. Tenure (13.96%) - Length of customer relationship
3. MonthlyCharges (13.71%) - Monthly subscription cost
4. Month-to-month Contract (5.00%) - Contract flexibility
5. No Online Security (3.28%) - Lack of security services

## Business Insights & Recommendations

### High-Risk Customer Profile
- Short tenure (< 12 months) + High monthly charges
- Month-to-month contracts
- Missing security services
- Fiber optic internet + Electronic payment methods

### Retention Strategies

#### Product & Service Improvements
- Promote longer-term contracts with incentives
- Bundle security services as default offerings
- Review pricing strategy for high monthly charge customers

#### Customer Segmentation
- High Risk: Short tenure + High charges + Month-to-month + No security
- Medium Risk: Medium tenure + Fiber optic + Electronic payment
- Low Risk: Long tenure + Lower charges + Long-term contracts

#### Proactive Interventions
- Early Warning System: Deploy model to score customers monthly
- Targeted Offers: Personalized retention campaigns for high-risk segments
- Customer Success: Proactive outreach to customers with tenure < 12 months

### Expected Business Impact
- Reduce churn rate by 15-20% through targeted interventions
- Improve customer lifetime value by extending average tenure
- Optimize marketing spend by focusing on high-risk customers
- Increase revenue through better retention and contract upgrades

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository (or download the project files)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
imbalanced-learn>=0.8.0
joblib>=1.1.0
```

### Usage

1. Data Exploration: Run `01_data_exploration_and_cleaning.ipynb`
2. Model Training: Run `02_feature_engineering_and_modeling.ipynb`
3. Model Evaluation: Run `03_model_evaluation_and_analysis.ipynb`
4. Advanced Optimization: Run `04_advanced_model_improvement.ipynb`

## Model Details

### Performance Metrics (Advanced Model)
- Sensitivity (Recall): 72.7%
- Specificity: 81.4%
- Precision: 55.3%
- F1-Score: 0.628
- ROC AUC: 0.840
- Cross-Validation F1: 0.840 (±0.014)
- Cross-Validation ROC-AUC: 0.915 (±0.006)

### Advanced Model Features
- 12 engineered features (customer value, lifecycle, service usage)
- SMOTE sampling for balanced training data
- Hyperparameter optimization using RandomizedSearchCV
- Ensemble approach with voting classifier

### Model Deployment Recommendations
- Use advanced Gradient Boosting model for production
- Set probability threshold at ~0.5 for balanced precision/recall
- Implement ensemble predictions for critical decisions
- Review model predictions monthly
- A/B test retention campaigns on high-risk customers
- Monitor model performance and feature drift with cross-validation metrics
- Retrain quarterly with new data using the established pipeline

## Visualizations

The project generates key visualizations:

1. Confusion Matrix - Model prediction accuracy breakdown
2. Feature Importance - Factors that influence churn
3. ROC & PR Curves - Model performance across thresholds
4. Advanced Model Comparison - Performance comparison across algorithms
5. Individual Model Feature Importance - Detailed feature analysis per algorithm

## Advanced Model Achievements

### Performance Improvements
- 4% F1-Score improvement: From 0.604 to 0.628
- 17% Recall improvement: From 55.9% to 72.7%
- Robust cross-validation: F1-Score 0.840 (±0.014)
- Gradient Boosting outperformed 5 other advanced models

### Technical Innovations
- Advanced feature engineering: 12 new features
- Class imbalance solution: SMOTE oversampling
- Hyperparameter optimization: RandomizedSearchCV
- Ensemble methods: Voting classifiers
- Cross-validation for consistent performance

### Business Impact
- Better churn detection: 73% vs 56% baseline
- Actionable insights from advanced features
- Production-ready model package
- Scalable pipeline for retraining and deployment

## Future Improvements

1. Deep learning models for pattern detection
2. Time series analysis for temporal patterns
3. External data integration (economic indicators, competitor analysis)
4. Real-time scoring pipeline
5. Automated feature selection with genetic algorithms
6. Model interpretability with SHAP values

## Contributing

This project demonstrates a complete data science workflow. Feel free to:
- Experiment with different models
- Add new features
- Improve preprocessing steps
- Enhance business insights

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Project completed using industry-standard data science practices with focus on business value and actionable insights. Advanced modeling techniques achieved 4% performance improvement with robust validation and production-ready deployment.
