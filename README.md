# Recipe Site Traffic Prediction

## 🎯 Project Overview

This project is part of the **DataCamp Data Scientist Professional Certification**. The goal is to build a machine learning model to predict which recipes will lead to high website traffic, helping Product Managers decide which recipes to feature on the homepage.

**Business Goal**: Achieve 80%+ correct predictions for high-traffic recipes.

## 📊 Dataset

The dataset contains 947 recipes with the following features:

| Feature | Description |
|---------|-------------|
| `recipe` | Unique recipe identifier |
| `calories` | Number of calories |
| `carbohydrate` | Amount of carbohydrates (g) |
| `sugar` | Amount of sugar (g) |
| `protein` | Amount of protein (g) |
| `category` | Recipe category (11 categories) |
| `servings` | Number of servings |
| `high_traffic` | Target variable (High/Low) |

## 🔍 Project Structure

```
├── workspace/
│   ├── notebook.ipynb              # Main Jupyter notebook with analysis
│   └── recipe_site_traffic_2212.csv # Dataset
├── app.py                          # Streamlit web application
├── explore_data.py                 # Data exploration script
├── test_notebook.py                # Test file
├── data_exploration.txt            # Exploration notes
└── README.md                       # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning
- **Streamlit** - Web application

## 📈 Key Findings

1. **Class Balance**: 60.6% of recipes are high-traffic
2. **Category Impact**: Certain categories (Vegetable, Potato, Pork) significantly influence traffic
3. **Nutritional Features**: Calories, protein, and carbohydrates show moderate correlation with traffic

## 🚀 Model Performance

The Logistic Regression model was selected as the final model:

- **Accuracy**: ~77%
- **Precision (High Traffic)**: ~83%
- **Recall (High Traffic)**: ~81%

## 💻 Running the Streamlit App

1. Install dependencies:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

2. Run the app:
```bash
streamlit run app.py
```

## 📝 Key Recommendations

1. **Feature Prioritization**: Focus on recipe category when selecting homepage content
2. **Data Collection**: Gather more data on user engagement patterns
3. **Model Updates**: Regularly retrain the model with new data
4. **A/B Testing**: Validate predictions with controlled experiments

## 👤 Author

**Nagendra Singh Rawat**
- GitHub: [@iNSRawat](https://github.com/iNSRawat)

## 📄 License

This project is part of the DataCamp Professional Certification.
