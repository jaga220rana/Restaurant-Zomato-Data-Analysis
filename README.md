# 🍽️ Bangalore Restaurants Analysis — Zomato Dataset

A comprehensive data science project performing **Exploratory Data Analysis**, **Classification**, **Regression**, and **Clustering** on Zomato's Bangalore restaurant data to uncover insights about dining trends, predict restaurant ratings, and segment restaurants into meaningful groups.

---

## 📌 About

Bangalore is one of India's most vibrant food cities, home to thousands of restaurants spanning every cuisine, price range, and dining style. This project leverages Zomato's publicly available restaurant dataset to answer key business questions:

- What factors influence a restaurant's rating?
- Can we predict whether a restaurant is expensive based on its attributes?
- Are there natural groupings (clusters) among Bangalore's restaurants?
- Do online ordering and table booking affect restaurant performance?

The analysis pipeline covers the full data science workflow — from raw data cleaning through to machine learning model evaluation.

---

## 📂 Dataset

- **Source:** [Kaggle — Zomato Bangalore Restaurants](https://www.kaggle.com/datasets)
- **Format:** CSV (`zomato.csv`)

### Features Used

| Feature | Description |
|---|---|
| `online_order` | Whether the restaurant accepts online orders (Yes / No) |
| `book_table` | Whether table booking is available (Yes / No) |
| `rate` | Customer rating out of 5 |
| `votes` | Total number of votes/reviews received |
| `approx_cost(for two people)` | Approximate dining cost for two people (₹) |
| `location` | Neighbourhood / area in Bangalore |
| `rest_type` | Type of restaurant (e.g., Quick Bites, Casual Dining) |
| `cuisines` | Primary cuisine(s) served |
| `listed_in(type)` | Listing category (e.g., Delivery, Dine-out, Buffet) |

> Columns dropped during preprocessing: `name`, `address`, `phone`, `dish_liked`

---

## ⚙️ Methodology

### 1. 🧹 Data Cleaning
- Converted `online_order` and `book_table` from Yes/No strings to binary (1/0)
- Stripped `/5` from the `rate` column and cast to numeric
- Removed commas from `approx_cost` and cast to numeric
- Consolidated rare categories in `location`, `rest_type`, `cuisines`, and `listed_in(type)` — retaining only the top 10 most frequent values and grouping the rest as `"Other"`

### 2. 🔧 Missing Value Imputation

| Feature | Strategy |
|---|---|
| `online_order`, `book_table` | Mode |
| `rate`, `votes` | Median |
| `approx_cost(for two people)` | Median |
| `location`, `rest_type`, `cuisines`, `listed_in(type)` | `"Unknown"` |

### 3. 📊 Exploratory Data Analysis (EDA)
- Histograms and KDE plots for all numerical features
- Count plots for all categorical features
- Pairplot to examine inter-feature correlations
- Box plots comparing `rate`, `votes`, and `cost` across restaurants that do/do not accept online orders

### 4. 🔨 Feature Engineering
- Created a binary `expensive` column: `1` if cost for two > median cost, else `0`
- Dropped the original `approx_cost(for two people)` column post-encoding

### 5. 🎯 Classification — Predicting Expensive Restaurants
Predicting whether a restaurant is expensive (above median cost) using:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

Evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

### 6. 📈 Regression — Predicting Restaurant Rating
Predicting the `rate` of a restaurant using:
- Linear Regression
- Ridge Regression (RidgeCV)
- Lasso Regression (LassoCV)
- ElasticNet Regression (ElasticNetCV)
- Random Forest Regressor
- Gradient Boosting Regressor

Model selection based on **RMSE** via **5-fold cross-validation**. Features encoded with `OneHotEncoder` and scaled with `MaxAbsScaler`.

### 7. 🔵 Clustering — Segmenting Restaurants
- Applied **K-Means clustering** to segment restaurants by characteristics
- Determined optimal cluster count using the **Elbow Method**
- Visualised clusters in 2D using **t-SNE** dimensionality reduction

---

## 📊 Key Findings

| Insight | Finding |
|---|---|
| Most common restaurant type | Quick Bites & Casual Dining |
| Most common cuisine | North Indian |
| Most common listing type | Delivery |
| Average rating | ~3.7 / 5 |
| Online ordering impact | Restaurants accepting online orders show slightly higher ratings and more votes |
| Rating vs. cost | No strong correlation — cost does not reliably reflect quality |
| Votes vs. cost | Positive correlation — pricier restaurants tend to get more votes |

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python 3 | Core programming language |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical computations |
| Matplotlib & Seaborn | Data visualisation |
| scikit-learn | Preprocessing, ML models, and evaluation |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook
```bash
git clone https://github.com/your-username/bangalore-restaurants-zomato.git
cd bangalore-restaurants-zomato
jupyter notebook bangalore-restaurants-a-zomato.ipynb
```

> Place the dataset at `/kaggle/input/zomato/zomato.csv` or update the path in the notebook to your local path.

---

## 📁 Repository Structure

```
bangalore-restaurants-zomato/
│
├── input/
│   └── zomato.csv                          # Raw Zomato dataset
│
├── bangalore-restaurants-a-zomato.ipynb    # Main analysis notebook
├── README.md                               # Project documentation
└── requirements.txt                        # Python dependencies
```

---

## 🗺️ Project Workflow

```
Raw Data
   │
   ▼
Data Cleaning & Imputation
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Feature Engineering
   │
   ├──► Classification  (Expensive or not?)
   │
   ├──► Regression      (Predict Rating)
   │
   └──► Clustering      (Segment Restaurants)
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋 Author

Made with ❤️ for learning and experimentation in data science. Contributions, issues, and forks are welcome!
