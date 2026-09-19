
# Customer Segmentation Using PCA and KMeans

## 📌 Project Overview

This project focuses on segmenting customers of an online retail store into meaningful behavioral groups using Machine Learning techniques.

The project uses **Pandas, StandardScaler, PCA, and KMeans Clustering** to analyze customer purchasing behavior and identify different customer segments.

The goal is to transform raw transaction data into actionable customer insights that can help businesses improve marketing strategies and customer retention.

---

## 🎯 Objectives

- Load and clean the Online Retail dataset.
- Perform customer-level feature engineering.
- Calculate total spending, number of transactions, average order value, recency, and unique products purchased.
- Normalize features using StandardScaler.
- Apply Principal Component Analysis (PCA) for dimensionality reduction.
- Generate a customer value score using PCA.
- Segment customers using KMeans clustering.
- Visualize customer segments using PCA scatter plots.
- Use the Elbow Method to analyze the optimal number of clusters.
- Develop a reusable customer scoring pipeline.
- Provide business insights for each customer segment.

---

## 📂 Dataset

**Dataset:** Online Retail Dataset

**Source:** Kaggle

The dataset contains transactional information from an online retail store.

### Important Columns

| Column | Description |
|---|---|
| InvoiceNo | Invoice number |
| StockCode | Product code |
| Description | Product description |
| Quantity | Number of products purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per product |
| CustomerID | Unique customer identifier |
| Country | Customer's country |

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** – Data loading, cleaning, and manipulation
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine learning algorithms

### Machine Learning Techniques

- StandardScaler
- Principal Component Analysis (PCA)
- KMeans Clustering
- Elbow Method
- Silhouette Score

---

## 🔄 Project Workflow

```text
Raw Transaction Data
        |
        ▼
Data Cleaning
        |
        ▼
Feature Engineering
        |
        ▼
Log Transformation
        |
        ▼
Feature Scaling
(StandardScaler)
        |
        ▼
PCA (2 Components)
        |
        ▼
Customer Value Score
(PCA - 1 Component)
        |
        ▼
KMeans Clustering
(k = 3)
        |
        ▼
Customer Segmentation
        |
        ▼
Business Analysis
```

---

## 📊 Features Engineered

The following features are calculated for each customer:

| Feature | Description |
|---|---|
| TotalSpending | Total amount spent by a customer |
| NumTransactions | Number of unique invoices |
| AverageOrderValue | Average spending per order |
| Recency | Number of days since the last purchase |
| UniqueProducts | Number of distinct products purchased |

---

## 🧠 Methodology

### 1. Data Cleaning

- Remove customers with missing CustomerID.
- Remove duplicate records.
- Remove invalid quantities and prices.
- Exclude cancelled invoices.
- Calculate total transaction amount.

### 2. Feature Transformation

Log transformation is applied to reduce the effect of highly skewed transaction values.

StandardScaler is then used to normalize the features so that they have comparable scales.

### 3. Principal Component Analysis (PCA)

PCA is used to reduce the dimensionality of the dataset.

- PCA with 2 components is used for visualization.
- PCA with 1 component is used to generate a customer value score.

### 4. KMeans Clustering

KMeans clustering is applied with:

```python
n_clusters = 3
random_state = 42
```

Customers are grouped based on their purchasing behavior.

---

## 👥 Customer Segments

The project assigns descriptive names to clusters based on their centroid characteristics.

### High-Value Regulars

Customers with relatively high spending and purchasing activity.

**Possible business strategy:**
- Loyalty rewards
- Personalized offers
- Customer retention programs

### Occasional Buyers

Customers with moderate purchasing activity.

**Possible business strategy:**
- Targeted promotions
- Repeat purchase incentives
- Personalized product recommendations

### One-Time / Low-Value Shoppers

Customers with relatively low purchasing activity or limited engagement.

**Possible business strategy:**
- Re-engagement campaigns
- First-to-second purchase incentives
- Promotional offers

> Segment characteristics are determined from the actual cluster centroids and may vary depending on the dataset.

---

## 📈 Visualizations

The project generates the following visualizations:

1. 2D PCA Scatter Plot
2. PCA Scatter Plot Colored by Cluster
3. Elbow Plot (k = 2 to 10)
4. Silhouette Score Plot

These visualizations help analyze cluster separation and customer behavior.

---

## 📁 Project Structure

```text
Week 4/
│
├── Online_Retail.csv
├── retail_analysis.py
├── customer_scores_segments.csv
├── complete_customer_segmentation.csv
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Navigate to the project folder:

```bash
cd "Week 4"
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

---

## ▶️ How to Run

1. Download the Online Retail dataset.
2. Place `Online_Retail.csv` in the project directory.
3. Ensure the dataset path matches the Python script.
4. Run the Python file:

```bash
python retail_analysis.py
```

5. View the generated customer segmentation results.

---

## 📤 Output

The project generates:

```text
customer_scores_segments.csv
complete_customer_segmentation.csv
```

The scoring output contains:

| Column | Description |
|---|---|
| customer_id | Unique customer identifier |
| score | PCA-based customer value score |
| segment | Assigned customer segment |

---

## 💼 Business Applications

Customer segmentation can help businesses:

- Identify valuable customers.
- Design targeted marketing campaigns.
- Improve customer retention.
- Understand purchasing behavior.
- Personalize customer experiences.
- Develop data-driven marketing strategies.

---

## 🚀 Future Improvements

- Experiment with different clustering algorithms.
- Optimize the number of clusters.
- Build an interactive dashboard using Power BI or Streamlit.
- Use a fitted pipeline for consistent scoring of new customers.
- Add customer lifetime value estimation.
- Explore advanced customer behavior features.

---

## 👩‍💻 Author

**Tushika Tibrewal**

B.Tech – Data Science and Business Systems  
SRM Institute of Science and Technology

---

## ⭐ Acknowledgements

- Kaggle – Online Retail Dataset
- Scikit-learn Documentation
- Pandas Documentation
