
# ============================================================
# CUSTOMER SEGMENTATION USING PCA AND KMEANS
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Change the path if your file is in another folder
import pandas as pd
from pathlib import Path

file_path = (
    Path(__file__).parent / "Online_Retail.csv"
)

df = pd.read_csv(
    file_path,
    encoding="ISO-8859-1"
)

print(df.head())
print(df.shape)

# ============================================================
# 3. DATA CLEANING
# ============================================================

data = df.copy()

# Missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove missing CustomerID
data = data.dropna(subset=["CustomerID"])

# Remove duplicates
data = data.drop_duplicates()

# Remove invalid quantities and prices
data = data[
    (data["Quantity"] > 0) &
    (data["UnitPrice"] > 0)
]

# Remove cancelled invoices
data = data[
    ~data["InvoiceNo"].astype(str).str.startswith("C")
]

# Convert data types
data["CustomerID"] = data["CustomerID"].astype(str)

data["InvoiceDate"] = pd.to_datetime(
    data["InvoiceDate"]
)

# Create total amount
data["TotalAmount"] = (
    data["Quantity"] * data["UnitPrice"]
)

print("\nCleaned Dataset Shape:", data.shape)
print(data.head())


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# Reference date: day after last purchase
reference_date = (
    data["InvoiceDate"].max() +
    pd.Timedelta(days=1)
)

# Customer-level aggregation
customer_df = data.groupby("CustomerID").agg(

    TotalSpending=("TotalAmount", "sum"),

    NumTransactions=("InvoiceNo", "nunique"),

    LastPurchaseDate=("InvoiceDate", "max"),

    UniqueProducts=("StockCode", "nunique")

).reset_index()

# Recency
customer_df["Recency"] = (
    reference_date -
    customer_df["LastPurchaseDate"]
).dt.days

# Average order value
customer_df["AverageOrderValue"] = (
    customer_df["TotalSpending"] /
    customer_df["NumTransactions"]
)

# Required features
features = [
    "TotalSpending",
    "NumTransactions",
    "AverageOrderValue",
    "Recency",
    "UniqueProducts"
]

customer_features = customer_df[
    ["CustomerID"] + features
].copy()

print("\nCustomer Features:")
print(customer_features.head())

print("\nFeature Statistics:")
print(customer_features[features].describe())


# ============================================================
# 5. LOG TRANSFORMATION
# ============================================================

# Reduce the effect of extreme values
model_df = customer_features.copy()

for col in features:
    model_df[col] = np.log1p(
        model_df[col]
    )

print("\nAfter Log Transformation:")
print(model_df[features].head())


# ============================================================
# 6. STANDARD SCALER
# ============================================================

X = model_df[features]

# Before scaling
print("\n" + "=" * 50)
print("BEFORE SCALING")
print("=" * 50)

print("\nMean:")
print(X.mean())

print("\nStandard Deviation:")
print(X.std())

# Apply StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Convert to DataFrame
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)

# After scaling
print("\n" + "=" * 50)
print("AFTER SCALING")
print("=" * 50)

print("\nMean:")
print(X_scaled_df.mean())

print("\nStandard Deviation:")
print(X_scaled_df.std())

print("\nScaled Data Shape:", X_scaled.shape)


# ============================================================
# 7. PCA WITH 2 COMPONENTS
# ============================================================

pca_2 = PCA(n_components=2)

X_pca_2 = pca_2.fit_transform(
    X_scaled
)

# PCA DataFrame
pca_df = pd.DataFrame(
    X_pca_2,
    columns=["PC1", "PC2"]
)

# Explained variance ratio
print("\n" + "=" * 50)
print("PCA WITH 2 COMPONENTS")
print("=" * 50)

print("\nExplained Variance Ratio:")
print(pca_2.explained_variance_ratio_)

print(
    "\nTotal Explained Variance:",
    pca_2.explained_variance_ratio_.sum()
)

print("\nPCA Data:")
print(pca_df.head())


# ============================================================
# 8. 2D PCA SCATTER PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("2D PCA Scatter Plot")

plt.grid(True)
plt.show()


# ============================================================
# 9. PCA WITH 1 COMPONENT - CUSTOMER VALUE SCORE
# ============================================================

pca_1 = PCA(n_components=1)

customer_score = pca_1.fit_transform(
    X_scaled
).ravel()

# Orient score so spending has positive loading
spending_idx = features.index(
    "TotalSpending"
)

if pca_1.components_[0, spending_idx] < 0:
    customer_score = -customer_score

# Add score
customer_features["Score"] = customer_score

# Rank customers
ranked_customers = customer_features.sort_values(
    by="Score",
    ascending=False
).reset_index(drop=True)

ranked_customers["Rank"] = (
    ranked_customers.index + 1
)

print("\n" + "=" * 50)
print("CUSTOMER VALUE SCORE")
print("=" * 50)

print(ranked_customers.head(10))


# ============================================================
# 10. KMEANS WITH 3 CLUSTERS
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(
    X_scaled
)

# Add cluster labels
customer_features["Cluster"] = cluster_labels

print("\n" + "=" * 50)
print("CLUSTER DISTRIBUTION")
print("=" * 50)

print(
    customer_features["Cluster"].value_counts()
    .sort_index()
)


# ============================================================
# 11. CLUSTER CENTROIDS AND ANALYSIS
# ============================================================

# Original-scale cluster summary
cluster_original_summary = (
    customer_features
    .groupby("Cluster")[features]
    .mean()
)

print("\n" + "=" * 50)
print("CLUSTER SUMMARY (ORIGINAL SCALE)")
print("=" * 50)

print(cluster_original_summary)


# ============================================================
# 12. MAP CLUSTERS TO NAMES
# ============================================================

# Cluster centroids in standardized feature space
centroids = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=features
)

# Business-oriented value indicator
# Spending, transactions, AOV and unique products:
# higher is generally associated with customer value.
# Recency: lower is more recent, so subtract it.

centroids["BusinessValue"] = (
    centroids["TotalSpending"] +
    centroids["NumTransactions"] +
    centroids["AverageOrderValue"] +
    centroids["UniqueProducts"] -
    centroids["Recency"]
)

# Sort cluster IDs by business value
ordered_clusters = (
    centroids
    .sort_values(
        "BusinessValue",
        ascending=False
    )
    .index
    .tolist()
)

cluster_names = [
    "High-Value Regulars",
    "Occasional Buyers",
    "One-Time / Low-Value Shoppers"
]

cluster_mapping = {
    cluster_id: cluster_names[i]
    for i, cluster_id in enumerate(
        ordered_clusters
    )
}

# Map labels to names
customer_features["Segment"] = (
    customer_features["Cluster"]
    .map(cluster_mapping)
)

print("\n" + "=" * 50)
print("CLUSTER MAPPING")
print("=" * 50)

print(cluster_mapping)

print("\nCustomer Segments:")
print(
    customer_features[
        ["CustomerID", "Cluster", "Segment"]
    ].head(10)
)


# ============================================================
# 13. PCA SCATTER PLOT COLORED BY CLUSTER
# ============================================================

pca_plot_df = pca_df.copy()

pca_plot_df["Cluster"] = cluster_labels

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=pca_plot_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set1",
    s=60,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Customer Segmentation Using PCA and KMeans"
)

plt.legend(title="Cluster")
plt.grid(True)

plt.show()


# ============================================================
# 14. ELBOW PLOT (K = 2 TO 10)
# ============================================================

inertias = []

K_range = range(2, 11)

for k in K_range:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)

    inertias.append(km.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))

plt.plot(
    list(K_range),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")

plt.title("Elbow Method for Optimal k")

plt.xticks(list(K_range))
plt.grid(True)

plt.show()


# ============================================================
# 15. SILHOUETTE SCORE
# ============================================================

silhouette_scores = []

for k in K_range:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(10, 6))

plt.plot(
    list(K_range),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")

plt.title(
    "Silhouette Score for Different k Values"
)

plt.xticks(list(K_range))
plt.grid(True)

plt.show()


# ============================================================
# 16. CRITICAL EXERCISE
#     score_customers(df)
# ============================================================

def score_customers(df):

    """
    Takes a raw Online Retail transaction DataFrame
    and returns a DataFrame containing:

    customer_id
    score
    segment
    """

    data = df.copy()

    # ------------------------------
    # DATA CLEANING
    # ------------------------------

    data = data.dropna(
        subset=["CustomerID"]
    )

    data = data.drop_duplicates()

    data = data[
        (data["Quantity"] > 0) &
        (data["UnitPrice"] > 0)
    ]

    data = data[
        ~data["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    ]

    data["CustomerID"] = (
        data["CustomerID"].astype(str)
    )

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"]
    )

    data["TotalAmount"] = (
        data["Quantity"] *
        data["UnitPrice"]
    )

    # ------------------------------
    # FEATURE ENGINEERING
    # ------------------------------

    reference_date = (
        data["InvoiceDate"].max() +
        pd.Timedelta(days=1)
    )

    customer_df = data.groupby(
        "CustomerID"
    ).agg(

        TotalSpending=("TotalAmount", "sum"),

        NumTransactions=("InvoiceNo", "nunique"),

        LastPurchaseDate=("InvoiceDate", "max"),

        UniqueProducts=("StockCode", "nunique")

    ).reset_index()

    customer_df["Recency"] = (
        reference_date -
        customer_df["LastPurchaseDate"]
    ).dt.days

    customer_df["AverageOrderValue"] = (
        customer_df["TotalSpending"] /
        customer_df["NumTransactions"]
    )

    features = [
        "TotalSpending",
        "NumTransactions",
        "AverageOrderValue",
        "Recency",
        "UniqueProducts"
    ]

    X = customer_df[features].copy()

    # ------------------------------
    # LOG TRANSFORMATION
    # ------------------------------

    for col in features:
        X[col] = np.log1p(X[col])

    # ------------------------------
    # STANDARD SCALER
    # ------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ------------------------------
    # PCA SCORE
    # ------------------------------

    pca = PCA(n_components=1)

    scores = pca.fit_transform(
        X_scaled
    ).ravel()

    spending_idx = features.index(
        "TotalSpending"
    )

    if pca.components_[0, spending_idx] < 0:
        scores = -scores

    # ------------------------------
    # KMEANS
    # ------------------------------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        X_scaled
    )

    # ------------------------------
    # CLUSTER NAMING
    # ------------------------------

    centroids = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=features
    )

    centroids["BusinessValue"] = (
        centroids["TotalSpending"] +
        centroids["NumTransactions"] +
        centroids["AverageOrderValue"] +
        centroids["UniqueProducts"] -
        centroids["Recency"]
    )

    ordered_clusters = (
        centroids
        .sort_values(
            "BusinessValue",
            ascending=False
        )
        .index
        .tolist()
    )

    cluster_names = [
        "High-Value Regulars",
        "Occasional Buyers",
        "One-Time / Low-Value Shoppers"
    ]

    cluster_mapping = {
        cluster_id: cluster_names[i]
        for i, cluster_id in enumerate(
            ordered_clusters
        )
    }

    # ------------------------------
    # FINAL RESULT
    # ------------------------------

    result = pd.DataFrame({

        "customer_id":
            customer_df["CustomerID"],

        "score":
            scores,

        "segment": [
            cluster_mapping[c]
            for c in clusters
        ]

    })

    return result


# ============================================================
# 17. RUN THE SCORING FUNCTION
# ============================================================

result = score_customers(df)

print("\n" + "=" * 50)
print("FINAL CUSTOMER SCORING OUTPUT")
print("=" * 50)

print(result.head(10))

print("\nOutput Columns:")
print(result.columns.tolist())

print("\nSegment Distribution:")
print(result["segment"].value_counts())


# ============================================================
# 18. SAVE RESULTS
# ============================================================

result.to_csv(
    "customer_scores_segments.csv",
    index=False
)

print(
    "\nResults saved as "
    "customer_scores_segments.csv"
)


# ============================================================
# 19. BUSINESS ANALYSIS
# ============================================================

# Merge score and segment information
final_df = customer_features.copy()

# Segment summary
segment_summary = (
    final_df
    .groupby("Segment")[features]
    .mean()
)

print("\n" + "=" * 50)
print("BUSINESS ANALYSIS")
print("=" * 50)

print(segment_summary)

# Number of customers per segment
print("\nCustomer Count per Segment:")

print(
    final_df["Segment"].value_counts()
)


# ============================================================
# 20. FINAL DATASET
# ============================================================

final_output = final_df[
    [
        "CustomerID",
        "Score",
        "Cluster",
        "Segment"
    ] + features
].copy()

print("\nFinal Output:")
print(final_output.head())

# Save complete customer segmentation
final_output.to_csv(
    "complete_customer_segmentation.csv",
    index=False
)

print(
    "\nComplete segmentation saved successfully!"
)