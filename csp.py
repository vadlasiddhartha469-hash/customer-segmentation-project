import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

np.random.seed(42)
n_customers = 500

data = pd.DataFrame({
    'CustomerID': range(1001, 1001 + n_customers),
    'Age': np.random.randint(18, 70, size=n_customers),
    'AnnualIncome': np.random.randint(20000, 120000, size=n_customers),
    'SpendingScore': np.random.randint(1, 100, size=n_customers),
    'Recency': np.random.randint(1, 365, size=n_customers),
    'Frequency': np.random.randint(1, 50, size=n_customers)
})

features = ['Age', 'AnnualIncome', 'SpendingScore', 'Recency', 'Frequency']
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_summary = data.groupby('Cluster')[features].mean()
print("Cluster Profiles (Averages):\n", cluster_summary)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=data, 
    x='AnnualIncome', 
    y='SpendingScore', 
    hue='Cluster', 
    palette='viridis', 
    s=60
)
plt.title('Customer Segments: Income vs Spending Score')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Segment')
plt.tight_layout()
plt.savefig('customer_segments.png')
plt.show()