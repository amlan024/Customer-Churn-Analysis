
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the customer churn dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn information:")
print(df.info())

# Clean the data
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

# Check missing values and duplicates
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Calculate churn statistics
print("\nChurn counts:")
print(df["Churn"].value_counts())

churn_rate = (df["Churn"] == "Yes").mean() * 100
print(f"\nCustomer churn rate: {churn_rate:.2f}%")

# Visualize churn distribution
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("churn_distribution.png")
plt.show()

