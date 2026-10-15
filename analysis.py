
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


# Business insights: churn by contract and tenure

# Churn rate by contract type
contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\nChurn rate by contract type (%):")
print(contract_churn.round(2))

contract_churn.plot(kind="bar", rot=0)
plt.title("Customer Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("churn_by_contract.png")
plt.close()

# Churn rate by customer tenure
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months",
            "25-48 months", "49-72 months"]
)

tenure_churn = (
    df.groupby("TenureGroup", observed=False)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

print("\nChurn rate by tenure group (%):")
print(tenure_churn.round(2))

tenure_churn.plot(kind="bar", rot=0)
plt.title("Customer Churn Rate by Tenure")
plt.xlabel("Customer Tenure")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("churn_by_tenure.png")
plt.close()
