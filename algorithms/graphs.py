import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
results_df = pd.read_csv("algorithm_comparison_results.csv")

# Example: Boxplot for Execution Time
plt.figure(figsize=(10, 6))
sns.boxplot(x="Algorithm", y="Execution Time (s)", data=results_df)
plt.title("Execution Time Comparison")
plt.xticks(rotation=45)
plt.show()

# Example: Boxplot for Difference from Optimal Cost
plt.figure(figsize=(10, 6))
sns.boxplot(x="Algorithm", y="Difference", data=results_df)
plt.title("Difference from Optimal Cost Comparison")
plt.xticks(rotation=45)
plt.show()

# Additional analysis as required...
