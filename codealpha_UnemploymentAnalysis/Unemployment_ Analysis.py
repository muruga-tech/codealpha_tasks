# ---------------------------------------------------------
# Unemployment Analysis 
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
data = pd.read_csv("unemployment.csv")

# Fix date parsing
data['date'] = pd.to_datetime(data['date'], dayfirst=True)

# Sort by date
data = data.sort_values(by="date")

# Extract year and month
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month_name()

# Target column renamed for convenience
data['unemployment_rate'] = data[' Estimated Unemployment Rate (%)']

print("\n--- Dataset Information ---")
print(data.info())
print("\n--- Summary Statistics ---")
print(data.describe(include='all'))

# 2. Plot unemployment trend
plt.figure(figsize=(12,5))
plt.plot(data['date'], data['unemployment_rate'], marker='o')
plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# 3. COVID-19 Analysis (2020 only in your data)
covid_period = data[data['year'] == 2020]

plt.figure(figsize=(12,5))
plt.plot(covid_period['date'], covid_period['unemployment_rate'], color='red', marker='o')
plt.title("Unemployment Rate During COVID-19 (2020)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# 4. Seasonal Trend (Monthly Average)
monthly_trend = data.groupby('month')['unemployment_rate'].mean()

# Reorder months
months_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
monthly_trend = monthly_trend.reindex(months_order)

plt.figure(figsize=(10,5))
sns.barplot(x=monthly_trend.index, y=monthly_trend.values)
plt.xticks(rotation=45)
plt.title("Average Unemployment Rate by Month")
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Month")
plt.show()

# 5. Yearly Trend (2019–2020)
yearly_trend = data.groupby('year')['unemployment_rate'].mean()

plt.figure(figsize=(10,5))
sns.lineplot(x=yearly_trend.index, y=yearly_trend.values, marker='o')
plt.title("Yearly Average Unemployment Rate")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# 6. Insights
print("\n--- KEY INSIGHTS ---")

# COVID impact
if len(covid_period) > 0:
    covid_increase = covid_period['unemployment_rate'].max() - covid_period['unemployment_rate'].min()
    print(f"• COVID-19 caused an unemployment change of {covid_increase:.2f}% in 2020.")
else:
    print("• No 2020 data found for COVID impact.")

# Seasonal peaks
peak_month = monthly_trend.idxmax()
low_month = monthly_trend.idxmin()

print(f"• Highest unemployment month: {peak_month}")
print(f"• Lowest unemployment month: {low_month}")

# Long-term change
yearly_change = yearly_trend.iloc[-1] - yearly_trend.iloc[0]
print(f"• Overall unemployment change from first to last year: {yearly_change:.2f}%")

print("\nAnalysis Completed Successfully!")
