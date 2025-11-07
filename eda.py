# Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Create visuals directory if not exists
if not os.path.exists("visuals"):
    os.makedirs("visuals")

# -----------------------------
# 1. Load Data
# -----------------------------
def load_data():
    customer_cols = [
        'customer_id', 'Gender', 'age', 'driving_licence_present',
        'region_code', 'previously_insured', 'vehicle_age', 'vehicle_damage'
    ]
    policy_cols = [
        'customer_id', 'annual_premium', 'sales_channel_code',
        'vintage', 'response'
    ]

    customer_df = pd.read_csv('customer_details.csv', names=customer_cols, header=0)
    policy_df = pd.read_csv('customer_policy_details.csv', names=policy_cols, header=0)
    return customer_df, policy_df

# -----------------------------
# 2. Data Cleaning
# -----------------------------
def handle_nulls(df):
    print("Null Value Summary:\n", df.isnull().sum())

    df = df[df['customer_id'].notnull()].copy()

    # Fill numeric with mean
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))

    # Fill categorical with mode
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def handle_outliers(df):
    outlier_summary = {}
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_summary[col] = outliers.shape[0]

        # Replace outliers with mean
        mean_val = df[col].mean()
        df.loc[(df[col] < lower) | (df[col] > upper), col] = mean_val

    print("Outlier Summary:\n", outlier_summary)
    return df


def clean_strings(df):
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def remove_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates().copy()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows.")
    return df

# -----------------------------
# 3. Merge Datasets
# -----------------------------
def merge_data(customer_df, policy_df):
    master_df = pd.merge(customer_df, policy_df, on='customer_id', how='inner')
    print(f"Merged dataset shape: {master_df.shape}")
    return master_df

# -----------------------------
# 4. Generate Insights + Matplotlib Visuals
# -----------------------------
def generate_insights(df):
    # Gender-wise average premium
    gender_premium = df.groupby('Gender')['annual_premium'].mean()
    print("\nGender-wise Average Premium:\n", gender_premium)

    plt.figure(figsize=(6,4))
    gender_premium.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Gender-wise Average Premium')
    plt.ylabel('Average Premium')
    plt.tight_layout()
    plt.savefig('visuals/gender_premium_barplot.png')
    plt.show()

    # Age-wise average premium
    age_premium = df.groupby('age')['annual_premium'].mean()
    plt.figure(figsize=(8,4))
    age_premium.plot(kind='line', color='green')
    plt.title('Age-wise Premium Trend')
    plt.ylabel('Average Premium')
    plt.xlabel('Age')
    plt.tight_layout()
    plt.savefig('visuals/age_premium_lineplot.png')
    plt.show()

    # Gender distribution pie chart
    gender_counts = df['Gender'].value_counts()
    plt.figure(figsize=(5,5))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue','pink'])
    plt.title('Gender Distribution')
    plt.tight_layout()
    plt.savefig('visuals/gender_distribution_pie.png')
    plt.show()

    # Vehicle age-wise premium
    vehicle_premium = df.groupby('vehicle_age')['annual_premium'].mean()
    plt.figure(figsize=(6,4))
    vehicle_premium.plot(kind='bar', color='violet')
    plt.title('Vehicle Age-wise Premium')
    plt.ylabel('Average Premium')
    plt.tight_layout()
    plt.savefig('visuals/vehicle_age_premium_barplot.png')
    plt.show()

# -----------------------------
# 5. Correlation Analysis
# -----------------------------
def correlation_analysis(df):
    numeric_df = df.select_dtypes(include=np.number)

    if 'age' in numeric_df.columns and 'annual_premium' in numeric_df.columns:
        corr_value = numeric_df['age'].corr(numeric_df['annual_premium'])
        print(f"\nCorrelation between Age and Annual Premium: {corr_value:.2f}")
        if corr_value < -0.5:
            print("Strong Negative Relationship")
        elif corr_value > 0.5:
            print("Strong Positive Relationship")
        else:
            print("No Significant Relationship")

    plt.figure(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('visuals/correlation_heatmap.png')
    plt.show()

# -----------------------------
# 6. Main Function
# -----------------------------
def main():
    customer_df, policy_df = load_data()

    # Cleaning
    customer_df = handle_nulls(customer_df)
    policy_df = handle_nulls(policy_df)

    customer_df = handle_outliers(customer_df)
    policy_df = handle_outliers(policy_df)

    customer_df = clean_strings(customer_df)
    policy_df = clean_strings(policy_df)

    customer_df = remove_duplicates(customer_df)
    policy_df = remove_duplicates(policy_df)

    # Merge
    master_df = merge_data(customer_df, policy_df)

    # Save cleaned data
    master_df.to_csv('cleaned_master_table.csv', index=False)
    print("\n✅ Cleaned master table saved as 'cleaned_master_table.csv'")

    # Generate insights
    generate_insights(master_df)

    # Correlation Analysis
    correlation_analysis(master_df)


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    main()
