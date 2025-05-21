import pandas as pd
from functools import reduce

# File paths
file_paths = {
    "sales": "sales.xlsx",
    "tv_promo": "tv_promo.xlsx",
    "tv_branding": "tv_branding.xlsx",
    "radio_national": "radio_national.xlsx",
    "radio_local": "radio_local.xlsx",
    "social": "social.xlsx",
    "email": "email.xlsx",
    "ooh": "ooh.xlsx",
    "promo": "promo.xlsx"
}

# Load all Excel files
dataframes = {name: pd.read_excel(path) for name, path in file_paths.items()}

# Standardize inconsistent date column names
dataframes["tv_promo"].rename(columns={"datum": "date"}, inplace=True)
dataframes["tv_branding"].rename(columns={"datum": "date"}, inplace=True)
dataframes["radio_national"].rename(columns={"dag": "date"}, inplace=True)
dataframes["radio_local"].rename(columns={"dag": "date"}, inplace=True)

# Function to convert date to Monday and group weekly
def normalize(df, cols, agg='sum'):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='d')
    df['date'] = df['date'].dt.date
    if agg == 'sum':
        df = df.groupby('date')[cols].sum().reset_index()
    elif agg == 'mean':
        df = df.groupby('date')[cols].mean().reset_index()
    return df

# Normalize all datasets
sales_df = normalize(dataframes["sales"], ["sales"])
tv_promo_df = normalize(dataframes["tv_promo"], ["tv_promo_grps", "tv_promo_cost"])
tv_branding_df = normalize(dataframes["tv_branding"], ["tv_branding_grps", "tv_branding_cost"])
radio_national_df = normalize(dataframes["radio_national"], ["radio_national_grps", "radio_national_cost"])
radio_local_df = normalize(dataframes["radio_local"], ["radio_local_grps", "radio_local_cost"])
social_df = normalize(dataframes["social"].rename(columns={
    "impressions": "social_impressions", "costs": "social_costs"
}), ["social_impressions", "social_costs"])
email_df = normalize(dataframes["email"], ["email_campaigns"], agg='mean')
ooh_df = normalize(dataframes["ooh"], ["ooh_spend"])

# Promo: pivot promo types into columns and map to a single value
promo_df = dataframes["promo"].copy()
promo_df['date'] = pd.to_datetime(promo_df['date'], errors='coerce')
promo_df['date'] = promo_df['date'] - pd.to_timedelta(promo_df['date'].dt.weekday, unit='d')
promo_df['date'] = promo_df['date'].dt.date

# Create separate flags
promo_counts = promo_df.pivot_table(index="date", columns="promotion_type", aggfunc="size", fill_value=0).reset_index()
promo_counts.columns.name = None
promo_counts = promo_counts.rename(columns={
    "Buy One Get One": "buy_one_get_one",
    "Limited Time Offer": "limited_time_offer",
    "Price Discount": "price_discount"
})

# Assign a single promotion_type value
def classify_promo(row):
    if row.get("buy_one_get_one", 0) > 0:
        return 1
    elif row.get("limited_time_offer", 0) > 0:
        return 2
    elif row.get("price_discount", 0) > 0:
        return 3
    else:
        return 0

promo_counts["promotion_type"] = promo_counts.apply(classify_promo, axis=1)
promo_df_final = promo_counts[["date", "promotion_type"]]

# Merge all datasets
all_dfs = [
    sales_df, tv_promo_df, tv_branding_df, radio_national_df, radio_local_df,
    social_df, email_df, ooh_df, promo_df_final
]
combined_df = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), all_dfs)

# Filter from 2022 onwards
combined_df["date"] = pd.to_datetime(combined_df["date"], errors='coerce')
filtered_df = combined_df[combined_df["date"].dt.year >= 2022]

# Export final result
output_path = "combined_marketing_data.xlsx"
filtered_df.to_excel(output_path, index=False)

print("Final file created:", output_path)
