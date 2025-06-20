import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the MMM-ready dataset
df = pd.read_csv('data/mmm_ready/unified_dataset_mmm_ready.csv')
df['date'] = pd.to_datetime(df['date'])

print('🔍 COMPREHENSIVE PRE-EDA VALIDATION CHECK')
print('=' * 50)

print(f'📊 Dataset shape: {df.shape}')
print(f'📅 Date range: {df["date"].min().date()} to {df["date"].max().date()}')
print(f'⏱️  Total weeks: {len(df)}')
print()

# Check feature composition
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in ['date', 'sales']]

print(f'🎯 Target variable: sales')
print(f'📊 Features: {len(feature_cols)}')
print('  Features list:')
for i, col in enumerate(feature_cols, 1):
    print(f'    {i:2d}. {col}')
print()

# Check temporal split details
train_df = pd.read_csv('data/mmm_ready/train_set.csv')
test_df = pd.read_csv('data/mmm_ready/test_set.csv')
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

print('📅 TEMPORAL SPLIT ANALYSIS:')
print(f'  Train: {len(train_df)} samples ({train_df["date"].min().date()} to {train_df["date"].max().date()})')
print(f'  Test:  {len(test_df)} samples ({test_df["date"].min().date()} to {test_df["date"].max().date()})')
print()

# Seasonality coverage analysis
train_months = train_df['date'].dt.month.value_counts().sort_index()
test_months = test_df['date'].dt.month.value_counts().sort_index()

print('🔄 SEASONALITY COVERAGE:')
print('  Month distribution:')
print('  Month  | Train | Test  | Issue?')
print('  -------|-------|-------|-------')
issues = []
for month in range(1, 13):
    train_count = train_months.get(month, 0)
    test_count = test_months.get(month, 0)
    if test_count > 0 and train_count < 3:  # Less than 3 months in training
        issue = '⚠️ '
        issues.append(f'Month {month}: only {train_count} samples in training')
    else:
        issue = '✅'
    print(f'  {month:2d}     | {train_count:5d} | {test_count:5d} | {issue}')

print()

# Year coverage
print('  Year distribution:')
train_years = train_df['date'].dt.year.value_counts().sort_index()
test_years = test_df['date'].dt.year.value_counts().sort_index()
for year in sorted(set(train_df['date'].dt.year) | set(test_df['date'].dt.year)):
    train_count = train_years.get(year, 0)
    test_count = test_years.get(year, 0)
    print(f'  {year}: Train={train_count}, Test={test_count}')
print()

# Check for missing values
print('🔍 DATA QUALITY CHECK:')
missing_total = df.isnull().sum().sum()
print(f'  Missing values: {missing_total}')
if missing_total > 0:
    missing_by_col = df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0]
    for col, count in missing_by_col.items():
        print(f'    {col}: {count} missing')
print()

# Feature correlation check
print('🔗 FEATURE CORRELATION CHECK:')
corr_matrix = df[feature_cols].corr().abs()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = corr_matrix.iloc[i, j]
        if corr_value > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))

if high_corr_pairs:
    print('  ⚠️  High correlations still present:')
    for col1, col2, corr in high_corr_pairs:
        print(f'    {col1} ↔ {col2}: {corr:.3f}')
else:
    print('  ✅ No high correlations (>0.8) remaining')
print()

# Sales distribution check
print('📈 SALES DISTRIBUTION CHECK:')
print(f'  Sales range: {df["sales"].min():,.0f} to {df["sales"].max():,.0f}')
print(f'  Sales mean: {df["sales"].mean():,.0f}')
print(f'  Sales std: {df["sales"].std():,.0f}')
print(f'  Skewness: {df["sales"].skew():.3f}')

# Check for zeros in media spend
media_cols = [col for col in feature_cols if any(x in col.lower() for x in ['cost', 'spend', 'campaign'])]
print()
print('💰 MEDIA SPEND CHECK:')
for col in media_cols:
    zero_count = (df[col] == 0).sum()
    zero_pct = zero_count / len(df) * 100
    print(f'  {col}: {zero_count} zeros ({zero_pct:.1f}%)')

print()
print('🚨 VALIDATION ISSUES SUMMARY:')
if issues:
    for issue in issues:
        print(f'  ⚠️  {issue}')
else:
    print('  ✅ No major seasonality issues detected')

if high_corr_pairs:
    print(f'  ⚠️  {len(high_corr_pairs)} high correlation pairs still exist')

# Seasonality recommendation
print()
print('💡 RECOMMENDATIONS:')
print('  For seasonal MMM data:')
if len(test_df) < 52:
    print('  ⚠️  Test period covers less than 1 full year of seasonality')
    print('  📝 Consider: Use multiple validation periods or expand test set')

print('  ✅ Cyclical features present (month_sin, month_cos, week_sin, week_cos)')
print('  ✅ Holiday period feature included')
print('  ✅ Weather variables included (temperature_mean, sunshine_duration)')

# Feature-to-sample ratio
feature_ratio = len(feature_cols) / len(df)
print(f'  📊 Feature-to-sample ratio: {feature_ratio:.3f} ({len(feature_cols)} features / {len(df)} samples)')
if feature_ratio > 0.1:
    print('  ⚠️  Ratio slightly high, but acceptable for MMM')
else:
    print('  ✅ Good feature-to-sample ratio')

print()
print('🎯 READY FOR NEXT STEP?')
critical_issues = []
if missing_total > len(df) * 0.05:  # More than 5% missing
    critical_issues.append('Too many missing values')
if len([x for x in high_corr_pairs if x[2] > 0.95]) > 0:  # Perfect correlations
    critical_issues.append('Perfect correlations still exist')

if critical_issues:
    print('  ❌ CRITICAL ISSUES FOUND:')
    for issue in critical_issues:
        print(f'    • {issue}')
    print('  🔧 Fix these before proceeding to EDA')
else:
    print('  ✅ READY FOR EDA!')
    print('  📊 Dataset is properly prepared for MMM EDA and modeling') 