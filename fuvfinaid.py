import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import joblib

st.set_page_config(page_title="Financial Aid Estimator for Fulbright Students", layout="centered")

st.title("🎓 Fulbright University Vietnam Financial Aid Estimator")
st.markdown("This platform is exclusively designed for Fulbright applicants to estimate what **percentage of financial aid** you may receive based on your household data. Please fill in the form below to get your final result:")

TUITION_FEE = 500_000_000  # 500 million VND

# Define exchange rate
EXCHANGE_RATE = 468.54  # 1 PHP = 468.54 VND

# User inputs (in VND)
famsize_ip = st.slider("Family Size", 1, 15, 4)
income_total_ip = st.number_input("Total Annual Household Income (VND)", value=150_000_000)
cashrep_ip = st.number_input("Annual Cash Remittances (VND)", value=5_000_000)
rentals_nonagri_ip = st.number_input("Annual Non-Agricultural Rental Income (VND)", value=0)
income_ea_ip = st.number_input("Annual Earnings from Economic Activities (VND)", value=30_000_000)
interest_ip = st.number_input("Annual Interest Income (VND)", value=0)
pension_ip = st.number_input("Annual Pension Income (VND)", value=0)
dividends_ip = st.number_input("Annual Dividends (VND)", value=0)

food_ip = st.number_input("Annual Food Expenses (VND)", value=20_000_000)
clothing_ip = st.number_input("Annual Clothing Expenses (VND)", value=5_000_000)
housing_ip = st.number_input("Annual Housing Expenses (VND)", value=15_000_000)
health_ip = st.number_input("Annual Health Expenses (VND)", value=3_000_000)
transport_ip = st.number_input("Annual Transport Expenses (VND)", value=5_000_000)
communication_ip = st.number_input("Annual Communication Expenses (VND)", value=2_000_000)
recreation_ip = st.number_input("Annual Recreation Expenses (VND)", value=1_000_000)
education_ip = st.number_input("Annual Education Expenses (VND)", value=10_000_000)
misc_ip = st.number_input("Annual Miscellaneous Expenses (VND)", value=2_000_000)

dur_furniture_ip = st.number_input("Annual Durable Furniture (VND)", value=1_000_000)
cash_loan_ip = st.number_input("Annual Cash Loan Payments (VND)", value=0)
app_install_ip = st.number_input("Annual Appliance Installments (VND)", value=0)
veh_install_ip = st.number_input("Annual Vehicle Installments (VND)", value=0)
residence_ip = st.radio("Residence Type", ["Urban", "Rural"])
rural_ip = 1 if residence_ip == "Rural" else 0

# Load dataset (values in PHP)
df = pd.read_csv("ph_households_vF.csv")
df['cashrep'] = df['cashrep_abroad'] + df['cashrep_domestic']
df['rural'] = (df['residence'] == 'Rural').astype(int)
df = df.drop(['cashrep_abroad', 'cashrep_domestic', 'residence'], axis=1)
print(df.describe())
print(df.isnull().sum())

# Define variables
var_names = ['famsize', 'income_total', 'cashrep', 'rentals_nonagri', 
             'income_ea', 'interest', 'pension', 'dividends', 'food',
             'clothing', 'housing', 'health', 'transport', 'communication',
             'recreation', 'education', 'misc', 'dur_furniture', 'cash_loan',
             'app_install', 'veh_install', 'rural']

# Create binary rural column (Rural = 1, Urban = 0)
df = df.drop(columns=['ID', 'province', 'income_reg', 'income_ses'])

X = df[var_names]

# Verify dimensions
n, p = X.shape 
if p != len(var_names):
    raise ValueError(f"Mismatch: X has {p} columns, but var_names has {len(var_names)} elements")
print(f"Size of X: [{n}, {p}]")

# Standardize and fit PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=16)
X_pca = pca.fit_transform(X_scaled)

# Compute scores for financial aid percentage
ix_income = [var_names.index(name) for name in ['income_total', 'cashrep', 'rentals_nonagri', 'income_ea']]
ix_demo = [var_names.index(name) for name in ['famsize', 'rural']]
ix_expenses = [var_names.index(name) for name in ['food', 'housing', 'health', 'education', 'clothing', 'misc', 'transport', 'communication', 'recreation']]
ix_debt = [var_names.index(name) for name in ['cash_loan', 'app_install', 'veh_install']]
ix_passive_income = [var_names.index(name) for name in ['pension', 'dividends', 'interest']]

w_income = 0.30
w_demo = 0.10
w_exp = 0.25
w_debt = 0.15
w_assets = 0.20

# Standardize training data for score computation
X_standardized = scaler.transform(X)
score_income = -np.sum(X_standardized[:, ix_income], axis=1)
score_demo = np.sum(X_standardized[:, ix_demo], axis=1)
score_exp = np.sum(X_standardized[:, ix_expenses], axis=1)
score_debt = np.sum(X_standardized[:, ix_debt], axis=1)
score_assets = -np.sum(X_standardized[:, ix_passive_income], axis=1)

# Final score
aid_need_score = w_income * score_income + w_demo * score_demo + w_exp * score_exp + w_debt * score_debt + w_assets * score_assets

# Rescale
target_mean = 50
target_std = 25
current_mean = np.mean(aid_need_score)
current_std = np.std(aid_need_score)
FA_per = (aid_need_score - current_mean) / current_std  # z-score
FA_per = FA_per * target_std + target_mean  # rescale
FA_per = 100 * (FA_per - np.min(FA_per)) / (np.max(FA_per) - np.min(FA_per))
FA_per = np.clip(FA_per, 0, 100)  # Clip target variable before regression

# Fit regression model
reg = LinearRegression()
reg.fit(X_pca, FA_per)
beta_0 = reg.intercept_
beta = reg.coef_

print("Intercept:", beta_0)
print("Coefficient:", beta)

# Create input DataFrame (values in VND)
input_df = pd.DataFrame([[
    famsize_ip, income_total_ip, cashrep_ip, rentals_nonagri_ip, income_ea_ip, interest_ip,
    pension_ip, dividends_ip, food_ip, clothing_ip, housing_ip, health_ip, transport_ip,
    communication_ip, recreation_ip, education_ip, misc_ip, dur_furniture_ip, cash_loan_ip,
    app_install_ip, veh_install_ip, rural_ip
]], columns=var_names)

# Convert monetary features from VND to PHP
monetary_features = [
    'income_total', 'cashrep', 'rentals_nonagri', 'income_ea', 'interest', 'pension', 'dividends',
    'food', 'clothing', 'housing', 'health', 'transport', 'communication', 'recreation', 'education',
    'misc', 'dur_furniture', 'cash_loan', 'app_install', 'veh_install'
]
for feature in monetary_features:
    input_df[feature] = input_df[feature] / EXCHANGE_RATE

# Standardize new student data using the fitted scaler (which expects PHP values)
input_scaled = scaler.transform(input_df[var_names])
input_pca = pca.transform(input_scaled)

print("input_pca:", input_pca)
print("X_pca range:", np.min(X_pca, axis=0), np.max(X_pca, axis=0))

# Use trained regression coefficients
intercept = beta_0
pc_coeffs = beta

# Predict for training data
train_predictions = np.maximum(0, intercept + np.dot(X_pca, pc_coeffs))
print("Training predictions range:", np.min(train_predictions), np.max(train_predictions))

# Predict FA percentage for input
raw_input_prediction = intercept + np.dot(input_pca, pc_coeffs)
fa_percentage = np.maximum(0, raw_input_prediction)
fa_percentage = np.clip(fa_percentage.item(), 0, 100)

# Calculate VND amount (TUITION_FEE is already in VND)
final_aid_value = round(fa_percentage / 100 * TUITION_FEE)

# Debug
print("Raw input prediction:", raw_input_prediction)
print("input_scaled range:", np.min(input_scaled, axis=0), np.max(input_scaled, axis=0))
print("X_scaled range:", np.min(X_scaled, axis=0), np.max(X_scaled, axis=0))

# Display result
st.subheader("📊 Estimated Financial Aid Result")
st.success(f"Estimated Financial Aid: **{fa_percentage:.2f}%**")
st.markdown(f"🎯 This covers approximately **{final_aid_value:,.0f} VND** out of the tuition fee of {TUITION_FEE:,.0f} VND.")

st.caption("Note: This tool uses a PCA-based linear regression model. Actual financial aid decisions may consider additional factors.")
st.caption("Warning: Prediction is near bounds and may be less reliable.")
