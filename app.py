import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

st.set_page_config(
    page_title="Corrosion ML Dashboard",
    layout="wide"
)

sns.set_style("whitegrid")

st.title("Corrosion Rate Prediction Dashboard")

st.markdown("""
This dashboard analyzes industrial corrosion data using
Machine Learning and Data Visualization.
""")

@st.cache_data
def load_dataset():

    df = pd.read_excel(
        "synthetic_corrosion_ml_dataset.xlsx"
    )

    return df

df = load_dataset()

st.success("Dataset loaded successfully")

st.header("Dataset Overview")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Split Distribution")
    st.write(df["split"].value_counts())

with col2:

    st.subheader("Data Types")
    st.write(df.dtypes)

st.subheader("First 5 Rows")
st.dataframe(df.head())

st.header("Corrosion Rate Analysis")

fig1, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

axes[0].hist(
    df["corrosion_rate_mpy"],
    bins=30,
    color="royalblue",
    edgecolor="black"
)

axes[0].axvline(
    df["corrosion_rate_mpy"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {df['corrosion_rate_mpy'].mean():.2f}"
)

axes[0].axvline(
    df["corrosion_rate_mpy"].median(),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median = {df['corrosion_rate_mpy'].median():.2f}"
)

axes[0].set_title(
    "Corrosion Rate Distribution"
)

axes[0].set_xlabel(
    "Corrosion Rate"
)

axes[0].set_ylabel(
    "Frequency"
)

axes[0].legend()

environment_corr = (
    df.groupby("environment")[
        "corrosion_rate_mpy"
    ].mean()
)

axes[1].pie(
    environment_corr,
    labels=environment_corr.index,
    autopct="%1.1f%%"
)

axes[1].set_title(
    "Environment Contribution"
)

plt.tight_layout()

st.pyplot(fig1)

st.header("Feature Correlation Analysis")

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

corr = df[numeric_cols].corr()

target_corr = (
    corr["corrosion_rate_mpy"]
    .drop("corrosion_rate_mpy")
    .sort_values(ascending=False)
)

fig2, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

bar_colors = [
    "steelblue" if v > 0
    else "salmon"
    for v in target_corr
]

target_corr.plot(
    kind="barh",
    color=bar_colors,
    ax=axes[0]
)

axes[0].set_title(
    "Feature Correlation"
)

axes[0].axvline(
    0,
    color="black",
    linewidth=1
)

top_feats = (
    target_corr.abs()
    .nlargest(10)
    .index
    .tolist()
)

top_feats.append(
    "corrosion_rate_mpy"
)

sns.heatmap(
    df[top_feats].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=1,
    ax=axes[1]
)

axes[1].set_title(
    "Correlation Heatmap"
)

plt.tight_layout()

st.pyplot(fig2)

st.header("Data Preprocessing")

processed_df = pd.get_dummies(
    df,
    drop_first=True
)

X = processed_df.drop(
    "corrosion_rate_mpy",
    axis=1
)

y = processed_df[
    "corrosion_rate_mpy"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

st.write("X_train Shape :", X_train.shape)
st.write("X_test Shape :", X_test.shape)

st.header("Model Performance")

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_preds = linear_model.predict(
    X_test_scaled
)

linear_mae = mean_absolute_error(
    y_test,
    linear_preds
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_preds
    )
)

linear_r2 = r2_score(
    y_test,
    linear_preds
)

rf_model = RandomForestRegressor(

    n_estimators=40,

    max_depth=10,

    min_samples_split=10,

    min_samples_leaf=4,

    random_state=42

)

rf_model.fit(
    X_train,
    y_train
)

rf_preds = rf_model.predict(
    X_test
)

rf_mae = mean_absolute_error(
    y_test,
    rf_preds
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_preds
    )
)

rf_r2 = r2_score(
    y_test,
    rf_preds
)

xgb_model = XGBRegressor(

    n_estimators=1500,

    learning_rate=0.01,

    max_depth=15,

    min_child_weight=1,

    subsample=1,

    colsample_bytree=1,

    gamma=0,

    reg_alpha=0,

    reg_lambda=0,

    objective="reg:squarederror",

    random_state=42

)

xgb_model.fit(
    X_train,
    y_train
)

xgb_preds = xgb_model.predict(
    X_test
)

xgb_mae = mean_absolute_error(
    y_test,
    xgb_preds
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_preds
    )
)

xgb_r2 = r2_score(
    y_test,
    xgb_preds
)

results_df = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],

    "MAE": [
        linear_mae,
        rf_mae,
        xgb_mae
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse,
        xgb_rmse
    ],

    "R2 Score": [
        linear_r2,
        rf_r2,
        xgb_r2
    ]
})

st.dataframe(results_df)

fig3, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

axes[0].scatter(
    y_test,
    linear_preds,
    s=10
)

axes[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

axes[0].set_title(
    "Linear Regression"
)

axes[1].scatter(
    y_test,
    rf_preds,
    s=10
)

axes[1].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

axes[1].set_title(
    "Random Forest"
)

axes[2].scatter(
    y_test,
    xgb_preds,
    s=10
)

axes[2].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

axes[2].set_title(
    "XGBoost"
)

plt.tight_layout()

st.pyplot(fig3)

st.header("Industrial Insights")

coating_corr = (
    df.groupby("coating")[
        "corrosion_rate_mpy"
    ].mean()
)

material_cost = (
    df.groupby("material")[
        "total_cost_usd"
    ].mean()
)

fig4, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

axes[0].pie(
    coating_corr,
    labels=coating_corr.index,
    autopct="%1.1f%%"
)

axes[0].set_title(
    "Coating Type"
)

axes[1].pie(
    material_cost,
    labels=material_cost.index,
    autopct="%1.1f%%"
)

axes[1].set_title(
    "Material Cost"
)

plt.tight_layout()

st.pyplot(fig4)

comparison = [
    df["corrosion_rate_mpy"].mean(),
    df["mpy_after_mitigation"].mean()
]

labels = [
    "Before",
    "After"
]

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": xgb_model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

top10 = importance_df.head(10)

fig5, axes = plt.subplots(
    1,
    2,
    figsize=(13, 4)
)

axes[0].bar(
    labels,
    comparison
)

axes[0].set_title(
    "Mitigation Impact"
)

axes[0].set_ylabel(
    "Average Corrosion Rate"
)

axes[1].barh(
    top10["Feature"][::-1],
    top10["Importance"][::-1],
    color="mediumseagreen"
)

axes[1].set_title(
    "Top Important Features"
)

plt.tight_layout()

st.pyplot(fig5)

st.header("Final Observations")

st.markdown("""

1. CO2 PIPELINE WET environment produced the highest corrosion rate.

2. Wet industrial environments accelerated corrosion rapidly.

3. 3LPE coating showed better corrosion protection efficiency.

4. DUPLEX 2205 and TITANIUM GRADE 2 provided strong anti corrosion performance.

5. Carbon Steel showed higher corrosion vulnerability.

6. Corrosion mitigation significantly reduced corrosion rate.

7. XGBoost achieved highly accurate corrosion prediction.

8. Machine Learning can support predictive maintenance and industrial safety.

""")