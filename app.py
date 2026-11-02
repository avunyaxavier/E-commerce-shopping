import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-commerce Delivery Prediction",
    page_icon="📦",
    layout="wide"
)

DATA_FILE = "E-commerce_Delivery_Shipping_Data_2026.csv"
MODEL_FILE = "best_model.pkl"

# These transformations are copied from the user's notebook.
FLOAT_LOG_COLUMNS = [
    "product_weight_kg",
    "order_value_usd",
    "distance_km",
    "shipping_cost_usd",
    "warehouse_processing_hours",
]
INT_LOG_COLUMNS = ["delivery_attempts", "delivery_delay_days"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_resource
def rebuild_preprocessing():
    """Recreate the notebook's preprocessing using its original CSV.

    The notebook did not save its selector, scaler or OrdinalEncoder.
    This recreates them using the same split and preprocessing sequence.
    """
    df = pd.read_csv(DATA_FILE)

    required = {"late_delivery", "actual_delivery_days", "delivery_delay_days", "order_date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Same feature preparation as notebook cells 14-45.
    df_new = df.drop(
        columns=["order_id", "customer_id", "return_reason", "warehouse_id"],
        errors="ignore"
    ).copy()

    df_new["order_date"] = pd.to_datetime(df_new["order_date"])
    df_new["order_Year"] = df_new["order_date"].dt.year
    df_new["order_Month"] = df_new["order_date"].dt.month
    df_new["order_Day"] = df_new["order_date"].dt.day
    df_new.drop(columns=["order_date", "order_Year"], inplace=True)

    df_object = df_new.select_dtypes(include="object").copy()
    df_float = df_new.select_dtypes(include="float64").copy()
    df_int = df_new.select_dtypes(include="int64").copy()

    for col in FLOAT_LOG_COLUMNS:
        if col in df_float.columns:
            if (df_float[col] < 0).any():
                raise ValueError(f"Negative values found in {col}; log1p cannot be applied.")
            df_float[col] = np.log1p(df_float[col])

    for col in INT_LOG_COLUMNS:
        if col in df_int.columns:
            if (df_int[col] < 0).any():
                raise ValueError(f"Negative values found in {col}; log1p cannot be applied.")
            df_int[col] = np.log1p(df_int[col])

    df_num = pd.concat([df_float, df_int], axis=1)

    # Same IQR replacement as notebook cell 41.
    q1 = df_num.quantile(0.25)
    q3 = df_num.quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    median_value = df_num.median()

    df_iqr = df_num.mask(
        (df_num < lower_limit) | (df_num > upper_limit),
        median_value,
        axis=1
    )

    df_final = pd.concat([df_iqr, df_object], axis=1)
    X = df_final.drop(
        columns=["late_delivery", "actual_delivery_days", "delivery_delay_days"]
    )
    y = df_final[["late_delivery"]]

    # Same split as notebook cell 48.
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_tr_int = X_train.select_dtypes(exclude="object")
    X_tr_object = X_train.select_dtypes(include="object")

    # Same feature selection and scaling as notebook cells 52 and 54.
    selector = SelectKBest(score_func=f_classif, k=5)
    X_tr_selected = selector.fit_transform(X_tr_int, y_train)

    scaler = StandardScaler()
    scaler.fit(X_tr_selected)

    # Same categorical encoding as notebook cell 56.
    feature_encoder = OrdinalEncoder()
    feature_encoder.fit(X_tr_object)

    # Target encoder: notebook cell 59 reused the variable name 'od'.
    target_encoder = OrdinalEncoder()
    target_encoder.fit(y_train)

    selected_numeric_columns = X_tr_int.columns[selector.get_support()].tolist()
    categorical_columns = X_tr_object.columns.tolist()

    return {
        "selector": selector,
        "scaler": scaler,
        "feature_encoder": feature_encoder,
        "target_encoder": target_encoder,
        "numeric_columns": X_tr_int.columns.tolist(),
        "selected_numeric_columns": selected_numeric_columns,
        "categorical_columns": categorical_columns,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "median_value": median_value,
        "raw_numeric_medians": X_train.select_dtypes(exclude="object").median(),
        "raw_numeric_min": X_train.select_dtypes(exclude="object").min(),
        "raw_numeric_max": X_train.select_dtypes(exclude="object").max(),
    }


# ---------------------------------------------------------
# LOAD FILES
# ---------------------------------------------------------
try:
    model = load_model()
    prep = rebuild_preprocessing()
except FileNotFoundError as e:
    st.error(
        f"File not found: {e}. Keep app.py, best_model.pkl and "
        f"{DATA_FILE} in the same project folder."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not prepare the model: {e}")
    st.stop()

# Confirm that the recreated input order matches the saved model.
model_columns = list(getattr(model, "feature_names_in_", []))
expected_columns = prep["selected_numeric_columns"] + prep["categorical_columns"]
if model_columns and model_columns != expected_columns:
    st.error(
        "The recreated preprocessing columns do not match the saved model. "
        "Please confirm that the CSV is the exact dataset used in the notebook.\n\n"
        f"Model expects: {model_columns}\n\n"
        f"Preprocessing creates: {expected_columns}"
    )
    st.stop()

# ---------------------------------------------------------
# USER INTERFACE
# ---------------------------------------------------------
st.title("📦 E-commerce Delivery Prediction")
st.subheader("Late Delivery Classification")
st.write(
    "Enter the order details below. The saved Gradient Boosting model "
    "will predict whether the order is likely to be delivered late."
)
st.caption("Model: Gradient Boosting Classifier | Features: 5 numerical + 14 categorical")
st.divider()

with st.form("delivery_prediction_form"):
    st.markdown("### Numerical details")
    numeric_values = {}
    numeric_cols = st.columns(2)

    for index, feature in enumerate(prep["numeric_columns"]):
        default = prep["raw_numeric_medians"].get(feature, 0)
        min_val = prep["raw_numeric_min"].get(feature, None)
        max_val = prep["raw_numeric_max"].get(feature, None)
        default = float(default) if pd.notna(default) else 0.0
        # Keep inputs usable even if the training column has a constant value.
        if pd.notna(min_val) and pd.notna(max_val) and float(min_val) < float(max_val):
            value = min(max(default, float(min_val)), float(max_val))
            with numeric_cols[index % 2]:
                numeric_values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=value,
                    key=f"num_{feature}"
                )
        else:
            with numeric_cols[index % 2]:
                numeric_values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=default,
                    key=f"num_{feature}"
                )

    st.divider()
    st.markdown("### Categorical details")
    categorical_values = {}
    categorical_cols = st.columns(2)

    for index, feature in enumerate(prep["categorical_columns"]):
        categories = list(prep["feature_encoder"].categories_[index])
        categories = [x for x in categories if not pd.isna(x)]
        if not categories:
            st.error(f"No available categories found for {feature} in the training dataset.")
            st.stop()
        with categorical_cols[index % 2]:
            categorical_values[feature] = st.selectbox(
                feature.replace("_", " ").title(),
                options=categories,
                key=f"cat_{feature}"
            )

    submitted = st.form_submit_button(
        "🔍 Predict Delivery Status",
        use_container_width=True
    )

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if submitted:
    try:
        # Build numeric row in the same order used during training.
        numeric_input = pd.DataFrame(
            [[numeric_values[c] for c in prep["numeric_columns"]]],
            columns=prep["numeric_columns"]
        )

        # Apply notebook's log1p transformations before IQR replacement.
        for col in FLOAT_LOG_COLUMNS + ["delivery_attempts"]:
            if col in numeric_input.columns:
                if numeric_input.loc[0, col] < 0:
                    raise ValueError(f"{col} cannot be negative.")
                numeric_input[col] = np.log1p(numeric_input[col])

        # Apply the notebook's IQR outlier replacement values.
        numeric_input = numeric_input.clip(
            lower=prep["lower_limit"],
            upper=prep["upper_limit"],
            axis=1
        )
        numeric_input = numeric_input.fillna(prep["median_value"])

        # Feature selection and scaling.
        selected_numeric = prep["selector"].transform(numeric_input)
        scaled_numeric = prep["scaler"].transform(selected_numeric)
        numeric_result = pd.DataFrame(
            scaled_numeric,
            columns=prep["selected_numeric_columns"]
        )

        # Encode categorical values with the categories learned from training data.
        categorical_input = pd.DataFrame(
            [[categorical_values[c] for c in prep["categorical_columns"]]],
            columns=prep["categorical_columns"]
        )
        encoded_categorical = prep["feature_encoder"].transform(categorical_input)
        categorical_result = pd.DataFrame(
            encoded_categorical,
            columns=prep["categorical_columns"]
        )

        # Combine exactly as in notebook cell 57.
        final_input = pd.concat([numeric_result, categorical_result], axis=1)
        if model_columns:
            final_input = final_input.reindex(columns=model_columns)

        prediction_encoded = model.predict(final_input)
        prediction_label = prep["target_encoder"].inverse_transform(
            np.asarray(prediction_encoded).reshape(-1, 1)
        )[0, 0]

        st.divider()
        st.markdown("### Prediction result")

        label_text = str(prediction_label).strip().lower()
        if label_text in {"yes", "1", "true", "late", "1.0"}:
            st.error("⚠️ Late delivery predicted")
            st.write("The model predicts that this order may be delivered late.")
        elif label_text in {"no", "0", "false", "on time", "on_time", "0.0"}:
            st.success("✅ On-time delivery predicted")
            st.write("The model predicts that this order may be delivered on time.")
        else:
            st.info(f"Predicted class: {prediction_label}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(final_input)[0]
            st.markdown("#### Prediction confidence (model probabilities)")
            probability_df = pd.DataFrame({
                "Class": [str(x) for x in prep["target_encoder"].categories_[0]],
                "Probability": probabilities
            })
            st.dataframe(probability_df, use_container_width=True, hide_index=True)

        st.markdown("#### Submitted order details")
        submitted_details = {**numeric_values, **categorical_values}
        st.dataframe(pd.DataFrame([submitted_details]), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("This prediction is based on the patterns learned from the training dataset; it is not a guarantee of delivery outcome.")

