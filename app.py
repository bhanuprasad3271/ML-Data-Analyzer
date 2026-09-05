import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)
from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
from sklearn.svm import (
    SVR,
    SVC
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================================
# PAGE
# ==========================================================

st.set_page_config(
    page_title="ML Data Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Data Analyzer")
st.write(
    "Upload a dataset, select a target, and let the application "
    "automatically prepare and train Machine Learning models."
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def clean_numeric_series(series):
    """
    Convert numeric-looking text into numbers.

    Examples:
        '16GB'      -> 16
        '16 GB'     -> 16
        '₹59,999'   -> 59999
        '59,999'    -> 59999
        '$1,299'    -> 1299
        '2.5 kg'    -> 2.5
    """

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    )

    return pd.to_numeric(cleaned, errors="coerce")


def detect_numeric_columns(df):
    """
    Detect columns that are actually numeric or mostly numeric.
    """

    numeric_columns = []

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        non_null = series.dropna()

        if len(non_null) == 0:
            continue

        converted = clean_numeric_series(non_null)

        success_rate = converted.notna().mean()

        # At least 95% of non-empty values must be numeric-looking
        if success_rate >= 0.95:
            numeric_columns.append(column)

    return numeric_columns


def prepare_dataframe(df):
    """
    Clean numeric-looking columns while leaving real categorical
    columns as categorical/text.
    """

    df = df.copy()

    numeric_columns = detect_numeric_columns(df)

    for column in numeric_columns:

        original = df[column]

        # Only convert object/string columns
        if not pd.api.types.is_numeric_dtype(original):

            converted = clean_numeric_series(original)

            success_rate = converted.notna().mean()

            if success_rate >= 0.95:
                df[column] = converted

    return df, numeric_columns


def detect_problem_type(series):
    """
    Automatically determine Classification or Regression.

    Numeric target:
        > 10 unique values -> Regression
        <= 10 unique values -> Classification

    Non-numeric target:
        Classification
    """

    non_null = series.dropna()

    if len(non_null) == 0:
        return None

    if pd.api.types.is_numeric_dtype(non_null):

        unique_count = non_null.nunique()

        if unique_count > 10:
            return "Regression"

        return "Classification"

    return "Classification"


def validate_classification_target(y):
    """
    Check whether classification target is suitable.
    """

    class_counts = y.value_counts()

    if len(class_counts) < 2:
        return False, "Classification requires at least 2 different classes."

    if class_counts.min() < 2:

        rare_classes = class_counts[
            class_counts < 2
        ].index.tolist()

        return False, (
            "Some classes contain only one record. "
            "Each class needs at least 2 records."
        )

    # Prevent extremely high-cardinality classification
    unique_count = y.nunique()
    row_count = len(y)

    if unique_count > 50 and unique_count > row_count * 0.20:

        return False, (
            "This target contains too many unique classes "
            "for reliable classification."
        )

    return True, ""


def create_one_hot_encoder():
    """
    Compatible with different scikit-learn versions.
    """

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


# ==========================================================
# 1. UPLOAD DATASET
# ==========================================================

st.header("Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a dataset",
    type=[
        "csv",
        "xlsx",
        "xls",
        "json"
    ]
)


if uploaded_file is not None:

    # ======================================================
    # READ DATASET
    # ======================================================

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        elif file_name.endswith(
            (".xlsx", ".xls")
        ):

            df = pd.read_excel(
                uploaded_file
            )

        elif file_name.endswith(".json"):

            df = pd.read_json(
                uploaded_file
            )

        else:

            st.error(
                "Unsupported file format."
            )

            st.stop()

    except Exception as e:

        st.error(
            f"Could not read dataset: {e}"
        )

        st.stop()


    # ======================================================
    # BASIC VALIDATION
    # ======================================================

    if df.empty:

        st.error(
            "The uploaded dataset is empty."
        )

        st.stop()


    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )


    # Remove completely empty rows
    df = df.dropna(
        axis=0,
        how="all"
    )


    if df.shape[1] < 2:

        st.error(
            "Dataset must contain at least "
            "one feature column and one target column."
        )

        st.stop()


    # ======================================================
    # CLEAN DATA TYPES
    # ======================================================

    df, detected_numeric_columns = prepare_dataframe(
        df
    )


    st.success(
        f"Dataset uploaded successfully: "
        f"{uploaded_file.name}"
    )


    # ======================================================
    # DATASET INFORMATION
    # ======================================================

    st.subheader(
        "Dataset Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Numeric Columns",
        len(detected_numeric_columns)
    )

    col4.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )


    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # ======================================================
    # 2. SELECT TARGET
    # ======================================================

    st.header(
        "Select Target Column"
    )

    target = st.selectbox(
        "What do you want to predict?",
        df.columns
    )


    # ======================================================
    # TARGET VALIDATION
    # ======================================================

    target_data = df[target].dropna()

    if len(target_data) == 0:

        st.error(
            "The selected target column contains no usable data."
        )

        st.stop()


    unique_target_values = target_data.nunique()


    # ======================================================
    # 3. AUTOMATIC PROBLEM DETECTION
    # ======================================================

    problem_type = detect_problem_type(
        df[target]
    )


    st.header(
        "Detected Problem Type"
    )


    if problem_type == "Regression":

        st.success(
            "✓ Regression — automatically detected"
        )

        st.info(
            f"Target contains {unique_target_values} "
            "different numeric values."
        )


    elif problem_type == "Classification":

        st.success(
            "✓ Classification — automatically detected"
        )

        st.info(
            f"Target contains {unique_target_values} "
            "different classes/values."
        )


    else:

        st.error(
            "Could not determine the problem type."
        )

        st.stop()


    # ======================================================
    # SHOW TARGET INFORMATION
    # ======================================================

    with st.expander(
        "🔍 Target Information"
    ):

        st.write(
            "Data type:",
            str(df[target].dtype)
        )

        st.write(
            "Number of records:",
            len(target_data)
        )

        st.write(
            "Unique values:",
            unique_target_values
        )

        st.write(
            "Sample values:"
        )

        st.write(
            target_data.head(10).tolist()
        )


    # ======================================================
    # CLASSIFICATION VALIDATION
    # ======================================================

    if problem_type == "Classification":

        classification_ok, classification_message = \
            validate_classification_target(
                target_data
            )

        if not classification_ok:

            st.error(
                f"❌ Target validation failed: "
                f"{classification_message}"
            )

            st.warning(
                "Please select another target column "
                "with repeated classes."
            )

            st.subheader(
                "Class Distribution"
            )

            class_distribution = (
                target_data
                .value_counts()
                .rename("Records")
            )

            st.dataframe(
                class_distribution,
                use_container_width=True
            )

            st.stop()


    # ======================================================
    # ALGORITHMS
    # ======================================================

    if problem_type == "Regression":

        algorithms = {

            "Linear Regression":
                LinearRegression(),

            "Decision Tree Regressor":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "Random Forest Regressor":
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),

            "SVR":
                SVR()
        }

    else:

        algorithms = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=3000
                ),

            "KNN":
                KNeighborsClassifier(
                    n_neighbors=5
                ),

            "Decision Tree Classifier":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest Classifier":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),

            "SVM":
                SVC(),

            "Naive Bayes":
                GaussianNB()
        }


    # ======================================================
    # 4. SELECT ALGORITHMS
    # ======================================================

    st.header(
        "Select Algorithm(s)"
    )

    selected_algorithms = []


    for algorithm_name in algorithms:

        if st.checkbox(
            algorithm_name,
            key=f"algorithm_{algorithm_name}"
        ):

            selected_algorithms.append(
                algorithm_name
            )


    # ======================================================
    # 5. TRAIN / TEST SPLIT
    # ======================================================

    st.header(
        "Training / Testing Split"
    )

    split_option = st.radio(
        "Choose split",
        [
            "70% / 30%",
            "80% / 20% ⭐ Recommended",
            "90% / 10%",
            "Custom"
        ],
        horizontal=True
    )


    if split_option == "70% / 30%":

        train_percentage = 70

    elif split_option == "80% / 20% ⭐ Recommended":

        train_percentage = 80

    elif split_option == "90% / 10%":

        train_percentage = 90

    else:

        train_percentage = st.slider(
            "Training Percentage",
            min_value=10,
            max_value=90,
            value=80
        )


    test_percentage = (
        100 - train_percentage
    )


    st.info(
        f"Training: **{train_percentage}%** | "
        f"Testing: **{test_percentage}%**"
    )


    # ======================================================
    # 6. TRAIN MODELS
    # ======================================================

    if st.button(
        "🚀 Train Selected Models",
        type="primary"
    ):

        # --------------------------------------------------
        # CHECK ALGORITHM
        # --------------------------------------------------

        if len(selected_algorithms) == 0:

            st.warning(
                "Please select at least one algorithm."
            )

            st.stop()


        # --------------------------------------------------
        # REMOVE MISSING TARGET ROWS
        # --------------------------------------------------

        data = df.dropna(
            subset=[target]
        ).copy()


        if len(data) < 10:

            st.error(
                "Dataset has too few usable rows. "
                "At least 10 rows are recommended."
            )

            st.stop()


        # --------------------------------------------------
        # X AND Y
        # --------------------------------------------------

        X = data.drop(
            columns=[target]
        )

        y_original = data[target]


        # --------------------------------------------------
        # CLASSIFICATION TARGET
        # --------------------------------------------------

        label_encoder = None

        if problem_type == "Classification":

            label_encoder = LabelEncoder()

            y = label_encoder.fit_transform(
                y_original.astype(str)
            )

        else:

            y = pd.to_numeric(
                y_original,
                errors="coerce"
            )

            valid_rows = y.notna()

            X = X.loc[
                valid_rows
            ].copy()

            y = y.loc[
                valid_rows
            ]


        # --------------------------------------------------
        # CHECK FEATURE COLUMNS
        # --------------------------------------------------

        if X.shape[1] == 0:

            st.error(
                "No feature columns are available "
                "after selecting the target."
            )

            st.stop()


        # --------------------------------------------------
        # DETECT FEATURE TYPES
        # --------------------------------------------------

        numeric_columns = X.select_dtypes(
            include=np.number
        ).columns.tolist()


        categorical_columns = X.select_dtypes(
            exclude=np.number
        ).columns.tolist()


        # --------------------------------------------------
        # NUMERIC PIPELINE
        # --------------------------------------------------

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )


        # --------------------------------------------------
        # CATEGORICAL PIPELINE
        # --------------------------------------------------

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    create_one_hot_encoder()
                )
            ]
        )


        # --------------------------------------------------
        # PREPROCESSOR
        # --------------------------------------------------

        transformers = []


        if numeric_columns:

            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns
                )
            )


        if categorical_columns:

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns
                )
            )


        if not transformers:

            st.error(
                "No usable input features were found."
            )

            st.stop()


        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )


        # --------------------------------------------------
        # TRAIN / TEST SPLIT
        # --------------------------------------------------

        try:

            test_size = (
                test_percentage / 100
            )


            if problem_type == "Classification":

                class_counts = pd.Series(
                    y
                ).value_counts()

                number_of_classes = len(
                    class_counts
                )

                estimated_test_rows = int(
                    np.ceil(
                        len(y) * test_size
                    )
                )

                estimated_train_rows = (
                    len(y) -
                    estimated_test_rows
                )


                # Stratification is possible only when
                # both sets can contain every class.
                can_stratify = (
                    class_counts.min() >= 2
                    and
                    estimated_test_rows >= number_of_classes
                    and
                    estimated_train_rows >= number_of_classes
                )


                if can_stratify:

                    X_train, X_test, y_train, y_test = \
                        train_test_split(
                            X,
                            y,
                            test_size=test_size,
                            random_state=42,
                            stratify=y
                        )

                else:

                    st.warning(
                        "⚠️ Stratified splitting is not "
                        "possible with this dataset/split. "
                        "Using a normal random split instead."
                    )

                    X_train, X_test, y_train, y_test = \
                        train_test_split(
                            X,
                            y,
                            test_size=test_size,
                            random_state=42
                        )


            else:

                X_train, X_test, y_train, y_test = \
                    train_test_split(
                        X,
                        y,
                        test_size=test_size,
                        random_state=42
                    )


        except Exception as e:

            st.error(
                f"Could not split dataset: {e}"
            )

            st.stop()


        # --------------------------------------------------
        # MODEL TRAINING
        # --------------------------------------------------

        st.header(
            "Model Results"
        )


        results = []

        trained_models = {}


        for algorithm_name in selected_algorithms:

            model = algorithms[
                algorithm_name
            ]


            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "model",
                        model
                    )
                ]
            )


            try:

                with st.spinner(
                    f"Training {algorithm_name}..."
                ):

                    pipeline.fit(
                        X_train,
                        y_train
                    )


                predictions = pipeline.predict(
                    X_test
                )


                # Save successful model
                trained_models[
                    algorithm_name
                ] = pipeline


                # ==================================================
                # CLASSIFICATION RESULTS
                # ==================================================

                if problem_type == "Classification":

                    accuracy = accuracy_score(
                        y_test,
                        predictions
                    )

                    precision = precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )

                    recall = recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )

                    f1 = f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )


                    results.append({

                        "Algorithm":
                            algorithm_name,

                        "Accuracy (%)":
                            round(
                                accuracy * 100,
                                2
                            ),

                        "Precision (%)":
                            round(
                                precision * 100,
                                2
                            ),

                        "Recall (%)":
                            round(
                                recall * 100,
                                2
                            ),

                        "F1 Score (%)":
                            round(
                                f1 * 100,
                                2
                            )
                    })


                # ==================================================
                # REGRESSION RESULTS
                # ==================================================

                else:

                    r2 = r2_score(
                        y_test,
                        predictions
                    )

                    rmse = np.sqrt(
                        mean_squared_error(
                            y_test,
                            predictions
                        )
                    )


                    results.append({

                        "Algorithm":
                            algorithm_name,

                        "R² Score [ Bad 0 <----> 1 Perfet ]":
                            round(
                                r2,
                                4
                            ),

                        "RMSE [ Low <----> High] Error":
                            round(
                                rmse,
                                2
                            )
                    })


            except Exception as e:

                # One failed model should NOT
                # stop the entire application.

                st.warning(
                    f"⚠️ {algorithm_name} "
                    f"could not be trained: {e}"
                )


        # ======================================================
        # SHOW RESULTS
        # ======================================================

        if not results:

            st.error(
                "❌ None of the selected algorithms "
                "could be trained successfully."
            )

            st.stop()


        result_df = pd.DataFrame(
            results
        )


        st.dataframe(
            result_df,
            use_container_width=True
        )


        # ======================================================
        # BEST MODEL
        # ======================================================

        if problem_type == "Classification":

            best_index = result_df[
                "Accuracy (%)"
            ].idxmax()

        else:

            best_index = result_df[
                "R² Score [ Bad 0 <----> 1 Perfet ]"
            ].idxmax()


        best_model_name = result_df.loc[
            best_index,
            "Algorithm"
        ]


        best_model = trained_models[
            best_model_name
        ]


        st.success(
            f"🏆 Best Model: "
            f"**{best_model_name}**"
        )


        # ======================================================
        # SAVE MODEL INFORMATION
        # ======================================================

        st.session_state[
            "trained_models"
        ] = trained_models


        st.session_state[
            "best_model"
        ] = best_model


        st.session_state[
            "features"
        ] = X.columns.tolist()


        st.session_state[
            "numeric_features"
        ] = numeric_columns


        st.session_state[
            "categorical_features"
        ] = categorical_columns


        st.session_state[
            "target"
        ] = target


        st.session_state[
            "problem_type"
        ] = problem_type


        st.session_state[
            "label_encoder"
        ] = label_encoder


        st.session_state[
            "trained"
        ] = True


        # ======================================================
        # DATA QUALITY INFORMATION
        # ======================================================

        st.subheader(
            "📊 Data Used for Training"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Training Rows",
            len(X_train)
        )

        c2.metric(
            "Testing Rows",
            len(X_test)
        )

        c3.metric(
            "Input Features",
            X.shape[1]
        )


    # ==========================================================
    # 7. PREDICTION
    # ==========================================================

    if st.session_state.get(
        "trained",
        False
    ):

        st.header(
            "Make a Prediction"
        )

        st.write(
            "Enter values for the input features."
        )


        input_data = {}


        for column in st.session_state[
            "features"
        ]:

            if column in st.session_state[
                "numeric_features"
            ]:

                value = st.text_input(
                    f"{column} (numeric)",
                    key=f"prediction_{column}"
                )

            else:

                value = st.text_input(
                    f"{column}",
                    key=f"prediction_{column}"
                )


            input_data[
                column
            ] = value


        if st.button(
            "🔮 Predict"
        ):

            # --------------------------------------------------
            # PREPARE INPUT
            # --------------------------------------------------

            input_df = pd.DataFrame(
                [input_data]
            )


            # Convert numeric features
            for column in st.session_state[
                "numeric_features"
            ]:

                raw_value = input_df.loc[
                    0,
                    column
                ]


                if str(raw_value).strip() == "":

                    input_df.loc[
                        0,
                        column
                    ] = np.nan

                else:

                    converted = clean_numeric_series(
                        pd.Series([raw_value])
                    ).iloc[0]


                    input_df.loc[
                        0,
                        column
                    ] = converted


            # --------------------------------------------------
            # PREDICT
            # --------------------------------------------------

            try:

                prediction = st.session_state[
                    "best_model"
                ].predict(
                    input_df
                )


                # ==================================================
                # CLASSIFICATION PREDICTION
                # ==================================================

                if st.session_state[
                    "problem_type"
                ] == "Classification":

                    encoder = st.session_state[
                        "label_encoder"
                    ]


                    prediction = encoder.inverse_transform(
                        prediction.astype(int)
                    )


                    st.success(
                        f"🎯 Prediction: "
                        f"**{prediction[0]}**"
                    )


                # ==================================================
                # REGRESSION PREDICTION
                # ==================================================

                else:

                    predicted_value = prediction[0]


                    st.success(
                        f"🎯 Prediction: "
                        f"**{predicted_value:.2f}**"
                    )


            except Exception as e:

                st.error(
                    f"❌ Prediction failed: {e}"
                )