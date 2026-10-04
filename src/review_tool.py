import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
from datetime import datetime
from io import BytesIO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Bengali ML QA Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = "data/raw"
MODEL_PATH = "models/bengali_ml_qa_model.pkl"
DB_PATH = "data/raw/ml_qa_lab.db"

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            project_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT UNIQUE,
            description TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT,
            label_name TEXT,
            created_at TEXT,
            UNIQUE(project_name, label_name)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT,
            record_id TEXT,
            text TEXT,
            label TEXT,
            reviewer_decision TEXT,
            reviewer_comment TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT,
            record_id TEXT,
            text TEXT,
            expected_label TEXT,
            predicted_label TEXT,
            confidence REAL,
            automatic_result TEXT,
            qa_decision TEXT,
            comment TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS acceptance_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT,
            decision TEXT,
            comment TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "project": None,
    "dataset": None,
    "annotation_index": 0,
    "prediction_ready": False,
    "model_df": None,
    "labels": []
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_project_labels(project_name):

    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT label_name
        FROM project_labels
        WHERE project_name = ?
        ORDER BY id
        """,
        conn,
        params=(project_name,)
    )

    conn.close()

    return df["label_name"].tolist()


def save_project_label(project_name, label_name):

    conn = get_connection()

    conn.execute(
        """
        INSERT OR IGNORE INTO project_labels
        (project_name, label_name, created_at)
        VALUES (?, ?, ?)
        """,
        (
            project_name,
            label_name,
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()


def create_excel_report(reviews, annotations, model_results, metrics_df, acceptance_df):

    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:

        if model_results is not None and len(model_results) > 0:
            model_results.to_excel(
                writer,
                index=False,
                sheet_name="Model Evaluation"
            )

        if len(reviews) > 0:
            reviews.to_excel(
                writer,
                index=False,
                sheet_name="Human Reviews"
            )

        if len(annotations) > 0:
            annotations.to_excel(
                writer,
                index=False,
                sheet_name="Annotations"
            )

        if metrics_df is not None and len(metrics_df) > 0:
            metrics_df.to_excel(
                writer,
                index=False,
                sheet_name="QA Metrics"
            )

        if len(acceptance_df) > 0:
            acceptance_df.to_excel(
                writer,
                index=False,
                sheet_name="Acceptance"
            )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧪 Bengali ML QA Lab")
st.sidebar.caption("ML QA Engineer Simulation")

menu = st.sidebar.radio(
    "WORKSPACE",
    [
        "🏠 Project Home",
        "📂 Dataset & Annotation",
        "🤖 Model Evaluation",
        "👀 Human Review",
        "📊 QA Metrics",
        "🗄️ SQL Data QA",
        "📑 Export & Acceptance"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title("🧪 Bengali ML QA Lab")

if st.session_state.project:

    st.caption(
        f"Active Project: **{st.session_state.project}**"
    )


# ============================================================
# PROJECT HOME
# ============================================================

if menu == "🏠 Project Home":

    st.subheader("🏗️ Create ML QA Project")

    with st.form("project_form"):

        project_name = st.text_input(
            "Project Name",
            placeholder="Example: Bengali Text Classification QA"
        )

        description = st.text_area(
            "Project Description",
            placeholder="Describe the ML QA objective..."
        )

        submitted = st.form_submit_button(
            "🚀 Create Project"
        )

        if submitted:

            if not project_name.strip():

                st.error(
                    "Project name is required."
                )

            else:

                conn = get_connection()

                try:

                    conn.execute(
                        """
                        INSERT INTO projects
                        (project_name, description, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (
                            project_name.strip(),
                            description,
                            datetime.now().isoformat()
                        )
                    )

                    conn.commit()

                    st.session_state.project = project_name.strip()

                    st.success(
                        f"Project '{project_name}' created successfully!"
                    )

                except sqlite3.IntegrityError:

                    st.warning(
                        "Project already exists. It is now active."
                    )

                    st.session_state.project = project_name.strip()

                finally:

                    conn.close()


    # --------------------------------------------------------
    # LABEL CONFIGURATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("🏷️ Project Label Configuration")

    if not st.session_state.project:

        st.info(
            "Create or select a project first."
        )

    else:

        st.write(
            "Create the labels that will be used during manual annotation."
        )

        label_input = st.text_input(
            "New Label",
            placeholder="Example: VALID_BENGALI"
        )

        if st.button("➕ Add Label"):

            label = label_input.strip()

            if not label:

                st.warning(
                    "Enter a label name first."
                )

            else:

                save_project_label(
                    st.session_state.project,
                    label
                )

                st.success(
                    f"Label '{label}' added."
                )

                st.rerun()


        current_labels = get_project_labels(
            st.session_state.project
        )

        if current_labels:

            st.write("### Current Labels")

            for label in current_labels:

                st.write(f"🔹 {label}")

            st.session_state.labels = current_labels

        else:

            st.info(
                "No labels created yet."
            )


    # --------------------------------------------------------
    # EXISTING PROJECTS
    # --------------------------------------------------------

    st.divider()

    st.subheader("📋 Existing Projects")

    conn = get_connection()

    projects = pd.read_sql_query(
        """
        SELECT *
        FROM projects
        ORDER BY project_id DESC
        """,
        conn
    )

    conn.close()

    if len(projects) > 0:

        st.dataframe(
            projects,
            use_container_width=True,
            hide_index=True
        )

        project_choices = projects["project_name"].tolist()

        selected_project = st.selectbox(
            "Open Existing Project",
            project_choices
        )

        if st.button("📂 Open Project"):

            st.session_state.project = selected_project

            st.session_state.labels = get_project_labels(
                selected_project
            )

            st.success(
                f"Active project: {selected_project}"
            )

            st.rerun()

    else:

        st.info(
            "No projects created yet."
        )


# ============================================================
# DATASET & ANNOTATION
# ============================================================

elif menu == "📂 Dataset & Annotation":

    st.subheader("📂 Import Dataset")

    if not st.session_state.project:

        st.warning(
            "Create/open a project first from Project Home."
        )

    else:

        uploaded = st.file_uploader(
            "Upload CSV Dataset",
            type=["csv"],
            key="dataset_uploader"
        )

        if uploaded:

            dataset = pd.read_csv(uploaded)

            # Keep imported dataset in session
            st.session_state.dataset = dataset

            st.success(
                f"Dataset imported: {len(dataset)} records"
            )

            # ------------------------------------------------
            # DATASET PREVIEW
            # ------------------------------------------------

            st.subheader("🔎 Dataset Preview")

            st.dataframe(
                dataset.head(20),
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # DATASET INFORMATION
            # ------------------------------------------------

            st.divider()

            st.subheader("📊 Dataset Information")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Records",
                len(dataset)
            )

            c2.metric(
                "Columns",
                len(dataset.columns)
            )

            if "expected_label" in dataset.columns:

                c3.metric(
                    "Expected Label Types",
                    dataset["expected_label"].nunique()
                )

            elif "label" in dataset.columns:

                c3.metric(
                    "Existing Label Types",
                    dataset["label"].nunique()
                )

            # ------------------------------------------------
            # MANUAL ANNOTATION
            # ------------------------------------------------

            st.divider()

            st.subheader("✏️ Manual Annotation Workspace")

            if "text" not in dataset.columns:

                st.error(
                    "Dataset must contain a 'text' column."
                )

            else:

                labels = get_project_labels(
                    st.session_state.project
                )

                if not labels:

                    st.warning(
                        "No labels configured. "
                        "Create labels from Project Home first."
                    )

                else:

                    total_records = len(dataset)

                    # Safety check
                    if (
                        st.session_state.annotation_index < 0
                        or
                        st.session_state.annotation_index >= total_records
                    ):

                        st.session_state.annotation_index = 0

                    index = st.session_state.annotation_index

                    row = dataset.iloc[index]

                    st.markdown(
                        f"### Record {index + 1} / {total_records}"
                    )

                    # Progress
                    st.progress(
                        (index + 1) / total_records
                    )

                    st.markdown("### 📥 Text")

                    st.info(
                        str(row["text"])
                    )

                    st.caption(
                        "Make your independent QA annotation. "
                        "The original expected label is intentionally hidden."
                    )

                    selected_label = st.selectbox(
                        "Select Correct Label",
                        labels,
                        key=f"annotation_label_{index}"
                    )

                    comment = st.text_area(
                        "Annotation Comment",
                        placeholder="Explain your annotation decision...",
                        key=f"annotation_comment_{index}"
                    )

                    st.divider()

                    col1, col2, col3 = st.columns(3)

                    # ------------------------------------------------
                    # PREVIOUS
                    # ------------------------------------------------

                    with col1:

                        if st.button(
                            "⬅️ Previous",
                            key=f"previous_{index}"
                        ):

                            if index > 0:

                                st.session_state.annotation_index = index - 1

                                st.rerun()

                    # ------------------------------------------------
                    # SAVE + NEXT
                    # ------------------------------------------------

                    with col2:

                        if st.button(
                            "💾 Save & Next",
                            type="primary",
                            key=f"save_next_{index}"
                        ):

                            conn = get_connection()

                            record_id = str(
                                row.get(
                                    "id",
                                    index + 1
                                )
                            )

                            # Prevent duplicate annotation
                            existing = pd.read_sql_query(
                                """
                                SELECT id
                                FROM annotations
                                WHERE project_name = ?
                                AND record_id = ?
                                """,
                                conn,
                                params=(
                                    st.session_state.project,
                                    record_id
                                )
                            )

                            if len(existing) > 0:

                                conn.execute(
                                    """
                                    UPDATE annotations
                                    SET
                                        text = ?,
                                        label = ?,
                                        reviewer_decision = ?,
                                        reviewer_comment = ?,
                                        created_at = ?
                                    WHERE project_name = ?
                                    AND record_id = ?
                                    """,
                                    (
                                        str(row["text"]),
                                        selected_label,
                                        "ANNOTATED",
                                        comment,
                                        datetime.now().isoformat(),
                                        st.session_state.project,
                                        record_id
                                    )
                                )

                            else:

                                conn.execute(
                                    """
                                    INSERT INTO annotations
                                    (
                                        project_name,
                                        record_id,
                                        text,
                                        label,
                                        reviewer_decision,
                                        reviewer_comment,
                                        created_at
                                    )
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        st.session_state.project,
                                        record_id,
                                        str(row["text"]),
                                        selected_label,
                                        "ANNOTATED",
                                        comment,
                                        datetime.now().isoformat()
                                    )
                                )

                            conn.commit()
                            conn.close()

                            # Move to next record
                            if index < total_records - 1:

                                st.session_state.annotation_index = index + 1

                                st.success(
                                    "Annotation saved. Moving to next record..."
                                )

                                st.rerun()

                            else:

                                st.success(
                                    "🎉 All records have been reached. "
                                    "Annotation session complete!"
                                )

                    # ------------------------------------------------
                    # SKIP
                    # ------------------------------------------------

                    with col3:

                        if st.button(
                            "➡️ Skip",
                            key=f"skip_{index}"
                        ):

                            if index < total_records - 1:

                                st.session_state.annotation_index = index + 1

                                st.rerun()

                            else:

                                st.info(
                                    "This is the last record."
                                )

                    # ------------------------------------------------
                    # ANNOTATION PROGRESS
                    # ------------------------------------------------

                    conn = get_connection()

                    annotation_count = pd.read_sql_query(
                        """
                        SELECT COUNT(*) AS total
                        FROM annotations
                        WHERE project_name = ?
                        """,
                        conn,
                        params=(st.session_state.project,)
                    ).iloc[0]["total"]

                    conn.close()

                    st.divider()

                    st.metric(
                        "Saved Annotations",
                        int(annotation_count)
                    )

# ============================================================
# MODEL EVALUATION
# ============================================================

elif menu == "🤖 Model Evaluation":

    st.subheader("🤖 Model Evaluation")

    if st.session_state.dataset is None:

        st.warning(
            "Import a dataset first from Dataset & Annotation."
        )

    elif not os.path.exists(MODEL_PATH):

        st.error(
            "ML model not found."
        )

    else:

        dataset = st.session_state.dataset

        st.write(
            f"**Test Records:** {len(dataset)}"
        )

        st.write(
            "**Model:** Bengali ML QA Classification Model"
        )

        st.info(
            "The model runs only when you explicitly start evaluation."
        )

        if st.button(
            "🚀 Run Model Evaluation",
            type="primary"
        ):

            model = joblib.load(MODEL_PATH)

            predictions = model.predict(
                dataset["text"]
            )

            probabilities = model.predict_proba(
                dataset["text"]
            )

            confidence = probabilities.max(axis=1)

            result = dataset.copy()

            result["predicted_label"] = predictions

            result["confidence"] = confidence

            # ------------------------------------------------
            # Ground Truth
            # ------------------------------------------------

            expected_column = None

            # First preference:
            # manually saved annotations

            conn = get_connection()

            annotations = pd.read_sql_query(
                """
                SELECT record_id, label
                FROM annotations
                WHERE project_name = ?
                """,
                conn,
                params=(st.session_state.project,)
            )

            conn.close()

            if len(annotations) > 0:

                annotation_map = dict(
                    zip(
                        annotations["record_id"].astype(str),
                        annotations["label"]
                    )
                )

                result["expected_label"] = result[
                    "id"
                ].astype(str).map(annotation_map)

                expected_column = "expected_label"

            # If no manual annotation exists yet,
            # use expected_label/label only as supporting dataset
            # information.

            elif "expected_label" in result.columns:

                expected_column = "expected_label"

            elif "label" in result.columns:

                expected_column = "label"

            if expected_column:

                result["automatic_result"] = (
                    result[expected_column] ==
                    result["predicted_label"]
                ).map({
                    True: "PASS",
                    False: "FAIL"
                })

            st.session_state.model_df = result

            st.session_state.prediction_ready = True

            st.success(
                "Model evaluation completed."
            )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        if st.session_state.prediction_ready:

            result = st.session_state.model_df

            st.subheader("🔍 Model Output")

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            # Quick summary

            if "automatic_result" in result.columns:

                total = len(result)

                passed = (
                    result["automatic_result"] == "PASS"
                ).sum()

                failed = (
                    result["automatic_result"] == "FAIL"
                ).sum()

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total",
                    total
                )

                c2.metric(
                    "PASS",
                    passed
                )

                c3.metric(
                    "FAIL",
                    failed
                )


# ============================================================
# HUMAN REVIEW
# ============================================================

elif menu == "👀 Human Review":

    st.subheader("👀 Human Review Queue")

    if not st.session_state.prediction_ready:

        st.warning(
            "Run Model Evaluation first."
        )

    else:

        df = st.session_state.model_df

        # -----------------------------------------------
        # Filters
        # -----------------------------------------------

        review_filter = st.selectbox(
            "Review Filter",
            [
                "Low Confidence",
                "Model FAIL",
                "All Cases"
            ]
        )

        if review_filter == "Low Confidence":

            review_df = df[
                df["confidence"] < 0.70
            ].copy()

        elif review_filter == "Model FAIL":

            if "automatic_result" in df.columns:

                review_df = df[
                    df["automatic_result"] == "FAIL"
                ].copy()

            else:

                review_df = df.copy()

        else:

            review_df = df.copy()

        st.metric(
            "Cases in Review Queue",
            len(review_df)
        )

        if len(review_df) == 0:

            st.success(
                "No cases match this review filter."
            )

        else:

            display_columns = [
                "id",
                "text",
                "predicted_label",
                "confidence"
            ]

            if "expected_label" in review_df.columns:

                display_columns.insert(
                    2,
                    "expected_label"
                )

            if "automatic_result" in review_df.columns:

                display_columns.append(
                    "automatic_result"
                )

            st.dataframe(
                review_df[display_columns],
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            selected_id = st.selectbox(
                "Select Test Case for Human Review",
                review_df["id"].tolist()
            )

            row = review_df[
                review_df["id"] == selected_id
            ].iloc[0]

            st.markdown("### 📥 Input")

            st.info(
                str(row["text"])
            )

            c1, c2, c3 = st.columns(3)

            if "expected_label" in row:

                c1.metric(
                    "Expected",
                    row["expected_label"]
                )

            else:

                c1.metric(
                    "Expected",
                    "Not Available"
                )

            c2.metric(
                "Model Prediction",
                row["predicted_label"]
            )

            c3.metric(
                "Confidence",
                f"{row['confidence']:.2%}"
            )

            st.divider()

            st.markdown(
                "### 👨‍⚖️ Your QA Decision"
            )

            decision = st.radio(
                "Review Result",
                [
                    "Correct",
                    "Incorrect",
                    "Borderline"
                ]
            )

            comment = st.text_area(
                "QA Comment",
                placeholder="Explain your reasoning..."
            )

            if st.button(
                "💾 Submit Human Review",
                type="primary"
            ):

                expected_value = None

                if "expected_label" in row:

                    expected_value = row["expected_label"]

                automatic_value = None

                if "automatic_result" in row:

                    automatic_value = row["automatic_result"]

                conn = get_connection()

                conn.execute(
                    """
                    INSERT INTO model_reviews
                    (
                        project_name,
                        record_id,
                        text,
                        expected_label,
                        predicted_label,
                        confidence,
                        automatic_result,
                        qa_decision,
                        comment,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        st.session_state.project,
                        str(row["id"]),
                        str(row["text"]),
                        expected_value,
                        str(row["predicted_label"]),
                        float(row["confidence"]),
                        automatic_value,
                        decision,
                        comment,
                        datetime.now().isoformat()
                    )
                )

                conn.commit()
                conn.close()

                st.success(
                    "Human QA review saved successfully."
                )


# ============================================================
# QA METRICS
# ============================================================

elif menu == "📊 QA Metrics":

    st.subheader("📊 Model QA Metrics")

    if not st.session_state.prediction_ready:

        st.warning(
            "Run Model Evaluation first."
        )

    else:

        df = st.session_state.model_df

        if "expected_label" not in df.columns:

            st.warning(
                "Expected labels are not available yet. "
                "Complete annotation or provide expected labels first."
            )

        else:

            clean_df = df.dropna(
                subset=[
                    "expected_label",
                    "predicted_label"
                ]
            )

            if len(clean_df) == 0:

                st.warning(
                    "No comparable labelled records found."
                )

            else:

                y_true = clean_df[
                    "expected_label"
                ]

                y_pred = clean_df[
                    "predicted_label"
                ]

                accuracy = accuracy_score(
                    y_true,
                    y_pred
                )

                precision = precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )

                recall = recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )

                f1 = f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Accuracy",
                    f"{accuracy:.2%}"
                )

                c2.metric(
                    "Macro Precision",
                    f"{precision:.2%}"
                )

                c3.metric(
                    "Macro Recall",
                    f"{recall:.2%}"
                )

                c4.metric(
                    "Macro F1",
                    f"{f1:.2%}"
                )

                # -----------------------------------------
                # Metrics Table
                # -----------------------------------------

                metrics_df = pd.DataFrame({
                    "Metric": [
                        "Accuracy",
                        "Macro Precision",
                        "Macro Recall",
                        "Macro F1"
                    ],
                    "Value": [
                        accuracy,
                        precision,
                        recall,
                        f1
                    ]
                })

                st.divider()

                st.subheader(
                    "📋 Metrics Summary"
                )

                st.dataframe(
                    metrics_df,
                    use_container_width=True,
                    hide_index=True
                )

                # -----------------------------------------
                # Confusion Matrix
                # -----------------------------------------

                st.divider()

                st.subheader(
                    "🎯 Confusion Matrix"
                )

                labels = sorted(
                    set(y_true) | set(y_pred)
                )

                matrix = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=labels
                )

                matrix_df = pd.DataFrame(
                    matrix,
                    index=labels,
                    columns=labels
                )

                st.dataframe(
                    matrix_df,
                    use_container_width=True
                )


# ============================================================
# SQL DATA QA
# ============================================================

elif menu == "🗄️ SQL Data QA":

    st.subheader("🗄️ SQL Data QA Console")

    conn = get_connection()

    tables = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name
        """,
        conn
    )

    st.write("### Available Tables")

    st.dataframe(
        tables,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    query = st.text_area(
        "SQL Query",
        value="SELECT * FROM model_reviews LIMIT 20;",
        height=140
    )

    st.caption(
        "Use SELECT queries for QA investigation."
    )

    if st.button(
        "▶️ Execute SQL",
        type="primary"
    ):

        try:

            cleaned_query = query.strip().lower()

            if not cleaned_query.startswith("select"):

                st.error(
                    "For this QA console, use SELECT queries only."
                )

            else:

                result = pd.read_sql_query(
                    query,
                    conn
                )

                st.success(
                    f"{len(result)} rows returned."
                )

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.error(
                f"SQL Error: {e}"
            )

    conn.close()


# ============================================================
# EXPORT & ACCEPTANCE
# ============================================================

elif menu == "📑 Export & Acceptance":

    st.subheader("📑 QA Reports & Release Decision")

    # --------------------------------------------------------
    # LOAD ALL QA DATA
    # --------------------------------------------------------

    conn = get_connection()

    reviews = pd.read_sql_query(
        """
        SELECT *
        FROM model_reviews
        ORDER BY id
        """,
        conn
    )

    annotations = pd.read_sql_query(
        """
        SELECT *
        FROM annotations
        ORDER BY id
        """,
        conn
    )

    acceptance = pd.read_sql_query(
        """
        SELECT *
        FROM acceptance_decisions
        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    model_results = st.session_state.model_df

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_model_cases = (
        len(model_results)
        if model_results is not None
        else 0
    )

    total_reviews = len(reviews)

    total_annotations = len(annotations)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model Cases",
        total_model_cases
    )

    c2.metric(
        "Human Reviews",
        total_reviews
    )

    c3.metric(
        "Annotations",
        total_annotations
    )

    # --------------------------------------------------------
    # MODEL RESULTS EXPORT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🤖 Model Evaluation Report"
    )

    if model_results is not None:

        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True
        )

        model_csv = model_results.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "⬇️ Download Model Evaluation CSV",
            model_csv,
            "model_evaluation_report.csv",
            "text/csv"
        )

    else:

        st.info(
            "Run model evaluation to generate this report."
        )


    # --------------------------------------------------------
    # HUMAN REVIEW EXPORT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "👀 Human Review Report"
    )

    if len(reviews) > 0:

        st.dataframe(
            reviews,
            use_container_width=True,
            hide_index=True
        )

        review_csv = reviews.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "⬇️ Download Human Review CSV",
            review_csv,
            "human_review_report.csv",
            "text/csv"
        )

    else:

        st.info(
            "No human reviews completed yet."
        )


    # --------------------------------------------------------
    # ANNOTATION EXPORT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "✏️ Annotation Report"
    )

    if len(annotations) > 0:

        st.dataframe(
            annotations,
            use_container_width=True,
            hide_index=True
        )

        annotation_csv = annotations.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "⬇️ Download Annotation CSV",
            annotation_csv,
            "annotation_report.csv",
            "text/csv"
        )

    else:

        st.info(
            "No manual annotations completed yet."
        )


    # --------------------------------------------------------
    # METRICS EXPORT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📊 QA Metrics Report"
    )

    metrics_df = pd.DataFrame()

    if (
        model_results is not None
        and "expected_label" in model_results.columns
    ):

        metric_df_clean = model_results.dropna(
            subset=[
                "expected_label",
                "predicted_label"
            ]
        )

        if len(metric_df_clean) > 0:

            y_true = metric_df_clean[
                "expected_label"
            ]

            y_pred = metric_df_clean[
                "predicted_label"
            ]

            metrics_df = pd.DataFrame({
                "Metric": [
                    "Accuracy",
                    "Macro Precision",
                    "Macro Recall",
                    "Macro F1"
                ],
                "Value": [
                    accuracy_score(
                        y_true,
                        y_pred
                    ),
                    precision_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0
                    ),
                    recall_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0
                    ),
                    f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0
                    )
                ]
            })

            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True
            )

            metrics_csv = metrics_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "⬇️ Download QA Metrics CSV",
                metrics_csv,
                "qa_metrics_report.csv",
                "text/csv"
            )

        else:

            st.info(
                "No labelled evaluation records available."
            )

    else:

        st.info(
            "Run model evaluation with expected labels to generate metrics."
        )


    # --------------------------------------------------------
    # COMPLETE EXCEL REPORT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📦 Complete QA Report"
    )

    st.write(
        "One Excel file containing the complete QA evidence."
    )

    acceptance_for_export = acceptance.copy()

    excel_data = create_excel_report(
        reviews=reviews,
        annotations=annotations,
        model_results=model_results,
        metrics_df=metrics_df,
        acceptance_df=acceptance_for_export
    )

    st.download_button(
        "📥 Download Complete ML QA Excel Report",
        excel_data,
        "Bengali_ML_QA_Complete_Report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )


    # --------------------------------------------------------
    # FINAL ACCEPTANCE
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🏁 Final QA Acceptance Decision"
    )

    final_decision = st.radio(
        "Release Recommendation",
        [
            "🟢 ACCEPT",
            "🟡 ACCEPT WITH CONDITIONS",
            "🔴 REJECT / SEND BACK TO ML TEAM"
        ]
    )

    acceptance_comment = st.text_area(
        "Final QA Comment",
        placeholder="Summarize the overall QA finding and release recommendation..."
    )

    if st.button(
        "🏁 Submit Final QA Decision",
        type="primary"
    ):

        conn = get_connection()

        conn.execute(
            """
            INSERT INTO acceptance_decisions
            (
                project_name,
                decision,
                comment,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                st.session_state.project or "Unassigned",
                final_decision,
                acceptance_comment,
                datetime.now().isoformat()
            )
        )

        conn.commit()
        conn.close()

        st.success(
            f"Final QA Decision saved: {final_decision}"
        )

        st.info(
            "The decision has been permanently stored in SQLite."
        )


    # --------------------------------------------------------
    # PREVIOUS ACCEPTANCE DECISIONS
    # --------------------------------------------------------

    if len(acceptance) > 0:

        st.divider()

        st.subheader(
            "📜 Acceptance Decision History"
        )

        st.dataframe(
            acceptance,
            use_container_width=True,
            hide_index=True
        )