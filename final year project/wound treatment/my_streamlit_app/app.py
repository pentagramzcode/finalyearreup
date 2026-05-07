import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import csv
import os

# ==============================
# BASE PATH FIX (IMPORTANT)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def path(file_name):
    return os.path.join(DATA_DIR, file_name)


# ==============================
# SAFE LOADERS
# ==============================

def load_csv_header():
    file_path = path("model_for_com.csv")

    if not os.path.exists(file_path):
        st.error(f"Missing file: {file_path}")
        st.stop()

    with open(file_path, "r") as f:
        return list(csv.reader(f))[0]


def load_pickle(file_name):
    file_path = path(file_name)

    if not os.path.exists(file_path):
        st.error(f"Missing model: {file_path}")
        st.stop()

    return pickle.load(open(file_path, "rb"))


# ==============================
# CORE LOGIC (UNCHANGED)
# ==============================

def replacement(store_list):
    store = store_list.copy()
    store[0] = 0
    var = 0
    txt_flag = [False, 1]

    i = 0
    while i < len(store):
        if 19 < i < 26:
            if not txt_flag[0]:
                store[i + var] = "@:V" + str(txt_flag[1]) + " fibrinous"

                store.insert(i + var, "#-V" + str(txt_flag[1]) + " Slough")
                store.insert(i + var, "#-V" + str(txt_flag[1]) + " Hypergranulation")
                store.insert(i + var, "#+V" + str(txt_flag[1]) + " re-epithelialisation")
                store.insert(i + var, "#+V" + str(txt_flag[1]) + " Granulation")

                var += 4
                txt_flag[0] = True
                txt_flag[1] += 1
            else:
                txt_flag[0] = False
                if (i + var + 1) < len(store):
                    store.pop(i + var + 1)

        elif i > 0:
            if (i + var) < len(store):
                store[i + var] = ",," + str(store[i + var])

        i += 1

    return store


def verify_p1(data_array, dat1):
    length = 33
    d_list = dat1.to_numpy().tolist()
    outside = 0

    for i in range(length - 1):
        try:
            val = d_list[0][i + 1]
            var = round(((val - float(data_array[0][i])) / val) * 100, 3)

            if var > 150:
                outside += float((1 - float(data_array[2 if var > 250 else 4][i])))
            elif var < -150:
                outside += float((1 - float(data_array[5 if var < -250 else 3][i])))

        except:
            continue

    return outside, (outside > 0.13)


# ==============================
# MODELS
# ==============================

def align_features(X, clf):
    """Align a DataFrame's columns to match what a loaded XGBoost model expects."""
    expected = clf.get_booster().feature_names
    if expected is None:
        return X.values  # model was trained on a numpy array — strip names
    missing = set(expected) - set(X.columns)
    for col in missing:
        X[col] = 0
    return X[expected]  # select & reorder to match training exactly


def class1_1(X):
    clf = load_pickle("xgb_class1_1.pkl")
    X_aligned = align_features(X.copy(), clf)
    return clf.predict_proba(X_aligned), clf.predict(X_aligned)


def class2_1(X):
    clf = load_pickle("xgb_class2_1.pkl")
    X_aligned = align_features(X.copy(), clf)
    return clf.predict_proba(X_aligned), clf.predict(X_aligned)


def VoteClass_1(treatment, Out, out_pred):
    X_test = treatment.iloc[:, 1:31] if len(treatment.columns) > 31 else treatment.iloc[:, 1:]

    _, res1 = class1_1(X_test)
    _, res2 = class2_1(X_test)

    dat2 = pd.concat([treatment.iloc[[0]], pd.DataFrame(res1)], axis=1)
    dat2 = pd.concat([dat2, pd.DataFrame(res2)], axis=1)

    X_final = dat2.iloc[:, 1:]

    clf = load_pickle("xgb_class3_1.pkl")
    X_final_aligned = align_features(X_final.copy(), clf)

    y_pred = clf.predict(X_final_aligned)
    pred_prob = clf.predict_proba(X_final_aligned)

    res_text = (
        f"atrauman({round(pred_prob[0][0], 3)}%)"
        if y_pred[0] == 0
        else f"Complex treatment({round(pred_prob[0][1], 3)}%)"
    )

    outcome = "Healing" if Out == 0 else "Non healing"
    return f"{res_text} | Outcome: {outcome} ({out_pred}%)"


# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Wound Healing AI", layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False
if "post" not in st.session_state:
    st.session_state.post = -2
if "dat1" not in st.session_state:
    st.session_state.dat1 = pd.DataFrame()
if "store" not in st.session_state:
    st.session_state.store = []


# ==============================
# LOGIN
# ==============================

if not st.session_state.auth:
    st.title("Admin Login")

    with st.form("login"):
        user = st.text_input("Username")
        pas = st.text_input("Password", type="password")

        if st.form_submit_button("Login"):
            if user == "admin" and pas == "password":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Invalid credentials")


# ==============================
# APP FLOW
# ==============================

else:
    if st.session_state.post == -2:
        st.title("Data Input Mode")

        if st.button("Manual Input"):
            names = load_csv_header()
            st.session_state.store = replacement(names)
            st.session_state.post = 1
            st.rerun()

    elif st.session_state.post < len(st.session_state.store):
        idx = st.session_state.post
        prompt = st.session_state.store[idx]

        if prompt.startswith(",,") or prompt.startswith("@:"):
            label = prompt[2:]
        elif prompt.startswith("#-") or prompt.startswith("#+"):
            label = prompt[3:]
        else:
            label = prompt

        st.progress(idx / max(len(st.session_state.store) - 1, 1))
        st.subheader(f"Step {idx}: {label}")

        val = st.text_input("Enter Value", key=f"step_{idx}")

        if st.button("Continue"):
            num = float(val) if val else float("nan")

            names = load_csv_header()

            try:
                offset = int(st.session_state.store[0])
                col_name = names[idx - offset]
            except:
                col_name = label

            new_col = pd.DataFrame({col_name: [num]})
            st.session_state.dat1 = pd.concat([st.session_state.dat1, new_col], axis=1)

            st.session_state.post += 1
            st.rerun()

    else:
        st.title("Predictive Outcome")

        ref = np.array(list(csv.reader(open(path("model_for_com.csv")))))[1:, 1:].tolist()
        penalty, failed = verify_p1(ref, st.session_state.dat1)

        if failed:
            st.warning(f"Deviation detected: {round(penalty, 2)}%")

        if st.button("Run XGBoost Analysis"):
            clf = load_pickle("xgb_class1_3.pkl")

            X = st.session_state.dat1.iloc[:, 1:].fillna(0)

            # Align columns to exactly what the model was trained on
            expected_features = clf.get_booster().feature_names
            if expected_features is not None:
                missing = set(expected_features) - set(X.columns)
                extra   = set(X.columns) - set(expected_features)
                if missing:
                    st.warning(f"Columns missing from your data (defaulting to 0): {missing}")
                    for col in missing:
                        X[col] = 0
                if extra:
                    st.info(f"Extra columns in your data (ignored): {extra}")
                X_input = X[expected_features]   # reorder to match training exactly
            else:
                X_input = X.iloc[:, :21].values  # model trained on array, strip names

            y_pred = clf.predict(X_input)
            prob   = clf.predict_proba(X_input)

            report = VoteClass_1(
                st.session_state.dat1,
                y_pred[0],
                round(prob[0][y_pred[0]], 3)
            )

            st.success(report)

        if st.button("Reset"):
            st.session_state.post = -2
            st.session_state.dat1 = pd.DataFrame()
            st.rerun()
