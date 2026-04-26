import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import csv
import os

# ==============================
# SAFE PATH CONFIG (IMPORTANT)
# ==============================
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "model_for_com.csv")

MODEL_1 = os.path.join(DATA_DIR, "xgb_class1_1.pkl")
MODEL_2 = os.path.join(DATA_DIR, "xgb_class2_1.pkl")
MODEL_3 = os.path.join(DATA_DIR, "xgb_class3_1.pkl")
MODEL_4 = os.path.join(DATA_DIR, "xgb_class1_3.pkl")


# ==============================
# SAFE LOADERS (NO MORE CRASHES)
# ==============================

def safe_open_csv_header():
    if not os.path.exists(CSV_PATH):
        st.error(f"Missing file: {CSV_PATH}")
        st.stop()

    with open(CSV_PATH, "r") as f:
        return list(csv.reader(f))[0]


def safe_pickle(path):
    if not os.path.exists(path):
        st.error(f"Missing model file: {path}")
        st.stop()

    return pickle.load(open(path, "rb"))


# ==============================
# CORE LOGIC
# ==============================

def replacement(store_list):
    store = store_list.copy()
    store[0] = 0
    var = 0
    txt_flag = [False, 1]

    i = 0
    while i < len(store):
        if i > 19 and i < 26:
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

    for i in range(0, length - 1):
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

def class1_1(X):
    clf = safe_pickle(MODEL_1)
    return clf.predict_proba(X), clf.predict(X)


def class2_1(X):
    clf = safe_pickle(MODEL_2)
    return clf.predict_proba(X), clf.predict(X)


def VoteClass_1(treatment, Out, out_pred):
    X_test = treatment.iloc[:, 1:31] if len(treatment.columns) > 31 else treatment.iloc[:, 1:]

    _, res1 = class1_1(X_test)
    _, res2 = class2_1(X_test)

    dat2 = pd.concat([treatment.iloc[[0]], pd.DataFrame(res1)], axis=1)
    dat2 = pd.concat([dat2, pd.DataFrame(res2)], axis=1)

    X_final = dat2.iloc[:, 1:]

    clf = safe_pickle(MODEL_3)

    y_pred = clf.predict(X_final)
    pred_prob = clf.predict_proba(X_final)

    res_text = (
        f"atrauman({round(pred_prob[0][0], 3)}%)"
        if y_pred[0] == 0
        else f"Complex treatment({round(pred_prob[0][1], 3)}%)"
    )

    outcome = "Healing" if Out == 0 else "Non healing"
    return f"{res_text} | Outcome: {outcome} ({out_pred}%)"


# ==============================
# STREAMLIT APP
# ==============================

st.set_page_config(page_title="Wound Healing AI", layout="wide")

# session state
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
# MAIN APP
# ==============================

else:
    if st.session_state.post == -2:
        st.title("Data Input Mode")

        if st.button("Manual Input"):
            names = safe_open_csv_header()
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

            names = safe_open_csv_header()

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

        ref = np.array(list(csv.reader(open(CSV_PATH))))[1:, 1:].tolist()
        penalty, failed = verify_p1(ref, st.session_state.dat1)

        if failed:
            st.warning(f"Warning: deviation detected ({round(penalty, 2)}%)")

        if st.button("Run XGBoost Analysis"):
            clf = safe_pickle(MODEL_4)

            X = st.session_state.dat1.iloc[:, 1:].fillna(0)

            y_pred = clf.predict(X.iloc[:, :21])
            prob = clf.predict_proba(X.iloc[:, :21])

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
