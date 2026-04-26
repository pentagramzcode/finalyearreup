import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import csv
import math
import os

# ==============================
# 1. CORE MATH & DATA LOGIC
# ==============================

def replacement(store_list):
    store = store_list.copy()
    store[0] = 0
    var = 0
    txt_flag = [False, 1]
    
    # We use a while loop or a careful index tracker because the list size changes
    i = 0
    while i < len(store):
        if i > 19 and i < 26:
            if not txt_flag[0]:
                # We are at the start of a wound tissue pair
                # Replace the current item
                store[i + var] = "@:V" + str(txt_flag[1]) + " fibrinous"
                
                # Insert the 4 supporting features BEFORE the current one
                store.insert(i + var, "#-V" + str(txt_flag[1]) + " Slough")
                store.insert(i + var, "#-V" + str(txt_flag[1]) + " Hypergranulation")
                store.insert(i + var, "#+V" + str(txt_flag[1]) + " re-epithelialisation")
                store.insert(i + var, "#+V" + str(txt_flag[1]) + " Granulation")
                
                var += 4
                txt_flag[0] = True
                txt_flag[1] += 1
            else:
                # This handles the second part of the original pair
                txt_flag[0] = False
                # Remove the redundant "V-" value from the original dataset
                if (i + var + 1) < len(store):
                    store.pop(i + var + 1)
        elif i > 0:
            # Add your checksum to all other standard features
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
        except: continue
    return outside, (outside > 0.13)

# ==============================
# 2. XGBOOST ENSEMBLE SYSTEM
# ==============================

def class1_1(X_test):
    clf = pickle.load(open("data/xgb_class1_1.pkl", "rb"))
    return clf.predict_proba(X_test, iteration_range=(0, 100)), clf.predict(X_test, iteration_range=(0, 100))

def class2_1(X_test):
    clf = pickle.load(open("data/xgb_class2_1.pkl", "rb"))
    return clf.predict_proba(X_test), clf.predict(X_test)

def VoteClass_1(treatment, Out, out_pred):
    # Ensure we only use columns the model expects
    X_test = treatment.iloc[:, 1:31] if len(treatment.columns) > 31 else treatment.iloc[:, 1:]
    
    prob1, res1 = class1_1(X_test)
    prob2, res2 = class2_1(X_test)
    
    n_test = pd.DataFrame(res1)
    dat2 = pd.concat([treatment.iloc[[0]], n_test], axis=1)
    dat2.columns = [*dat2.columns[:-1], '1 result']
    
    n_test2 = pd.DataFrame(res2)
    dat2 = pd.concat([dat2, n_test2], axis=1)
    dat2.columns = [*dat2.columns[:-1], '2 result']
    
    X_final = dat2.iloc[:, 1:]
    clf = pickle.load(open("data/xgb_class3_1.pkl", "rb"))
    
    y_pred = clf.predict(X_final, iteration_range=(0, 12))
    pred_prob = clf.predict_proba(X_final, iteration_range=(0, 12))
    
    res_text = f"atrauman({round(pred_prob[0][0], 3)}%)" if y_pred[0] == 0 else f"Complex treatment({round(pred_prob[0][1], 3)}%)"
    outcome = "Healing" if Out == 0 else "Non healing"
    return f"The result is: {res_text} with the outcome of {outcome}({out_pred}%)"

# ==============================
# 3. WEB INTERFACE (STREAMLIT)
# ==============================

st.set_page_config(page_title="Wound Healing AI", layout="wide")

if 'auth' not in st.session_state: st.session_state.auth = False
if 'post' not in st.session_state: st.session_state.post = -2
if 'dat1' not in st.session_state: st.session_state.dat1 = pd.DataFrame({'Blank': [float('nan')]})
if 'store' not in st.session_state: st.session_state.store = []

if not st.session_state.auth:
    st.title("Admin Login")
    with st.form("login"):
        user = st.text_input("Username")
        pas = st.text_input("Password", type="password")
        if st.form_submit_button("Enter"):
            if user == "admin" and pas == "password":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Invalid credentials")
else:
    if st.session_state.post == -2:
        st.title("Data Input Mode")
        if st.button("Manual Input"):
            with open("data/model_for_com.csv", 'r') as f:
                names = list(csv.reader(f))[0]
            st.session_state.store = replacement(names)
            st.session_state.post = 1
            st.rerun()

    elif st.session_state.post < len(st.session_state.store):
        idx = st.session_state.post
        prompt = st.session_state.store[idx]
        
        # Determine display label
        if prompt.startswith(",,") or prompt.startswith("@:"): clean_label = prompt[2:]
        elif prompt.startswith("#-") or prompt.startswith("#+"): clean_label = prompt[3:]
        else: clean_label = prompt
        
        st.progress(idx / (len(st.session_state.store)-1))
        st.subheader(f"Step {idx}: {clean_label}")
        
        val = st.text_input("Enter Value", key=f"step_{idx}")
        
        if st.button("Continue"):
            num = float(val) if val else float('nan')
            
            with open("data/model_for_com.csv", 'r') as f:
                names = list(csv.reader(f))[0]
            
            # --- SAFE INDEXING LOGIC ---
            try:
                offset = int(st.session_state.store[0])
                col_name = names[idx - offset]
            except IndexError:
                col_name = clean_label # Fallback to prompt name if index fails
            
            new_col = pd.DataFrame({col_name: [num]})
            st.session_state.dat1 = pd.concat([st.session_state.dat1, new_col], axis=1)
            
            st.session_state.post += 1
            st.rerun()

    else:
        st.title("Predictive Outcome")
        with open("data/model for com.csv", 'r') as f:
            ref = np.array(list(csv.reader(f)))[1:, 1:].tolist()
        
        penalty, failed = verify_p1(ref, st.session_state.dat1)
        if failed:
            st.warning(f"Warning: Data deviates from mean by {round(penalty, 2)}%")
        
        if st.button("Run XGBoost Analysis"):
            clf_heal = pickle.load(open("data/xgb_class1_3.pkl", "rb"))
            # Match training features (usually the first 21-30 columns)
            X = st.session_state.dat1.iloc[:, 1:].fillna(0)
            
            y_pred = clf_heal.predict(X.iloc[:, :21], iteration_range=(0, 12)) 
            prob = clf_heal.predict_proba(X.iloc[:, :21], iteration_range=(0, 12))
            
            report = VoteClass_1(st.session_state.dat1, y_pred[0], round(prob[0][y_pred[0]], 3))
            st.success(report)

        if st.button("Reset"):
            st.session_state.post = -2
            st.session_state.dat1 = pd.DataFrame({'Blank': [float('nan')]})
            st.rerun()
