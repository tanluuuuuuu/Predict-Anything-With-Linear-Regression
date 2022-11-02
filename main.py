import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils import transform_column_to_onehot, split_data_and_train_model, transform_data_to_predict, predict_and_evaluate_model, train_with_kFold

def resetSessionState():
    st.session_state['trained'] = False
    st.session_state['loss'] = [0, 0]
    return

if __name__=='__main__':
    if 'trained' not in st.session_state:
        st.session_state['trained'] = False
        st.session_state['loss'] = [0, 0]
        st.session_state['model'] = None
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(df)
            column_labels = df.columns.to_numpy()
            selected_columns = []
            st.header("Training feature ")
            
            for column_label in column_labels[:-1]:
                t = st.checkbox(column_label, on_change=resetSessionState)
                if t:
                    selected_columns.append(column_label)
                    
            st.write(f"Output feature: {column_labels[-1]}")
        
            if (len(selected_columns) == 0):
                st.info("Select training feature to continue")
                st.stop()
        
            y = df[column_labels[-1]].to_numpy().reshape(-1, 1)
            X = np.array([])
            dict_encoder = {}

            X, dict_encoder= transform_column_to_onehot(df=df, selected_columns=selected_columns, dict_encoder=dict_encoder)

            st.header("Train test split: ")
            kFold_mode = st.checkbox("K Fold", on_change=resetSessionState)
                    
            if kFold_mode:
                n_splits = int(st.number_input("Enter n split: ", min_value=2, step=1))
                if st.button("Train model!"):
                    st.session_state['trained'] = True
                    model, train_loss, test_loss = train_with_kFold(X, y, n_splits)
                    st.session_state['loss'] = [train_loss, test_loss]
                    st.session_state['model'] = model
            else:
                train_split_ratio = float(st.slider('Select train ratio in range', 0, 90, 80, on_change=resetSessionState))
                if st.button("Train model!"):
                    st.session_state['trained'] = True
                    model, X_train, X_test, y_train, y_test = split_data_and_train_model(X=X, y=y, train_split_ratio=train_split_ratio)
                    st.session_state['model'] = model
                    test_loss = predict_and_evaluate_model(model, X_test, y_test)
                    train_loss = predict_and_evaluate_model(model, X_train, y_train)
                    st.session_state['loss'] = [train_loss, test_loss]

            fig, ax = plt.subplots()
            if kFold_mode:
                if not st.session_state['trained']:
                    st.session_state['loss'] = [[0], [0]]
                r = np.arange(len(st.session_state['loss'][0]))
                ax.bar(r - 0.2, st.session_state['loss'][0], color ='b', width = 0.4, label='Train')
                ax.bar(r + 0.2, st.session_state['loss'][1], color ='g', width = 0.4, label='Test')
                ax.set_xticks(r, r)
                ax.set_xlabel("Fold")
                ax.set_ylabel("MSE")
                ax.set_title("Train test loss")
                ax.legend()
                st.pyplot(fig)
            else:
                ax.bar(["Train", "Test"], st.session_state['loss'], color ='maroon', width = 0.4)
                ax.set_xlabel("MSE")
                ax.set_ylabel("Set")
                ax.set_title("Train test loss")
                st.pyplot(fig)                
            
            if not st.session_state['trained']:
                st.info("Train model to continue")
                st.stop()
                
            st.header("Make Prediction")
            dict_data_to_prediction = {}
            for (index, data_column) in enumerate(selected_columns):
                if (df.dtypes[data_column] == 'object'):
                    unique_val = df[data_column].unique()
                    cat_val = st.selectbox(data_column, (unique_val))
                    cat_val_onehot = dict_encoder[data_column].transform([[cat_val]]).toarray().squeeze()
                    dict_data_to_prediction[data_column] = cat_val_onehot
                else:
                    dict_data_to_prediction[data_column] = st.number_input(data_column)
            
            if st.button("Predict!"):
                print("here")
                data = transform_data_to_predict(selected_columns, dict_data_to_prediction)
                try:
                    model = st.session_state['model'] 
                    prediction = model.predict(data)
                    st.write(f"{column_labels[-1]}: {prediction[0][0]}")
                except Exception as e:
                    st.write(e)
            
        except Exception as e:
            st.write(e)
