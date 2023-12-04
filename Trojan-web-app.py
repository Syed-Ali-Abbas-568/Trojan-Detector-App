# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 21:51:16 2023

@author: syedaliabbas
"""
import matplotlib
matplotlib.use('Agg')  # Use Agg backend

import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import base64
import seaborn as sns
from sklearn.metrics import roc_curve, auc

model_path = 'trained_gradient_model.sav'
csv_path = 'input.csv'

loaded_model = pickle.load(open(model_path, 'rb'))




def trojan_prediction(input_data, key_data=None):
    input=pd.read_csv(input_data)
    predictions = loaded_model.predict(input)
    trojan_count = 0
    total_flows = len(predictions)
    diagnosis=[]  
    
    # Load key data if provided
    if key_data:
        key_df = pd.read_csv(key_data)
        
    
    
    for i, prediction in enumerate(predictions):
        flow_id = input.loc[i, 'Flow ID']
        if prediction == 1:
           # st.markdown(f"The Flow ID {flow_id} has been identified as :red[trojan]")
            trojan_count += 1
            diagnosis.append("trojan")
        else:
            diagnosis.append("benign")
           # st.markdown(f"The Flow ID {flow_id} has been identified as :green[benign]")
    # Display total number of trojan flows and total flows analyzed
    st.write(f"Total Trojan Flows: {trojan_count}")
    st.write(f"Total Flows Analyzed: {total_flows}")
    trojan_percentage = (trojan_count / total_flows) * 100
    st.write(f"Percentage of Flows Identified as Trojan: {trojan_percentage:.2f}%")

    # Plot a bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(['Benign', 'Trojan'], [total_flows - trojan_count, trojan_count], color=['green', 'red'])
    plt.title('Distribution of Flows')
    plt.xlabel('Diagnosis')
    plt.ylabel('Number of Flows')
    st.pyplot(plt)
    
    st.write("Complete Report:")
    report_df = pd.DataFrame({'Flow ID': input['Flow ID'], 'Diagnosis': diagnosis})
    
    st.dataframe(report_df, height=400)  # Adjust the height as needed
    # Encode DataFrame to base64
    csv_data = report_df.to_csv(index=False).encode('utf-8')
    base64_encoded = base64.b64encode(csv_data).decode('utf-8')
    # Display download link for the complete report as a CSV file
    st.markdown(f"## [Download Complete Report as CSV](data:text/csv;base64,{base64_encoded})")  
    
    # Plot a pie chart
    plt.figure(figsize=(6, 6))
    labels = ['Benign', 'Trojan']
    sizes = [total_flows - trojan_count, trojan_count]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
    plt.title('Percentage of Flows')
    st.pyplot(plt)

    if key_data: 
        # Plot a confusion matrix
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        confusion_matrix = pd.crosstab(key_df['Class'], predictions, rownames=['Class'], colnames=['Predicted'])
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title('Confusion Matrix')
        st.pyplot(plt)
        # Display ROC curve
      
        
        
   
    



#main function
def main():
    
    st.markdown("<h1 style='text-align: center; color: black;'>Trojan Detection Web App</h1>", unsafe_allow_html=True)
    
    #Creating Title for our user page
    with st.columns(3)[1]:
        
        st.image(r"C:\Users\syeda\Documents\7th Semester\Final Code\1296059.svg", width=150)
    
    
    

    
    
    input_data= st.file_uploader("Upload a CSV") 
    
    use_key_data = st.checkbox("Use Key Data")

    # If checkbox is ticked, provide a file uploader for key data
    if use_key_data:
        key_data = st.file_uploader("Upload Key Data (CSV)", type=["csv"])
        
    else:
        key_data = None
    
    st.text("Model Used: Gradient Boosting Classifier")

    # Display definition of Gradient Boosting Classifier in bold
    st.markdown(
    """
    **Gradient Boosting Classifier:**
    Gradient Boosting is an ensemble machine learning technique for classification and regression tasks. 
    A Gradient Boosting Classifier builds multiple decision trees sequentially, where each tree corrects 
    the errors of the previous one. It combines the predictions of weak learners (individual trees) to 
    form a strong predictor. The key idea is to fit a new tree to the residuals (the differences between 
    the observed and predicted values) of the existing model. In scikit-learn, the `GradientBoostingClassifier` 
    is an implementation of this algorithm for classification tasks. It is known for its high predictive 
    power and ability to handle complex relationships in data.
    """
    )
    
    #Creating a button 
    if st.button("Detect Trojan"):
        trojan_prediction(input_data,key_data)   
        st.success("All packets analysed")
    
    st.subheader(f"What is a trojan horse?")
    st.write("Similar to the Trojan Horse known from ancient Greco-Roman tales, this type of malicious software uses disguise or misdirection to hide its true function.")
    st.markdown("<p><b>Made By:</b> <br>Syed Ali Abbas<br>Shabano Waqar<br>Huzaifa Bin Munir</p>", unsafe_allow_html=True)
    
   

if __name__ =='__main__':
    main()
    
    