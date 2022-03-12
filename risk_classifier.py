import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load the trained model
enc = pickle.load(open("enc_ac.pickle", "rb"))
LR = pickle.load(open("LR_ac.pickle", "rb"))

def main():

    # Title of the app page
    st.title('AML Risk Level Classifier')

    # Add a heading for input features
    st.subheader('Enter Feature For Classification')

    # Rquest for input fatures, but replod with some default values
    Acc_type = st.selectbox(
        'What is the type of account?',
        ("Individual", "Company"))
    Country = st.selectbox(
        'What is the place of birth or place of incorporation?',
        ("HK", "China", "British Virgin Island", "Caymond Island", "Taiwan", "Macau", "UK", "USA"))
    Occupation = st.selectbox(
        'What is the Occupation? (For individuals only)?',
        ("Housewife", "Students", "Bank Staff", "Accountant", "Waitress", "Civil Servant", "Entertainer", "Teacher", "Engineer",
               "Programmer", "Data Analyst", "Data Engineer", "Data Scientist", "Financial Company staff", "Others", "Not Disclosed",
               "Unemployed", "Retired", "Director", "Manager", "Singer"))
    Business_nature = st.selectbox(
        'What is the Business nature? (For Companies only)?',
        ("Trading of Electronic Equipment", "Trading of Household goods", "Accounting and Audit", "Law service providers",
               "Trading of plastic goods", "FInancial service", "Security Company", "Engineering and Construction", "Restaurant",
                "Education", "IT Service", "Shareholding Company", "Investment", "Not Disclosed", "Government and public Boby", "Garment",
               "Consulting", "Skin care", "Health goods", "Fitness products", "Event management", "Operating e-commerce","Natural gas and Petroleum", "Entertainment Company"))
    loan_investment = st.selectbox(
        'Do you maintain any investment or loan accounts with our bank?',
        ("Yes", "No"))

    if Acc_type == "Individual":
        Acc_type = "I"
    elif Acc_type == "Company":
        Acc_type = "C"
    else:
        return "Please enter the account type"

    if Acc_type == "I":
        Business_nature = Occupation

    # Get predictions when the button is pressed
    if st.button('Get Prediction'):

        # run predictions
        df_ohe = enc.transform([[Country,Acc_type, Business_nature, loan_investment]]).toarray()
        df_enc = pd.DataFrame(df_ohe)
        features = np.hstack(
            [df_enc])
        pred = LR.predict(features)

        if pred == False:
            st.write("This is classified as low risk account.")
        elif pred == True:
            st.write("This is classified as high risk account.")


if __name__ == "__main__":
    main()