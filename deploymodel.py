import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image


# use the current working directory to define the path of the diabetes.csv file
datafile = "diabetes.csv"

# use the current working directory to define the path of the trained_model.sav file
trained_model = "trained_model.sav"

# use the current working directory to define the path of the diabetes.csv file
datafile = "diabetes.csv"

def main():
    st.title("Diabetes Prediction System")
    st.write("Welcome to our diabetes prediction app. We have created this app to help predict whether an individual has diabetes based on various factors such as number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.")
    st.write("Our app utilizes a machine learning model that was trained on a diabetes dataset to make predictions. The dataset used in this app is from Kaggle and contains information about patients with and without diabetes.")

    menu = ["About","Dataset","PredictionApp"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Dataset":

        st.write("The dataset is shown below.")

        st.subheader("Dataset")
        df = pd.read_csv(datafile)
        st.dataframe(df)
        
        # Use Seaborn to create a bar plot of the number of patients with diabetes (outcome = 1) and without diabetes (outcome = 0)
        sns.countplot(x='Outcome', data=datafile)
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        plt.title('Diabetes Outcome Distribution')

        # Show the plot
        plt.show()
        
        st.write("\n This code creates a bar plot that shows the distribution of patients with diabetes (outcome = 1) and without diabetes (outcome = 0) in the dataset")
        st.write("\n")
        fig = px.bar(datafile, x="Outcome", y="Age", color='Outcome',title="Age Distribution by Outcome")
        fig.show()
        st.write("This will create a interactive bar plot showing the distribution of patients with diabetes (outcome = 1) and without diabetes (outcome = 0) by Age.")
                 
        st.write("\n")
                 
        st.write("Correlation As you can tell from the analysis the values in some columns are more closely related to the outcome compared to other columns. \n" + "This relationship is often expressed numerically using a measure called the _correlation coefficient_, which can be computed using the .corr method of a Pandas series.\n")
                 
        # use the current working directory to define the path of the corr.PNG file
        imagefile = f"{image_folder}/corr.PNG"
        st.image(loadimage(imagefile),width = 285)
                 
        st.write("The dataset contains various information such as patient's number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, age, and whether or not the patient has diabetes (outcome).")
        st.write(" To generalize it with the data of the world today, one can say that diabetes is a growing global health concern. \n According to the World Health Organization, an estimated 422 million adults were living with diabetes in 2014, and this number is projected to increase to 629 million by 2045.")
        st.write("\n")
        st.write("Diabetes is a leading cause of death and disability, and is a major contributor to cardiovascular disease, kidney failure, blindness, and amputations. \n Therefore, understanding and analyzing data such as the diabetes dataset can help in the development of effective strategies for the prevention, early detection, and management of diabetes, and ultimately improving the health outcomes of individuals with diabetes.")
    
    elif choice == "PredictionApp":
        st.subheader("PredictionApp")

        # loading the saved model
        loaded_model = pickle.load(open(trained_model,"rb"))

        Pregnancies = st.text_input("Number Of Pregnancies")
        Glucose = st.text_input("glucose Level")
        BloodPressure = st.text_input("Blood pressure value")
        SkinThickness = st.text_input("Skin Thickness Value")
        Insulin = st.text_input("Insulin Level")
        BMI = st.text_input("BMI Level")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
        Age = st.text_input("Age")

        if st.button("Predict"):
            prediction = loaded_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
            if prediction == 0:
                st.success("The patient does not have diabetes.")
            else:
                st.warning("The patient has diabetes.")

    elif choice == "About":
        st.subheader("About")
        imagefile = f"{image_folder}/pexels-artem-podrez-6823763.jpg"
        st.image(loadimage(imagefile),width = 300)
        
        st.write("This app is designed to help individuals, doctors, and healthcare providers better understand and manage diabetes. The app takes into account various factors such as number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age to make predictions. The app's machine learning model was trained on a diabetes dataset that was sourced from Kaggle.")
        st.write("\n")
        st.write("Our app is easy to use and provides clear and concise results. The user interface is user-friendly and the results are easy to understand. The app's predictions are based on data and research, so users can trust the results.")
        st.write("\n")
        st.write("We hope that this app will help individuals and healthcare providers better understand and manage diabetes, ultimately leading to improved health outcomes for individuals with diabetes.")
        st.write("\n")
        st.write("We are constantly working to improve the app and welcome any feedback or suggestions you may have. Feel free to contact us with any questions or concerns.")

if __name__ == "__main__":
    main()
