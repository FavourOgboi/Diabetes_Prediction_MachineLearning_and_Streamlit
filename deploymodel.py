!pip install matplotlib
import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# use the current working directory to define the path of the images folder
image_folder = "images"

# use the current working directory to define the path of the diabetes.csv file
datafile = "diabetes.csv"

# use the current working directory to define the path of the trained_model.sav file
trained_model = "trained_model.sav"

@st.cache
def loadimage(imagefile):
    img = Image.open(imagefile)
    return img

def main():
    st.title("Diabetes Prediction System")

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
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
        Age = st.text_input("Age Value")

        ans = ""

        # Creating a button for prediction
        if st.button("Diabetes Test Result"):
            input_data = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]

            # changin the input data to a numpy array

            input_data_as_numpy_array = np.asarray(input_data)

            # reshape the array as we are predicting for only one instance

            input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

            prediction = loaded_model.predict(input_data_reshape)

            print('The prediction is :', prediction)

            if prediction[0] == 0:
                ans = ("The Person Is Not Diabetic")
            else:
                ans = ("The Person Is Diabetic")

        st.success(ans)

    else:
        st.subheader("About")
        # use the current working directory to define the path of the pexels-artem-podrez-6823763.jpg file
        imagefile = f"{image_folder}/pexels-artem-podrez-6823763.jpg"
        st.image(loadimage(imagefile),width = 285)

        txtfile = ("Diabetes mellitus is a condition defined by persistently high levels of sugar (glucose) in the blood There are several types of diabetes.\n The two most common are called type 1 diabetes and type 2 diabetes.\n During digestion, food is broken down into its basic components. Carbohydrates are broken down into simple sugars, primarily glucose.\n Glucose is a critically important source of energy for the body cells.To provide energy to the cells, glucose needs to leave the bloodstream and get inside the cells.\n An organ in the abdomen called the pancreas produces a hormone called insulin, which is essential to helping glucose get into the body's cells. In a person without diabetes,the pancreas produces more insulin whenever blood levels of glucose rise (for example, after a meal),and the insulin signals the body's cells to take in the glucose. In diabetes,either the pancreas's ability to produce insulin or the cells' response to insulin is altered.The purpose of this app is to deploy a model that can predict if a person is diabetic or not based of the information given by th user.")
        st.write(txtfile)

if __name__ == "__main__":
    main()


