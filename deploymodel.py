import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import numpy as np

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

        st.write("Correlation As you can tell from the analysis the values in some columns are more closely related to the outcome compared to other columns." +
        
        "This relationship is often expressed numerically using a measure called the _correlation coefficient_, which can be computed using the .corr method of a Pandas series.")

        # use the current working directory to define the path of the corr.PNG file
        imagefile = f"{image_folder}/corr.PNG"
        st.image(loadimage(imagefile),width = 400)

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
        st.image(loadimage(imagefile),width = 500)

        txtfile = "about.txt"
        with open(txtfile, "r") as f:
            about_text = f.read()
        st.write(about_text)

if __name__ == "__main__":
    main()


