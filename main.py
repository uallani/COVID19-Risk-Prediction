#Importing the required libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import io

#Setting the matplotlib style
plt.style.use('fivethirtyeight')


#Function to collect Input
def get_input():
    pneumonia = st.sidebar.text_input("pneumonia(enter 1 or 2)", "")
    diabetes = st.sidebar.text_input("diabetes(enter 1 or 2)", "")
    copd = st.sidebar.text_input("copd(enter 1 or 2)", "")
    asthma = st.sidebar.text_input("asthma(enter 1 or 2)", "")
    hypertension = st.sidebar.text_input("hypertension(enter 1 or 2)", "")
    obesity = st.sidebar.text_input("obesity(enter 1 or 2)", "")
    contact_other_covid = st.sidebar.text_input("contact_other_covid(enter 1 or 2 or 99)", "")
    covid_res = st.sidebar.text_input("covid_res(enter 1 or 2 or 3)", "")
    return pneumonia, diabetes, copd, asthma, hypertension, obesity, contact_other_covid, covid_res


#Function to convert user input to numeric values for further processing
def get_data(pneumonia, diabetes, copd, asthma, hypertension, obesity, contact_other_covid, covid_res):
    pneumonia = pd.to_numeric(pneumonia)
    diabetes = pd.to_numeric(diabetes)
    copd = pd.to_numeric(copd)
    asthma = pd.to_numeric(asthma)
    hypertension = pd.to_numeric(hypertension)
    obesity = pd.to_numeric(obesity)
    contact_other_covid = pd.to_numeric(contact_other_covid)
    covid_res = pd.to_numeric(covid_res)
    return pneumonia, diabetes, copd, asthma, hypertension, obesity, contact_other_covid, covid_res


def main():
    #Streamlit web application
    #Title of the web application
    st.write("""
    # COVID-19 case prediction web application
    An Intelligent System for Prediction of COVID-19 Case using Machine Learning Framework-
    Logistic Regression """)

#Displays an image
    image = Image.open(r"C:\Users\uallani\PycharmProjects\Covid-19 Project\Covid-19.png")
    st.image(image, use_column_width=True)
#Streamlit Sidebar for user input
    st.sidebar.header('User Input')
    st.sidebar.subheader('(for prediction of covid-19 case for this particular input)')

    pneumonia, diabetes, copd, asthma, hypertension, obesity, contact_other_covid, covid_res = get_input()
    pneumonia, diabetes, copd, asthma, hypertension, obesity, contact_other_covid, covid_res = get_data(pneumonia,
                                                                                                        diabetes, copd,
                                                                                                        asthma,
                                                                                                        hypertension,
                                                                                                        obesity,
                                                                                                        contact_other_covid,
                                                                                                        covid_res)
#Inserts the user data into a dataset
    user_data = pd.DataFrame(
        data=[[pneumonia, diabetes, copd, asthma, hypertension, obesity, contact_other_covid, covid_res]],
        columns=['pneumonia', 'diabetes', 'copd', 'asthma', 'hypertension', 'obesity', 'contact_other_covid',
                 'covid_res'])
#Button
    if st.button('Insert User Data'):
        if 'user_data' not in st.session_state:
            st.session_state.user_data = []
        st.session_state.user_data.append(user_data)
        st.success('User data inserted successfully!')
#Button
    if st.button('Generate Confusion Matrix', key='confusion_matrix_btn'):
        if 'user_data' in st.session_state:
            user_data = pd.concat(st.session_state.user_data, ignore_index=True)
            if not user_data.empty:
                st.write(user_data)

                # Drop rows with missing values
                user_data.dropna(inplace=True)

                # Load COVID dataset
                covid_data = pd.read_csv(r"C:\Users\uallani\PycharmProjects\Covid-19 Project\covid data set.csv")

                #Displaying target variable
                st.header("Sample features used in our model\n")
                fig = plt.figure(figsize=(10, 4))
                sns.countplot(x='covid_res', data=covid_data, hue='copd', palette='rainbow')
                st.pyplot(fig)

                #first five records of data
                st.header('Some records of data')
                st.write(covid_data.head())

                # Print number of rows and columns
                st.write(f"Number of rows: {covid_data.shape[0]}")
                st.write(f"Number of columns: {covid_data.shape[1]}")

                # Append user data to COVID dataset
                combined_data = pd.concat([covid_data, user_data], ignore_index=True)

                # Save combined dataset to a file
                combined_data.to_csv('combined_data.csv', index=False)  # Change the filename/path as needed

                # Further processing using the combined dataset

                st.header('Some records of combined data')
                st.write(combined_data.tail())

                # Print number of rows and columns
                st.write(f"Number of rows: {combined_data.shape[0]}")
                st.write(f"Number of columns: {combined_data.shape[1]}")

                # Generate confusion matrix using the saved dataset
                combined_data = pd.read_csv('combined_data.csv')  # Read the saved dataset
                new_features = combined_data[
                    ['pneumonia', 'diabetes', 'copd', 'asthma', 'hypertension', 'obesity', 'contact_other_covid']]
                x = new_features
                y = combined_data['covid_res']

                # Check if there are enough samples for a train-test split
                if len(combined_data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=5)
                else:
                    st.warning("Insufficient data for train-test split. Consider adding more data.")
                    return

                logreg = LogisticRegression(solver='lbfgs', max_iter=3000)
                logreg.fit(x_train, y_train)
                y_pred = logreg.predict(x_test)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)


                st.header('Confusion Matrix')
                # Generate confusion matrix
                cm = confusion_matrix(y_test, y_pred)

                # Plot confusion matrix
                plt.figure(figsize=(8, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')

                # Save confusion matrix plot as an image file
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()

                # Display confusion matrix image
                buffer.seek(0)
                st.image(buffer, use_column_width=True)
            else:
                st.warning('No user data available. Please enter user data first.')
        else:
            st.warning('No user data available. Please enter user data first.')


if __name__ == "__main__":
    main()
