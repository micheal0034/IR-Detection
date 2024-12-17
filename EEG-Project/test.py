import streamlit as st
import pandas as pd
from joblib import load
from io import BytesIO
import matplotlib.pyplot as plt

# Load model (assumes model is already saved as 'random_forest_model.pkl')
model = load("random_forest_model.pkl")

# Define thresholds for valence and arousal
VALENCE_LOW = 3
AROUSAL_HIGH = 4

# User Authentication
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "user" and password == "user":
            return True
        else:
            st.error("Invalid username or password")
    return False

# User Details Input Page
def user_details():
    st.title("User Details")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    medical_history = st.text_area("Medical History")
    if st.button("Save Details"):
        st.success("Details saved successfully!")
        return name, age, medical_history
    return None, None, None

# Processing Functions
def determine_combination(valence, arousal):
    valence_status = "low" if valence <= VALENCE_LOW else "high"
    arousal_status = "high" if arousal >= AROUSAL_HIGH else "low"
    return valence_status, arousal_status

def stress_metrics(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "SEVERE", "red"]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MILD", "yellow"]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange"]
    return [valence, arousal, "GOOD", "green"]

def anxiety_metrics(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "SEVERE", "red"]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MILD", "yellow"]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange"]
    elif valence_status == "high" and arousal_status == "high":
        return [valence, arousal, "MODERATE", "orange"]
    return [valence, arousal, "GOOD", "green"]

def insomnia_metrics(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "SEVERE", "red"]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MILD", "yellow"]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange"]
    elif valence_status == "high" and arousal_status == "high":
        return [valence, arousal, "MODERATE", "orange"]
    return [valence, arousal, "GOOD", "green"]

def alzheimers_emotional_state(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "MILD", "red"]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "yellow"]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange"]
    elif valence_status == "high" and arousal_status == "high":
        return [valence, arousal, "GOOD", "green"]
    return [valence, arousal, "GOOD", "green"]

def process_eeg_data(df):
    try:
        predictions = model.predict(df)
        report = []
        for prediction in predictions:
            valence, arousal = float(prediction[0]), float(prediction[1])
            stress = stress_metrics(valence, arousal)
            anxiety = anxiety_metrics(valence, arousal)
            insomnia = insomnia_metrics(valence, arousal)
            alzheimers = alzheimers_emotional_state(valence, arousal)
            report.append({
                "Valence": valence,
                "Arousal": arousal,
                "Stress": stress,
                "Anxiety": anxiety,
                "Insomnia": insomnia,
                "Alzheimer's": alzheimers
            })
        return report
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return []

def create_line_chart(report_data):
    symptoms = ["Stress", "Anxiety", "Insomnia", "Alzheimer's"]
    severity_mapping = {"GOOD": 10, "NORMAL": 8, "MILD": 6, "MODERATE": 4, "SEVERE": 2}
    averages = {symptom: 0 for symptom in symptoms}

    for entry in report_data:
        for symptom in symptoms:
            averages[symptom] += severity_mapping.get(entry[symptom][2], 10)
    averages = {k: v / len(report_data) for k, v in averages.items()}

    plt.figure(figsize=(8, 5))
    plt.plot(averages.keys(), averages.values(), marker='o', linestyle='-', color='b')
    plt.title("Average Severity Levels")
    plt.ylabel("Severity (10 = Good, 2 = Severe)")
    plt.grid(True, alpha=0.6)
    st.pyplot(plt)

# Main Function
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a section", ["Login", "User Details", "EEG Analysis"])

    if app_mode == "Login":
        if login():
            st.session_state["authenticated"] = True

    if app_mode == "User Details":
        if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
            st.warning("Please log in first!")
        else:
            user_details()

    if app_mode == "EEG Analysis":
        if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
            st.warning("Please log in first!")
        else:
            st.title("EEG Disease Detection and Emotional State Analysis")
            uploaded_file = st.file_uploader("Upload your EEG data (CSV, Excel, or TXT)", type=["csv", "xlsx", "txt"])

            if uploaded_file:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension == 'csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_extension == 'xlsx':
                        df = pd.read_excel(uploaded_file)
                    elif file_extension == 'txt':
                        text = uploaded_file.read().decode('utf-8')
                        df = pd.read_csv(BytesIO(text.encode()), delim_whitespace=True)

                    st.write("Data Preview")
                    st.write(df.head())

                    report = process_eeg_data(df)
                    if report:
                        st.subheader("Analysis Report")
                        for entry in report:
                            st.write(entry)
                        create_line_chart(report)

                except Exception as e:
                    st.error(f"Error processing the file: {e}")

if __name__ == "__main__":
    main()
