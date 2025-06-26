import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

# Set background image using HTML + CSS
page_bg_img = f'''
<style>
.stApp {{
    background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),url("https://media.istockphoto.com/id/1334591614/photo/man-using-digital-tablet-online-connect-to-internet-banking-currency-exchange-online-shopping.jpg?s=612x612&w=0&k=20&c=nejA5SuHcN2fAdO7Bkaf9pJrwzyLPBCyOLZgMaslGko=");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


st.markdown("# 💳 Credit Risk Prediction Dashboard")
st.markdown("Use this dashboard to analyze the credit risk of applicants based on financial profile inputs.")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

saving_map = {'none': 0, 'little': 1, 'moderate': 2, 'rich': 3, 'quite rich': 4}
checking_map = {'unknown': 0, 'little': 1, 'moderate': 2, 'rich': 3}
sex_map = {'male': 1, 'female': 0}
housing_map = {'own': 1, 'free': 2, 'rent': 0}
purpose_map = {
    'radio/TV': 5, 'education': 3, 'furniture/equipment': 4, 'car': 1,
    'business': 0, 'domestic appliances': 2, 'repairs': 6, 'vacation/others': 7
}

with st.form("form"):
    age = st.number_input("Age", 18, 100, help="Enter age between 18 and 100.")
    sex = st.selectbox("Sex", list(sex_map.keys()))
    job = st.selectbox("Job (0: unskilled, 3: highly skilled)", [0, 1, 2, 3])
    housing = st.selectbox("Housing", list(housing_map.keys()))
    saving = st.selectbox("Saving Account", list(saving_map.keys()))
    checking = st.selectbox("Checking Account", list(checking_map.keys()))
    credit = st.number_input("Credit Amount", 0, 100000, 2000, help="Total amount of the loan.")
    duration = st.slider("Loan Duration (months)", 4, 72, 12, help="Total duration to repay the loan.")
    purpose = st.selectbox("Purpose", list(purpose_map.keys()))
    submit = st.form_submit_button("Predict")

if submit:
    if credit <= 0 or duration <= 0:
        st.warning("⚠️ Credit amount and duration must be greater than zero.")
        st.stop()

    input_data = pd.DataFrame([[
        age, sex_map[sex], job, housing_map[housing],
        saving_map[saving], checking_map[checking],
        credit, duration, purpose_map[purpose]
    ]], columns=[
        'Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
        'Checking account', 'Credit amount', 'Duration', 'Purpose'
    ])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    high_risk_prob = proba[1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Credit Amount", f"₹{credit:,}")
    col2.metric("Loan Duration", f"{duration} months")
    col3.metric("Risk Score", f"{high_risk_prob * 100:.1f}%", delta=f"{(high_risk_prob - 0.5) * 100:.1f}%", delta_color="inverse" if high_risk_prob > 0.5 else "normal")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error("⚠️ High Risk Applicant")
        else:
            st.success("✅ Low Risk Applicant")

        risk_meter = "🟩" * int(high_risk_prob * 10) + "⬜" * (10 - int(high_risk_prob * 10))
        st.markdown(f"**Emoji Risk Meter:**\n{risk_meter}")

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=high_risk_prob * 100,
            number={'suffix': "%", 'font': {'size': 28}},
            title={'text': "Risk Probability (%)"},
            gauge={
                'shape': "angular",
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if high_risk_prob > 0.6 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "#00cc96"},
                    {'range': [40, 70], 'color': "#ffa600"},
                    {'range': [70, 100], 'color': "#ef553b"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': high_risk_prob * 100
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📊 Numeric Input Summary")
        bar_df = pd.DataFrame({
            'Feature': ['Age', 'Credit Amount', 'Loan Duration'],
            'Value': [age, credit, duration]
        })
        fig_bar = px.bar(
            bar_df,
            x='Feature',
            y='Value',
            color='Feature',
            text='Value',
            template='plotly_white'
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col4:
        st.subheader("🌐 Categorical Radar Profile")
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                sex_map[sex],
                job,
                housing_map[housing],
                saving_map[saving],
                checking_map[checking],
                purpose_map[purpose]
            ],
            theta=["Sex", "Job", "Housing", "Saving", "Checking", "Purpose"],
            fill='toself'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 7])),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    st.subheader("🧠 Feature Importance (Model-Based)")
    try:
        importance_values = model.feature_importances_
        feature_names = input_data.columns

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values(by="Importance", ascending=False)

        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Reds',
            title="Which Features Matter Most to the Model?",
            template='plotly_white'
        )
        fig_importance.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_importance, use_container_width=True)

    except AttributeError:
        st.warning("⚠️ Feature importance is not available for this model type.")
    
    