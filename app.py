import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── Load artifacts ─────────────────────────────────────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# ── Feature columns (must match training order exactly) ───────────────────────
FEATURE_COLS = [
    'country', 'year', 'location_type', 'cellphone_access',
    'household_size', 'age_of_respondent', 'gender_of_respondent',
    'relationship_with_head', 'marital_status', 'education_level', 'job_type'
]

# ── Category options (real names from the dataset) ────────────────────────────
COUNTRIES          = ['Kenya', 'Rwanda', 'Tanzania', 'Uganda']
YEARS              = [2016, 2017, 2018]
LOCATION_TYPES     = ['Rural', 'Urban']
CELLPHONE_ACCESS   = ['No', 'Yes']
GENDERS            = ['Female', 'Male']
RELATIONSHIPS      = ['Child', 'Head of Household', 'Other non-relatives',
                      'Other relative', 'Parent', 'Spouse']
MARITAL_STATUSES   = ['Divorced/Seperated', 'Dont know',
                      'Married/Living together', 'Single/Never Married', 'Widowed']
EDUCATION_LEVELS   = ['No formal education', 'Other/Dont know/RTA',
                      'Primary education', 'Secondary education',
                      'Tertiary education', 'Vocational/Specialised training']
JOB_TYPES          = ['Dont Know/Refuse to answer', 'Farming and Fishing',
                      'Formally employed Government', 'Formally employed Private',
                      'Government Dependent', 'Informally employed',
                      'No Income', 'Other Income', 'Remittance Dependent', 'Self employed']


def encode_and_scale(raw: dict) -> np.ndarray:
    """Encode categorical inputs using saved encoders, then scale."""
    encoded = {}
    for col in FEATURE_COLS:
        val = raw[col]
        if col in encoders:
            encoded[col] = int(encoders[col].transform([val])[0])
        else:
            encoded[col] = val  # numeric - no encoding needed

    input_df = pd.DataFrame([encoded], columns=FEATURE_COLS)
    return scaler.transform(input_df)


# ── App layout ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title='Financial Inclusion Prediction',
        page_icon='🏦',
        layout='wide'
    )

    st.title('🏦 Financial Inclusion Prediction')
    st.write(
        'This app predicts whether an individual in East Africa is likely to have '
        'a bank account based on their demographic and household information. '
        'Fill in the details in the sidebar and click **Predict**.'
    )

    # ── Sidebar inputs ─────────────────────────────────────────────────────────
    st.sidebar.header('Respondent Details')

    country          = st.sidebar.selectbox('Country', COUNTRIES)
    year             = st.sidebar.selectbox('Survey Year', YEARS)
    location_type    = st.sidebar.selectbox('Location Type', LOCATION_TYPES)
    cellphone_access = st.sidebar.selectbox('Has Cellphone Access?', CELLPHONE_ACCESS)
    household_size   = st.sidebar.number_input(
                           'Household Size', min_value=1, max_value=21, value=3, step=1)
    age              = st.sidebar.number_input(
                           'Age of Respondent', min_value=16, max_value=100, value=30, step=1)
    gender           = st.sidebar.selectbox('Gender', GENDERS)
    relationship     = st.sidebar.selectbox('Relationship with Household Head', RELATIONSHIPS)
    marital_status   = st.sidebar.selectbox('Marital Status', MARITAL_STATUSES)
    education        = st.sidebar.selectbox('Education Level', EDUCATION_LEVELS)
    job_type         = st.sidebar.selectbox('Job Type', JOB_TYPES)

    raw_input = {
        'country': country,
        'year': year,
        'location_type': location_type,
        'cellphone_access': cellphone_access,
        'household_size': household_size,
        'age_of_respondent': age,
        'gender_of_respondent': gender,
        'relationship_with_head': relationship,
        'marital_status': marital_status,
        'education_level': education,
        'job_type': job_type
    }

    # ── Input summary table ────────────────────────────────────────────────────
    st.subheader('Respondent Summary')
    st.dataframe(pd.DataFrame([raw_input]).T.rename(columns={0: 'Value'}))

    # ── Prediction ─────────────────────────────────────────────────────────────
    if st.button('Predict'):
        input_scaled = encode_and_scale(raw_input)
        prediction = model.predict(input_scaled)[0]
        proba      = model.predict_proba(input_scaled)[0]

        has_account_prob = proba[1]
        no_account_prob  = proba[0]

        st.subheader('Prediction Result')

        if prediction == 1:
            st.success('✅ This individual is likely to **have a bank account**.')
        else:
            st.warning('⚠️ This individual is likely to **not have a bank account**.')

        col1, col2 = st.columns(2)
        col1.metric('Probability: Has Account',    f'{has_account_prob:.1%}')
        col2.metric('Probability: No Account',     f'{no_account_prob:.1%}')

        st.progress(float(has_account_prob))
        st.caption('Bar above shows probability of having a bank account (0% → 100%)')


if __name__ == '__main__':
    main()
