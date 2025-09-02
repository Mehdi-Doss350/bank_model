import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set page configuration
st.set_page_config(
    page_title="Loan Application Predictor",
    page_icon="https://apiv1.2l-courtage.com/public/storage/jpg/pPy95t2JQnO2rgBOpXVJoT9HD0YW4JSgee74GNKD.jpeg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# CSS with extracted styling
# -----------------------
st.markdown("""
<style>
    @font-face {
        font-family: 'Montserrat-Bold';
        src: url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');
    }
    @font-face {
        font-family: 'Montserrat-Regular';
        src: url('https://fonts.googleapis.com/css2?family=Montserrat&display=swap');
    }
    @font-face {
        font-family: 'Montserrat-ExtraBold';
        src: url('https://fonts.googleapis.com/css2?family=Montserrat:wght@800&display=swap');
    }
    @font-face {
        font-family: 'Montserrat-SemiBold';
        src: url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600&display=swap');
    }
    
    .main-header { 
        font-family: 'Montserrat', Arial, sans-serif;
        font-weight: 800;
        font-size: 2.2rem; 
        color: #22abc5; 
        text-align: center; 
        margin-bottom: 2rem; 
        letter-spacing: 1px;
    }
    .section-header { 
        font-family: 'Montserrat', Arial, sans-serif;
        font-weight: 700;
        font-size: 1.1rem; 
        text-align: left; 
        margin-top: 15px; 
        margin-bottom: 10px; 
        width: 100%; 
        border-bottom: 3px solid #22abc5; /* Make the underline blue and a bit thicker */
        color: #22abc5; 
        text-transform: uppercase; 
        letter-spacing: 1px;
    }
    .prediction-card { 
        background-color: #F58C29; 
        padding: 2rem; 
        border-radius: 12px; 
        margin-top: 2rem; 
        box-shadow: 0 4px 12px rgba(34,171,197,0.08); 
        border: 1px solid #F58C29;
    }
    .risk-high { color: #F58C29; font-weight: bold; }
    .risk-medium { color: #F58C29; font-weight: bold; }
    .risk-low { color: #22abc5; font-weight: bold; }
    .stButton>button, button[kind="primary"], .css-1cpxqw2, .css-1emrehy {
        width: 100%;
        background-color: #22abc5 !important;
        color: #fff !important;
        font-size: 1.2rem !important;
        padding: 0.7rem 0 !important;
        font-family: 'Montserrat', Arial, sans-serif !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(34,171,197,0.08) !important;
        transition: background 0.2s, color 0.2s !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        margin-top: 18px !important;
        margin-bottom: 8px !important;
    }
    .stButton>button:hover, button[kind="primary"]:hover, .css-1cpxqw2:hover, .css-1emrehy:hover,
    .stButton>button:focus, button[kind="primary"]:focus, .css-1cpxqw2:focus, .css-1emrehy:focus {
        background-color: #F58C29 !important;
        color: #fff !important;
        border: none !important;
        outline: none !important;
    }
    .info-label {
        font-family: 'Montserrat', Arial, sans-serif;
        font-size: 1rem;
        color: #22abc5;
        font-weight: 500;
    }
    .info-value {
        font-family: 'Montserrat', Arial, sans-serif;
        font-size: 1rem;
        color: #F58C29;
        font-weight: 700;
    }
    .column-title {
        font-family: 'Montserrat', Arial, sans-serif;
        font-weight: 700;
        text-align: left;
        vertical-align: middle;
        padding: 6px;
        margin-left: 5px;
        color: #FFFFFF;
        font-size: 1rem;
        border-left-style: solid;
        background: #22abc5;
        border-radius: 4px 4px 0 0;
    }
    .metric-card {
        background-color: #eaf7fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #22abc5;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(34,171,197,0.06);
    }
    .stTabs [role="tab"] {
        background: #22abc5 !important;
        color: #fff !important;
        font-family: 'Montserrat', Arial, sans-serif;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        margin-right: 6px;
        padding: 12px 32px !important;
        box-shadow: 0 2px 8px rgba(34,171,197,0.10);
        border: 1px solid #22abc5;
        border-bottom: none;
        transition: background 0.2s, color 0.2s;
        letter-spacing: 1px;
        font-size: 1.05rem;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: #F58C29 !important;
        color: #fff !important;
        border: 1px solid #F58C29;
        border-bottom: none;
        box-shadow: 0 4px 16px rgba(245,140,41,0.10);
    }
    .stTabs [role="tab"]:hover {
        background: #F58C29 !important;
        color: #fff !important;
        cursor: pointer;
    }
    /* Add a subtle shadow to the tab bar */
    .stTabs {
        box-shadow: 0 2px 8px rgba(34,171,197,0.07);
        margin-bottom: 24px;
        border-radius: 8px 8px 0 0;
        background: #f7fafd;
    }

    /* Add blue underline to all Streamlit input labels */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label {
        border-bottom: 2px solid #22abc5 !important;
        padding-bottom: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Logo and chart functions
# -----------------------
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# For demonstration, we'll create a function to generate a chart
def create_risk_chart(probability):
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Create gradient risk bar
    gradient = np.linspace(0, 100, 300).reshape(1, -1)
    ax.imshow(gradient, extent=[0, 100, 0, 1], aspect='auto', cmap='RdYlGn_r')
    
    # Add marker for current probability
    ax.axvline(x=probability, color='black', linestyle='--', linewidth=2)
    ax.plot(probability, 0.5, 'ko', markersize=10)
    ax.text(probability, 1.1, f'{probability:.1f}%', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize chart
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Risk Level (%)', fontsize=10)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add risk labels
    ax.text(10, -0.2, 'High Risk', ha='center', va='top', fontsize=9)
    ax.text(50, -0.2, 'Medium', ha='center', va='top', fontsize=9)
    ax.text(90, -0.2, 'Low Risk', ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    # Encode to base64
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return data

# -----------------------
# Robust model + scaler loader
# -----------------------
@st.cache_resource
def load_model_and_scaler(model_path='ultra_fast_model.pkl', scaler_path='scaler.pkl'):
    """
    Load model and scaler. The joblib file may contain:
      - a raw estimator, or
      - a dict with keys like 'model' and/or 'scaler', or
      - some wrapper object.
    This function tries to extract an actual estimator and a scaler if present.
    """
    try:
        loaded = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model file '{model_path}': {e}")
        return None, None

    # default
    estimator = None
    scaler = None

    # If loaded is dict-like, try to find estimator & scaler inside
    if isinstance(loaded, dict):
        # direct keys first
        if 'model' in loaded:
            estimator = loaded['model']
        elif 'estimator' in loaded:
            estimator = loaded['estimator']

        if 'scaler' in loaded:
            scaler = loaded['scaler']
        elif 'preprocessor' in loaded:
            scaler = loaded['preprocessor']

        # If we still don't have an estimator, search values for any object with predict
        if estimator is None:
            for v in loaded.values():
                if hasattr(v, 'predict') or hasattr(v, 'predict_proba'):
                    estimator = v
                    break

        # If estimator still None, keep the dict (will be handled later)
        if estimator is None:
            estimator = loaded

    else:
        # loaded is not a dict: hopefully it's the estimator
        estimator = loaded

    # If no scaler found inside model file, try to load separate scaler.pkl
    if scaler is None:
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None  # not fatal; we'll warn later

    return estimator, scaler

# -----------------------
# Derived features & preprocess
# -----------------------
def calculate_derived_features(input_data):
    data = input_data.copy()
    data['total_household_income'] = (
        data.get('borrower_salaire_mensuel', 0.0)
        + data.get('borrower_revenu_foncier', 0.0)
        + data.get('borrower_autres_revenus', 0.0)
        + data.get('co_borrower_salaire_mensuel', 0.0)
        + data.get('co_borrower_autres_revenus', 0.0)
    )
    data['total_project_cost'] = (
        data.get('cost_terrain', 0.0)
        + data.get('cost_logement', 0.0)
        + data.get('cost_travaux', 0.0)
        + data.get('cost_frais_notaire', 0.0)
    )
    data['debt_to_income_ratio'] = (
        data.get('montant_credit_initio', 0.0) / data['total_household_income']
        if data['total_household_income'] > 0 else 0.0
    )
    data['apport_percentage'] = (
        data.get('financing_apport_personnel', 0.0) / data['total_project_cost']
        if data['total_project_cost'] > 0 else 0.0
    )
    data['loan_to_value'] = (
        data.get('financing_pret_principal', 0.0) / data['total_project_cost']
        if data['total_project_cost'] > 0 else 0.0
    )

    data['has_viabilisation_costs'] = 1 if data.get('cost_viabilisation', 0) > 0 else 0
    data['has_mobilier_costs'] = 1 if data.get('cost_mobilier', 0) > 0 else 0
    data['has_agency_fees'] = 1 if data.get('cost_agency_fees', 0) > 0 else 0

    defaults = {
        'number_of_properties': 0,
        'total_credit_remaining_amount': 0.0,
        'total_credit_monthly_payment': 0.0,
        'nombre_of_credits': 0,
        'net_worth': 0.0,
    }
    for k, v in defaults.items():
        data.setdefault(k, v)
    return data

def prepare_input_data(input_data, scaler):
    expected_features = [
        'montant_credit_initio', 'co_borrower_categ_socio_prof',
        'co_borrower_contrat_travail', 'borrower_salaire_mensuel',
        'borrower_revenu_foncier', 'borrower_autres_revenus',
        'co_borrower_salaire_mensuel', 'co_borrower_autres_revenus',
        'project_nature', 'project_destination', 'project_zone',
        'project_type_logement', 'cost_terrain', 'cost_logement',
        'cost_travaux', 'cost_frais_notaire', 'financing_apport_personnel',
        'financing_pret_principal', 'total_credit_remaining_amount',
        'total_credit_monthly_payment', 'nombre_of_credits', 'net_worth',
        'number_of_properties', 'total_household_income', 'total_project_cost',
        'has_viabilisation_costs', 'has_mobilier_costs', 'has_agency_fees',
        'debt_to_income_ratio', 'apport_percentage', 'loan_to_value'
    ]
    df = pd.DataFrame({f: [0] for f in expected_features})
    for f, v in input_data.items():
        if f in df.columns:
            df.at[0, f] = v

    numeric_features = [
        'borrower_salaire_mensuel', 'co_borrower_salaire_mensuel',
        'cost_travaux', 'total_household_income', 'montant_credit_initio',
        'total_project_cost', 'financing_apport_personnel', 'financing_pret_principal',
        'debt_to_income_ratio', 'apport_percentage', 'loan_to_value'
    ]
    for col in numeric_features:
        if col not in df.columns:
            df[col] = 0.0

    if scaler is not None:
        try:
            df[numeric_features] = scaler.transform(df[numeric_features])
        except Exception as e:
            st.warning(f"Scaler transform failed — proceeding without scaling: {e}")
    else:
        st.warning("No scaler loaded — proceeding without scaling.")

    return df

# -----------------------
# Helpers for output
# -----------------------
def categorize_risk(probability):
    if probability >= 90: return "Very Low Risk"
    elif probability >= 70: return "Low Risk"
    elif probability >= 50: return "Medium Risk"
    elif probability >= 30: return "High Risk"
    else: return "Very High Risk"

def get_confidence_level(probability):
    distance = abs(probability - 50)
    if distance > 40: return "VERY HIGH"
    elif distance > 25: return "HIGH"
    elif distance > 15: return "MEDIUM"
    else: return "LOW"

# -----------------------
# App UI
# -----------------------
def main():
    # Header with logo
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        # Using a placeholder for the logo - in a real app, you would use the actual logo URL
        st.image("https://apiv1.2l-courtage.com/public/storage/jpg/pPy95t2JQnO2rgBOpXVJoT9HD0YW4JSgee74GNKD.jpeg", 
                 width=100)
    with col_title:
        st.markdown("""
        <div style="
            font-family: 'Montserrat', Arial, sans-serif;
            font-weight: 900;
            font-size: 2.8rem;
            color: #22abc5;
            background: linear-gradient(90deg, #22abc5 60%, #22abc5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
            text-align: center;
            letter-spacing: 2px;
            margin-bottom: 2rem;
            padding: 0.5rem 0;
            border-radius: 12px;
        ">
            Loan Application Predictor
        </div>
    """, unsafe_allow_html=True)

    model, scaler = load_model_and_scaler()
    if model is None:
        st.error("Model could not be loaded. Check model file.")
        return

    # (UI inputs — keep unique keys)
    tab1, tab2, tab3, tab4 = st.tabs(["Borrower Information", "Project Details", "Financial Information", "Existing Credits & Assets"])
    with st.form("loan_application_form"):
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-header">Borrower Info</div>', unsafe_allow_html=True)
                borrower_salaire_mensuel = st.number_input("Monthly Salary (€)", min_value=0.0, value=0.0, step=100.0, key="b_salary")
                borrower_revenu_foncier = st.number_input("Rental Income (€)", min_value=0.0, value=0.0, step=100.0, key="b_rental")
                borrower_autres_revenus = st.number_input("Other Income (€)", min_value=0.0, value=0.0, step=100.0, key="b_other")
            with col2:
                st.markdown('<div class="section-header">Co-Borrower Info</div>', unsafe_allow_html=True)
                co_borrower_salaire_mensuel = st.number_input("Co-borrower Salary (€)", min_value=0.0, value=0.0, step=100.0, key="cb_salary")
                co_borrower_autres_revenus = st.number_input("Co-borrower Other Income (€)", min_value=0.0, value=0.0, step=100.0, key="cb_other")
                co_borrower_categ_socio_prof = st.selectbox("Socio-professional Category", options=[0,1,2,3,4,5], index=0, key="cb_cat")
                co_borrower_contrat_travail = st.selectbox("Employment Contract", options=[0,1,2,3], index=0, key="cb_contract")

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-header">Project Details</div>', unsafe_allow_html=True)
                project_nature = st.selectbox("Project Nature", options=list(range(10)), index=0, key="proj_nature")
                project_destination = st.selectbox("Project Destination", options=list(range(5)), index=0, key="proj_dest")
                project_zone = st.selectbox("Project Zone", options=list(range(6)), index=0, key="proj_zone")
                project_type_logement = st.selectbox("Housing Type", options=list(range(6)), index=0, key="proj_housing")
            with col2:
                st.markdown('<div class="section-header">Project Costs</div>', unsafe_allow_html=True)
                cost_terrain = st.number_input("Land Cost (€)", min_value=0.0, value=0.0, step=1000.0, key="cost_terrain")
                cost_logement = st.number_input("Housing Cost (€)", min_value=0.0, value=0.0, step=1000.0, key="cost_logement")
                cost_travaux = st.number_input("Work Cost (€)", min_value=0.0, value=0.0, step=1000.0, key="cost_travaux")
                cost_frais_notaire = st.number_input("Notary Fees (€)", min_value=0.0, value=0.0, step=100.0, key="cost_notaire")

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-header">Loan Details</div>', unsafe_allow_html=True)
                montant_credit_initio = st.number_input("Loan Amount (€)", min_value=0.0, value=0.0, step=1000.0, key="loan_amount")
                financing_pret_principal = st.number_input("Principal Loan (€)", min_value=0.0, value=0.0, step=1000.0, key="loan_principal")
                financing_apport_personnel = st.number_input("Personal Contribution (€)", min_value=0.0, value=0.0, step=1000.0, key="loan_apport")
            with col2:
                st.markdown('<div class="section-header">Additional Costs</div>', unsafe_allow_html=True)
                cost_viabilisation = st.number_input("Utilities Cost (€)", min_value=0.0, value=0.0, step=100.0, key="cost_viab")
                cost_mobilier = st.number_input("Furniture Cost (€)", min_value=0.0, value=0.0, step=100.0, key="cost_mob")
                cost_agency_fees = st.number_input("Agency Fees (€)", min_value=0.0, value=0.0, step=100.0, key="cost_agency")

        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-header">Existing Credits</div>', unsafe_allow_html=True)
                total_credit_remaining_amount = st.number_input("Total Credit Remaining (€)", min_value=0.0, value=0.0, step=1000.0, key="credit_remaining")
                total_credit_monthly_payment = st.number_input("Monthly Credit Payments (€)", min_value=0.0, value=0.0, step=100.0, key="credit_monthly")
                nombre_of_credits = st.number_input("Number of Credits", min_value=0, value=0, step=1, key="num_credits")
            with col2:
                st.markdown('<div class="section-header">Assets</div>', unsafe_allow_html=True)
                net_worth = st.number_input("Net Worth (€)", min_value=0.0, value=0.0, step=1000.0, key="net_worth")
                number_of_properties = st.number_input("Number of Properties", min_value=0, value=0, step=1, key="num_properties")

        submitted = st.form_submit_button("Predict Loan Acceptance")

    # Prediction flow
    if submitted:
        input_data = {
            'montant_credit_initio': montant_credit_initio,
            'co_borrower_categ_socio_prof': co_borrower_categ_socio_prof,
            'co_borrower_contrat_travail': co_borrower_contrat_travail,
            'borrower_salaire_mensuel': borrower_salaire_mensuel,
            'borrower_revenu_foncier': borrower_revenu_foncier,
            'borrower_autres_revenus': borrower_autres_revenus,
            'co_borrower_salaire_mensuel': co_borrower_salaire_mensuel,
            'co_borrower_autres_revenus': co_borrower_autres_revenus,
            'project_nature': project_nature,
            'project_destination': project_destination,
            'project_zone': project_zone,
            'project_type_logement': project_type_logement,
            'cost_terrain': cost_terrain,
            'cost_logement': cost_logement,
            'cost_travaux': cost_travaux,
            'cost_frais_notaire': cost_frais_notaire,
            'financing_apport_personnel': financing_apport_personnel,
            'financing_pret_principal': financing_pret_principal,
            'total_credit_remaining_amount': total_credit_remaining_amount,
            'total_credit_monthly_payment': total_credit_monthly_payment,
            'nombre_of_credits': nombre_of_credits,
            'net_worth': net_worth,
            'number_of_properties': number_of_properties,
            'cost_viabilisation': cost_viabilisation,
            'cost_mobilier': cost_mobilier,
            'cost_agency_fees': cost_agency_fees,
        }
        input_data = calculate_derived_features(input_data)

        try:
            prepared = prepare_input_data(input_data, scaler)

            # Determine estimator object (in case 'model' variable is still a dict)
            estimator = model
            if isinstance(model, dict):
                # prefer direct 'model' key
                if 'model' in model and (hasattr(model['model'], 'predict') or hasattr(model['model'], 'predict_proba')):
                    estimator = model['model']
                else:
                    # search values for something that looks like an estimator
                    for v in model.values():
                        if hasattr(v, 'predict') or hasattr(v, 'predict_proba'):
                            estimator = v
                            break

            # Make prediction with robust handling
            if hasattr(estimator, 'predict_proba'):
                probs = estimator.predict_proba(prepared)
                # try common shapes
                try:
                    acceptance_prob = float(probs[0, 1]) * 100.0
                except Exception:
                    # maybe returns prob of positive class only as 1D
                    acceptance_prob = float(probs[0]) * 100.0
            elif hasattr(estimator, 'predict'):
                pred = estimator.predict(prepared)[0]
                acceptance_prob = 100.0 if pred == 1 else 0.0
            else:
                raise RuntimeError("Loaded object does not support predict or predict_proba.")

            risk_category = categorize_risk(acceptance_prob)
            confidence_level = get_confidence_level(acceptance_prob)
            is_accepted = acceptance_prob >= 50.0
            
            st.header("Prediction Results")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Acceptance Probability", f"{acceptance_prob:.1f}%")
            with res_col2:
                decision_color = "green" if is_accepted else "red"
                decision_text = "ACCEPTED" if is_accepted else "REJECTED"
                st.markdown(f"<h2 style='color: {decision_color}; text-align: center;'>{decision_text}</h2>", unsafe_allow_html=True)
            with res_col3:
                risk_color = "red" if 'High' in risk_category else "orange" if 'Medium' in risk_category else "green"
                st.markdown(f"<h3 style='color: {risk_color}; text-align: center;'>Risk: {risk_category}</h3>", unsafe_allow_html=True)

            st.subheader("Calculated Financial Metrics")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f'<div class="metric-card"><span class="info-label">Total Household Income:</span> <span class="info-value">{input_data.get("total_household_income", 0):.0f} €</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><span class="info-label">Total Project Cost:</span> <span class="info-value">{input_data.get("total_project_cost", 0):.0f} €</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><span class="info-label">Debt-to-Income Ratio:</span> <span class="info-value">{input_data.get("debt_to_income_ratio", 0):.2f}</span></div>', unsafe_allow_html=True)
            with info_col2:
                st.markdown(f'<div class="metric-card"><span class="info-label">Loan-to-Value:</span> <span class="info-value">{input_data.get("loan_to_value", 0):.2f}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><span class="info-label">Personal Contribution:</span> <span class="info-value">{input_data.get("apport_percentage", 0)*100:.1f}%</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><span class="info-label">Confidence Level:</span> <span class="info-value">{confidence_level}</span></div>', unsafe_allow_html=True)

            if is_accepted:
                if acceptance_prob >= 70:
                    st.success("This application shows strong financial indicators and is very likely to be approved.")
                else:
                    st.warning("This application has a moderate chance of approval. Review details.")
            else:
                st.error("This application shows significant risk factors that make approval unlikely.")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()