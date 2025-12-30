"""
Fraud Detection System - HistGradientBoosting Model
User inputs only Type, Amount, Sender ID, Receiver ID
System automatically retrieves balances from test set and calculates new balances
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #666;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .explanation-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .critical-alert {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load HistGradientBoosting model, preprocessor, and active features"""
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        hgb = joblib.load('hist_gradient_boosting.pkl')
        active_features = joblib.load('active_features.pkl')
        
        return {
            'preprocessor': preprocessor,
            'hgb': hgb,
            'active_features': active_features
        }
    except FileNotFoundError as e:
        st.error(f"❌ Required files not found. Please run save_models.py first.\n{str(e)}")
        st.stop()

# Load test data with full information (including nameOrig and nameDest)
@st.cache_data
def load_test_data():
    """Load test data with all information including nameOrig and nameDest"""
    try:
        test_data = joblib.load('test_data.pkl')
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Try to load full data from pickle first
        try:
            full_data = joblib.load('test_data_full.pkl')
            return full_data['X_test_full'], X_test, y_test
        except:
            # Try to load from CSV file
            try:
                if os.path.exists('test_data_full.csv'):
                    full_df = pd.read_csv('test_data_full.csv')
                    # Create X_test equivalent (without nameOrig, nameDest, isFraud)
                    X_test_from_csv = full_df.drop(columns=['nameOrig', 'nameDest', 'isFraud'], errors='ignore').copy()
                    y_test_from_csv = full_df['isFraud'].copy() if 'isFraud' in full_df.columns else y_test
                    return full_df, X_test_from_csv, y_test_from_csv
            except Exception as e:
                st.warning(f"⚠️ Could not load full data with names: {str(e)}")
                st.info("💡 The app will work but account balances will default to 0.0 if account IDs are not found in test data.")
                return None, X_test, y_test
    except FileNotFoundError:
        st.error("❌ Test data file not found.")
        st.stop()

def get_latest_balance_for_party(test_df_full, party_id: str, role: str):
    """
    Get the latest balance for a party (sender or receiver) from test data.
    role: 'orig' for sender (nameOrig), 'dest' for receiver (nameDest)
    Returns: (old_balance, new_balance)
    """
    if test_df_full is None or party_id is None or party_id.strip() == "":
        return 0.0, 0.0
    
    party_id = party_id.strip()
    
    try:
        if role == "orig":
            subset = test_df_full.loc[test_df_full["nameOrig"] == party_id]
            if subset.empty:
                return 0.0, 0.0
            # Get the last transaction for this sender
            row = subset.iloc[-1]
            return float(row["oldbalanceOrg"]), float(row["newbalanceOrig"])
        
        elif role == "dest":
            subset = test_df_full.loc[test_df_full["nameDest"] == party_id]
            if subset.empty:
                return 0.0, 0.0
            # Get the last transaction for this receiver
            row = subset.iloc[-1]
            return float(row["oldbalanceDest"]), float(row["newbalanceDest"])
    except:
        return 0.0, 0.0
    
    return 0.0, 0.0

def simulate_transaction_balances(tx_type: str, amount: float, sender_old: float, receiver_old: float):
    """
    Calculate realistic new balances based on transaction type and amount.
    """
    amount = max(0.0, float(amount))
    sender_new = sender_old
    receiver_new = receiver_old
    
    if tx_type in ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]:
        # Sender pays, receiver receives
        sender_new = max(0.0, sender_old - amount)
        receiver_new = receiver_old + amount
    
    elif tx_type == "CASH_IN":
        # Sender receives money
        sender_new = sender_old + amount
        receiver_new = receiver_old
    
    return sender_new, receiver_new

def calculate_engineered_features(step, hour, day, tx_type, amount, 
                                  old_org, new_org, old_dest, new_dest, is_flagged):
    """Calculate engineered features as done in preprocessing"""
    # Calculate deltas
    orig_delta = old_org - new_org
    dest_delta = new_dest - old_dest
    
    # Calculate balance errors
    balance_error_orig = orig_delta - amount
    balance_error_dest = dest_delta - amount
    
    # Clean floating-point noise
    epsilon = 1e-6
    balance_error_orig = balance_error_orig if abs(balance_error_orig) > epsilon else 0.0
    balance_error_dest = balance_error_dest if abs(balance_error_dest) > epsilon else 0.0
    
    return {
        'step': step,
        'hour': hour,
        'day': day,
        'type': tx_type,
        'amount': amount,
        'oldbalanceOrg': old_org,
        'newbalanceOrig': new_org,
        'oldbalanceDest': old_dest,
        'newbalanceDest': new_dest,
        'orig_delta': orig_delta,
        'dest_delta': dest_delta,
        'balance_error_orig': balance_error_orig,
        'balance_error_dest': balance_error_dest,
        'isFlaggedFraud': is_flagged
    }

def predict_hgb(features_dict, models):
    """Predict using HistGradientBoosting model with proper feature engineering"""
    # Create DataFrame with all required features
    active_features = models['active_features']
    
    # Build feature vector
    feature_data = {}
    for feat in active_features:
        if feat in features_dict:
            feature_data[feat] = [features_dict[feat]]
        else:
            # Default value if feature missing
            feature_data[feat] = [0.0]
    
    X = pd.DataFrame(feature_data)
    
    # Transform and predict
    X_pp = models['preprocessor'].transform(X)
    scores = models['hgb'].predict_proba(X_pp)[:, 1]
    
    # Use HGB-specific threshold (optimized threshold for HistGradientBoosting model)
    # This threshold was tuned separately for HGB model (value: 0.22)
    try:
        hgb_config = joblib.load('hgb_config.pkl')
        threshold = hgb_config.get('threshold', 0.22)
    except:
        # Default threshold for HGB (tuned value)
        threshold = 0.22
    
    preds = (scores >= threshold).astype(int)
    return preds, scores, threshold

def build_explanation_text(tx_type, amount, sender_id, receiver_id,
                           old_org, new_org, old_dest, new_dest, 
                           orig_delta, dest_delta, balance_error_orig, balance_error_dest,
                           proba, pred, has_critical_fraud=False, critical_reason=""):
    """
    Generate English explanation text with balance details and fraud indicators.
    """
    lines = []
    
    if has_critical_fraud:
        lines.append("🚨 **CRITICAL FRAUD ALERT**")
        lines.append(f"**{critical_reason}**")
        lines.append("This transaction is logically IMPOSSIBLE and must be flagged as FRAUD regardless of model score.")
        lines.append("")
    
    lines.append("**Decision Rationale (Model-Driven + Contextual Features):**")
    lines.append(f"- Transaction Type: {tx_type}")
    lines.append(f"- Transaction Amount: ${amount:,.2f}")
    lines.append(f"- Sender ID: {sender_id}")
    lines.append(f"- Receiver ID: {receiver_id}")
    lines.append("")
    
    lines.append("**Balance Dynamics Used by the System:**")
    lines.append(f"- Sender Balance: ${old_org:,.2f} → ${new_org:,.2f} (Δ {orig_delta:,.2f})")
    lines.append(f"- Receiver Balance: ${old_dest:,.2f} → ${new_dest:,.2f} (Δ {dest_delta:,.2f})")
    lines.append("")
    
    lines.append("**Risk Indicators:**")
    
    # Critical fraud checks
    if old_org == 0.0 and amount > 0 and tx_type in ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]:
        lines.append("- 🚨 **CRITICAL:** Sender has $0.00 balance but attempting to send money. This is IMPOSSIBLE and indicates fraud.")
    
    if old_org < amount and tx_type in ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]:
        lines.append(f"- 🚨 **CRITICAL:** Sender balance (${old_org:,.2f}) is less than transaction amount (${amount:,.2f}). This is IMPOSSIBLE and indicates fraud.")
    
    if abs(balance_error_orig) > 0.01:
        lines.append(f"- ⚠️ **Balance Error (Sender):** ${balance_error_orig:,.2f} - The balance change doesn't match the transaction amount. This is suspicious.")
    
    if abs(balance_error_dest) > 0.01:
        lines.append(f"- ⚠️ **Balance Error (Receiver):** ${balance_error_dest:,.2f} - The balance change doesn't match the transaction amount. This is suspicious.")
    
    if tx_type in ["TRANSFER", "CASH_OUT"]:
        lines.append(f"- ⚠️ **High-Risk Transaction Type:** {tx_type} transactions have a higher fraud rate historically.")
    
    lines.append("")
    lines.append("**Model Analysis:**")
    lines.append(f"- Model Estimated Fraud Probability: {proba:.4f} ({proba*100:.2f}%)")
    lines.append(f"- Prediction: {'FRAUD' if pred == 1 else 'NOT FRAUD'}")
    
    if has_critical_fraud:
        lines.append("")
        lines.append("**⚠️ FINAL VERDICT:** This transaction contains CRITICAL fraud indicators that make it logically impossible. It has been flagged as FRAUD regardless of model score.")
    
    return "\n".join(lines)

# Main header
st.markdown("""
<div class="main-header">
    🔍 Fraud Detection System
    <br>
    <span style="font-size: 1.2rem; font-weight: 400;">HistGradientBoosting Model</span>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner('🔄 Loading models...'):
    models = load_models()

# Load test data
with st.spinner('🔄 Loading test data...'):
    X_test_full, X_test, y_test = load_test_data()

# Input form
with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Account Information")
        sender_id = st.text_input(
            "Sender ID (nameOrig)",
            value="",
            help="Enter the sender account ID. System will retrieve balance from test data."
        )
        receiver_id = st.text_input(
            "Receiver ID (nameDest)",
            value="",
            help="Enter the receiver account ID. System will retrieve balance from test data."
        )
    
    with col2:
        st.markdown("### 💳 Transaction Details")
        tx_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
            help="Select the type of transaction"
        )
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0,
            format="%.2f",
            step=100.0,
            help="Enter the transaction amount"
        )
    
    submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

if submitted:
    # Validate inputs
    if not sender_id.strip() or not receiver_id.strip():
        st.error("❌ Please enter both Sender ID and Receiver ID")
    elif amount <= 0:
        st.error("❌ Transaction amount must be greater than 0")
    else:
        # Get balances from test data (hidden from user)
        with st.spinner('🔄 Retrieving account balances from test data...'):
            sender_old, _ = get_latest_balance_for_party(X_test_full, sender_id.strip(), role="orig")
            receiver_old, _ = get_latest_balance_for_party(X_test_full, receiver_id.strip(), role="dest")
        
        # Simulate new balances based on transaction
        sender_new, receiver_new = simulate_transaction_balances(tx_type, amount, sender_old, receiver_old)
        
        # Calculate current step, hour, day (use current time or random)
        import random
        current_step = random.randint(1, 744)
        current_hour = current_step % 24
        current_day = current_step // 24
        
        # Calculate engineered features
        orig_delta = sender_old - sender_new
        dest_delta = receiver_new - receiver_old
        balance_error_orig = orig_delta - amount
        balance_error_dest = dest_delta - amount
        
        epsilon = 1e-6
        balance_error_orig = balance_error_orig if abs(balance_error_orig) > epsilon else 0.0
        balance_error_dest = balance_error_dest if abs(balance_error_dest) > epsilon else 0.0
        
        # Check for critical fraud indicators
        has_critical_fraud = False
        critical_reason = ""
        
        if tx_type in ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]:
            if sender_old == 0.0 and amount > 0:
                has_critical_fraud = True
                critical_reason = "Sender has $0.00 balance but attempting to send money - IMPOSSIBLE!"
            elif sender_old < amount:
                has_critical_fraud = True
                critical_reason = f"Sender balance (${sender_old:,.2f}) is less than transaction amount (${amount:,.2f}) - IMPOSSIBLE!"
        
        # Build feature dictionary
        features_dict = calculate_engineered_features(
            step=current_step,
            hour=current_hour,
            day=current_day,
            tx_type=tx_type,
            amount=amount,
            old_org=sender_old,
            new_org=sender_new,
            old_dest=receiver_old,
            new_dest=receiver_new,
            is_flagged=0  # Default, can be enhanced
        )
        
        # Predict
        with st.spinner('🔄 Running fraud detection model...'):
            pred, score, threshold = predict_hgb(features_dict, models)
        
        # Override if critical fraud detected
        if has_critical_fraud:
            pred = np.array([1])
            score = np.array([0.99])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Show critical alert if needed
        if has_critical_fraud:
            st.markdown(f"""
            <div class="critical-alert">
                <h1 style='color: white; margin: 0; font-size: 2rem;'>🚨 CRITICAL FRAUD ALERT</h1>
                <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{critical_reason}</p>
                <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1rem;'>This transaction is logically IMPOSSIBLE and must be flagged as FRAUD!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display transaction information
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 2rem;'>
            <h2 style='color: #2c3e50; border-bottom: 3px solid #667eea; padding-bottom: 0.5rem;'>📋 Transaction Details</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 👤 Sender Information")
            st.write(f"**Account ID:** {sender_id.strip()}")
            st.write(f"**Old Balance:** ${sender_old:,.2f}")
            st.write(f"**New Balance:** ${sender_new:,.2f}")
            st.write(f"**Balance Change:** $-{orig_delta:,.2f}")
        
        with col2:
            st.markdown("### 👤 Receiver Information")
            st.write(f"**Account ID:** {receiver_id.strip()}")
            st.write(f"**Old Balance:** ${receiver_old:,.2f}")
            st.write(f"**New Balance:** ${receiver_new:,.2f}")
            st.write(f"**Balance Change:** $+{dest_delta:,.2f}")
        
        with col3:
            st.markdown("### 💳 Transaction Info")
            st.write(f"**Type:** {tx_type}")
            st.write(f"**Amount:** ${amount:,.2f}")
            st.write(f"**Step:** {current_step}")
            st.write(f"**Hour:** {current_hour}")
            st.write(f"**Day:** {current_day}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Prediction result
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 2rem;'>
            <h2 style='color: #2c3e50; border-bottom: 3px solid #667eea; padding-bottom: 0.5rem;'>🔍 Fraud Detection Result</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if pred[0] == 1:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); padding: 2rem; 
                border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h2 style='color: white; margin: 0;'>⚠️ FRAUD DETECTED</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 2rem; 
                border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h2 style='color: white; margin: 0;'>✅ NO FRAUD</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("📊 Fraud Probability", f"{score[0]:.4f}", 
                     delta=f"{(score[0]*100):.2f}%")
        
        with col3:
            st.metric("⚙️ Threshold Used", f"{threshold:.2f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Recommended actions
        if pred[0] == 1 or has_critical_fraud:
            st.markdown("""
            <div style='background: #fff3cd; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #ffc107; margin: 1rem 0;'>
                <h3 style='color: #856404; margin-top: 0;'>⚠️ Recommended Actions</h3>
                <ul style='font-size: 1.05rem; line-height: 2; color: #856404;'>
                    <li>Manual review of the transaction</li>
                    <li>Verify sender and receiver identities</li>
                    <li>Check transaction history for both accounts</li>
                    <li>Take additional precautionary measures</li>
                    <li>Consider blocking the transaction</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #d1ecf1; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #0c5460; margin: 1rem 0;'>
                <h3 style='color: #0c5460; margin-top: 0;'>✅ Transaction Status</h3>
                <p style='font-size: 1.05rem; line-height: 1.8; color: #0c5460;'>
                This transaction appears to be legitimate based on the model analysis. No suspicious patterns detected.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption(f"Analysis executed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <h3 style='color: #667eea; margin-bottom: 0.5rem;'>🔍 Fraud Detection System</h3>
    <p style='margin: 0; font-size: 0.9rem;'>Powered by HistGradientBoosting Model</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #999;'>Developed with Streamlit and Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
