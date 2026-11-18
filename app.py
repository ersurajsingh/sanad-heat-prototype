import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------------------------------------
# PAGE CONFIG & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sanad-Heat | Edge Worker Heat-Strain & Privacy Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .privacy-badge {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .alert-box-green { background-color: #dcfce7; border-left: 6px solid #16a34a; padding: 14px; border-radius: 6px; color: #14532d; }
    .alert-box-amber { background-color: #fef3c7; border-left: 6px solid #d97706; padding: 14px; border-radius: 6px; color: #78350f; }
    .alert-box-red   { background-color: #fee2e2; border-left: 6px solid #dc2626; padding: 14px; border-radius: 6px; color: #7f1d1d; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MULTILINGUAL ALERT DICTIONARY (UAE WORKFORCE LANGUAGES)
# ---------------------------------------------------------
ALERTS_I18N = {
    "English": {
        "GREEN": ("SAFE / NORMAL", "Continue normal pace. Drink 250ml water every 30 mins.", "1x Short Pulse (500ms)"),
        "AMBER": ("CAUTION: RISING HEAT STRAIN", "Hydrate immediately (500ml) and move to shaded zone for 15-min rotation.", "2x Medium Pulses + Amber LED"),
        "RED":   ("CRITICAL HEAT ALERT", "STOP WORK IMMEDIATELY. Move to cooling station. Site paramedic notified.", "Continuous Haptic + Audio Alarm")
    },
    "Arabic (العربية)": {
        "GREEN": ("آمن / طبيعي", "استمر في العمل بوتيرة طبيعية. اشرب ٢٥٠ مل من الماء كل ٣٠ دقيقة.", "نبضة اهتزاز قصيرة (٥٠٠ مللي ثانية)"),
        "AMBER": ("تنبيه: ارتفاع الإجهاد الحراري", "اشرب الماء فوراً (٥٠٠ مل) وانتقل إلى منطقة مظللة للراحة لمدة ١٥ دقيقة.", "نبضتان متوسطتان + إضاءة صفراء"),
        "RED":   ("تحذير حرج من الإجهاد الحراري", "توقف عن العمل فوراً. توجه إلى محطة التبريد. تم إبلاغ المسعف الميداني.", "اهتزاز مستمر + إنذار صوتي")
    },
    "Hindi (हिन्दी)": {
        "GREEN": ("सुरक्षित / सामान्य", "सामान्य गति से काम जारी रखें। हर 30 मिनट में 250ml पानी पिएं।", "1x हल्का वाइब्रेशन (500ms)"),
        "AMBER": ("चेतावनी: गर्मी का दबाव बढ़ रहा है", "तुरंत 500ml पानी पिएं और 15 मिनट के आराम के लिए छायादार क्षेत्र में जाएं।", "2x मध्यम वाइब्रेशन + पीली लाइट"),
        "RED":   ("गंभीर हीट अलर्ट", "तुरंत काम रोकें! कूलिंग स्टेशन पर जाएं। साइट पैरामेडिक को सूचित कर दिया गया है।", "लगातार वाइब्रेशन + ऑडियो अलार्म")
    },
    "Urdu (اردو)": {
        "GREEN": ("محفوظ / نارمل", "معمول کے مطابق کام جاری رکھیں۔ ہر 30 منٹ بعد 250 ملی لیٹر پانی پئیں۔", "1x مختصر وائبریشن"),
        "AMBER": ("انتباہ: گرمی کا دباؤ بڑھ رہا ہے", "فوری طور پر پانی پئیں اور 15 منٹ آرام کے لیے سایہ دار جگہ پر جائیں۔", "2x درمیانی وائبریشن"),
        "RED":   ("شدید گرمی کا خطرہ", "فوری طور پر کام روک دیں۔ کولنگ اسٹیشن پر جائیں۔ میڈیکل ٹیم کو اطلاع دے دی گئی ہے۔", "مسلسل وائبریشن + الارم")
    },
    "Tagalog": {
        "GREEN": ("LIGTAS / NORMAL", "Ipagpatuloy ang trabaho. Uminom ng 250ml na tubig kada 30 minuto.", "1x Maikling Vibration"),
        "AMBER": ("BABALA: TUMATAAS NA INIT SA KATAWAN", "Uminom agad ng tubig (500ml) at magpahinga sa lilim nang 15 minuto.", "2x Katamtamang Vibration"),
        "RED":   ("KRITIKAL NA BABALA SA INIT", "ITIGIL AGAD ANG TRABAHO. Pumunta sa cooling station. Inabisuhan na ang medic.", "Tuloy-tuloy na Vibration + Alarm")
    }
}

# ---------------------------------------------------------
# PHYSIOLOGICAL & MICROCLIMATE EDGE MODELS (ISO 7243 + EKF)
# ---------------------------------------------------------
def compute_wbgt_outdoor(temp_c: float, rh_pct: float, solar_load_w: float = 650.0) -> float:
    """Approximates outdoor Wet Bulb Globe Temperature (WBGT) using Stull wet-bulb + radiant load."""
    tw = temp_c * np.arctan(0.151977 * np.sqrt(rh_pct + 8.313659)) + \
         np.arctan(temp_c + rh_pct) - np.arctan(rh_pct - 1.676331) + \
         0.00391838 * (rh_pct ** 1.5) * np.arctan(0.023101 * rh_pct) - 4.686035
    tg = temp_c + (0.014 * solar_load_w)  # Black globe approximation under Gulf solar load
    wbgt = 0.7 * tw + 0.2 * tg + 0.1 * temp_c
    return round(float(wbgt), 2)

def predict_tcore_trajectory(wbgt: float, hr_bpm: int, workload_factor: float, ppe_insulation: float, consent_hr: bool):
    """
    Simulates on-chip Extended Kalman Filter (EKF) Core Body Temperature (T_core) trajectory
    over the next 20 minutes (steps of 2 mins).
    """
    minutes = np.arange(0, 22, 2)
    base_tcore = 37.0 + max(0.0, (wbgt - 28.0) * 0.045)
    if consent_hr:
        base_tcore += max(0.0, (hr_bpm - 85) * 0.011)
    else:
        # Fallback estimation using environmental + workload dynamics when HR consent is toggled off
        base_tcore += workload_factor * 0.35

    drift_rate = (max(0.0, wbgt - 29.5) * 0.0045 + workload_factor * 0.012) * ppe_insulation
    tcore_mean = base_tcore + drift_rate * minutes
    uncertainty = 0.08 + 0.006 * minutes  # Confidence interval widens slightly over horizon
    return minutes, tcore_mean, uncertainty

def classify_strain(wbgt: float, tcore_20m: float):
    if tcore_20m >= 38.3 or wbgt >= 33.5:
        return "RED", "Critical Heat Strain (15-min Lead Warning)"
    elif tcore_20m >= 37.75 or wbgt >= 31.0:
        return "AMBER", "Elevated Thermal Drift — Rotation Recommended"
    return "GREEN", "Normal Thermoregulation"

# ---------------------------------------------------------
# HEADER & GUEST ACCESS BANNER
# ---------------------------------------------------------
col_logo, col_badge = st.columns([3, 1])
with col_logo:
    st.title("🛡️ Sanad-Heat")
    st.caption("Privacy-First Edge Hardware & AI Early Warning for Climate-Resilient Worker Health | IndusArva Innovations")
with col_badge:
    st.markdown("<br><div class='privacy-badge'>🔓 Guest Evaluator Mode Active • Zero PII Stored</div>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# SIDEBAR: EDGE SENSOR PUCK & CONSENT SIMULATOR
# ---------------------------------------------------------
with st.sidebar:
    st.header("📟 Edge Micro-Puck Input")
    st.markdown("Simulate live telemetry from the **ESP32-S3 + BME280 + PPG** helmet/vest clip:")

    preset = st.selectbox(
        "Quick Scenario Preset",
        [
            "Custom Live Sliders",
            "Abu Dhabi Summer Peak (12:30 PM - High Strain)",
            "Morning Shift Baseline (08:00 AM - Safe)",
            "Heavy PPE Welding Crew (Elevated Drift)"
        ]
    )

    if preset == "Abu Dhabi Summer Peak (12:30 PM - High Strain)":
        def_temp, def_rh, def_hr, def_work, def_ppe = 47.5, 62, 138, 1.4, 1.25
    elif preset == "Morning Shift Baseline (08:00 AM - Safe)":
        def_temp, def_rh, def_hr, def_work, def_ppe = 34.0, 45, 92, 0.8, 1.0
    elif preset == "Heavy PPE Welding Crew (Elevated Drift)":
        def_temp, def_rh, def_hr, def_work, def_ppe = 42.0, 58, 126, 1.3, 1.4
    else:
        def_temp, def_rh, def_hr, def_work, def_ppe = 43.5, 60, 122, 1.2, 1.15

    temp_c = st.slider("Ambient Temperature (°C)", 25.0, 54.0, def_temp, 0.5)
    rh_pct = st.slider("Relative Humidity (%)", 15, 95, def_rh, 1)
    workload = st.slider("Metabolic Workload Index (MET Proxy)", 0.5, 2.0, def_work, 0.1)
    ppe_factor = st.slider("PPE Thermal Insulation Multiplier", 1.0, 1.6, def_ppe, 0.05)

    st.subheader("🔐 Worker Privacy & Consent Controls")
    consent_hr = st.toggle("Opt-In Consented PPG Heart Rate", value=True, help="Worker controls whether optical heart-rate is fused on-chip. Even when enabled, raw BPM never leaves the puck.")
    hr_bpm = st.slider("On-Chip PPG Heart Rate (BPM)", 60, 175, def_hr, 1, disabled=not consent_hr)

    selected_lang = st.selectbox("Worker Native Alert Language", list(ALERTS_I18N.keys()))

# ---------------------------------------------------------
# COMPUTE EDGE METRICS
# ---------------------------------------------------------
wbgt_val = compute_wbgt_outdoor(temp_c, rh_pct)
mins, tcore_pred, tcore_unc = predict_tcore_trajectory(wbgt_val, hr_bpm, workload, ppe_factor, consent_hr)
tcore_now = round(float(tcore_pred[0]), 2)
tcore_20m = round(float(tcore_pred[-1]), 2)
risk_level, risk_desc = classify_strain(wbgt_val, tcore_20m)

# ---------------------------------------------------------
# TABS: DUAL-LAYER ARCHITECTURE
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "1️⃣ Worker Edge Node & Multilingual Alerts",
    "2️⃣ Anonymized Supervisor & Smart Break Planner",
    "3️⃣ Edge Hardware & Privacy Architecture"
])

# === TAB 1: WORKER EDGE NODE ===
with tab1:
    st.subheader("On-Chip Edge Risk Calculation (<50ms Local Inference)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Local WBGT Index", f"{wbgt_val} °C", delta="High Heat" if wbgt_val >= 31 else "Normal", delta_color="inverse")
    m2.metric("Estimated Core Temp (Now)", f"{tcore_now} °C")
    m3.metric("Predicted T_core (+20 min)", f"{tcore_20m} °C", delta=f"{round(tcore_20m - tcore_now, 2)} °C drift", delta_color="inverse")
    m4.metric("Cloud Biometric Leakage", "0 Bytes", delta="100% On-Chip Privacy", delta_color="normal")

    # Alert Banner in Worker's Selected Language
    title_txt, action_txt, haptic_txt = ALERTS_I18N[selected_lang][risk_level]
    box_class = f"alert-box-{risk_level.lower()}"
    st.markdown(f"""
    <div class="{box_class}">
        <strong>🔔 LIVE PUCK ALERT ({selected_lang}) — {title_txt}</strong><br>
        <b>Action Instruction:</b> {action_txt}<br>
        <b>Hardware Actuator Output:</b> <code>{haptic_txt}</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly 20-Minute Core Temperature Trajectory Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([mins, mins[::-1]]),
        y=np.concatenate([tcore_pred + tcore_unc, (tcore_pred - tcore_unc)[::-1]]),
        fill='toself',
        fillcolor='rgba(234, 88, 12, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% EKF Confidence Band'
    ))
    fig.add_trace(go.Scatter(
        x=mins, y=tcore_pred,
        mode='lines+markers',
        line=dict(color='#ea580c', width=3),
        name='Predicted Core Temp (°C)'
    ))
    fig.add_hline(y=38.3, line_dash="dash", line_color="#dc2626", annotation_text="Critical Strain Threshold (38.3°C)")
    fig.add_hline(y=37.75, line_dash="dot", line_color="#d97706", annotation_text="Early Warning Rotation Threshold (37.75°C)")
    fig.update_layout(
        title="20-Minute Physiological Heat-Strain Forecast (On-Chip Extended Kalman Filter)",
        xaxis_title="Forecast Horizon (Minutes Ahead)",
        yaxis_title="Estimated Core Body Temperature (°C)",
        yaxis=dict(range=[36.8, 39.2]),
        height=380,
        margin=dict(l=20, r=20, t=45, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# === TAB 2: ANONYMIZED SUPERVISOR & SMART BREAK PLANNER ===
with tab2:
    st.subheader("Anonymized Crew-Level Risk Heatmap & Automated Break Planner")
    st.info("🔒 **Zero-Surveillance Guarantee:** Individual worker names, IDs, and raw heart-rate streams are cryptographically stripped at the edge. Supervisors see only aggregated crew zones.")

    # Dynamic crew table reacting to live slider state for Crew B (Active Field Zone)
    crew_data = pd.DataFrame([
        {
            "Crew Zone": "Zone A — North Steel Structure (18 Workers)",
            "Mean Zone WBGT": "30.4 °C",
            "Crew Strain Status": "GREEN",
            "Active Work/Rest Ratio": "50m Work / 10m Rest",
            "Recommended Action": "Maintain standard hydration schedule"
        },
        {
            "Crew Zone": "Zone B — South Concrete Pour (Live Sensor Node)",
            "Mean Zone WBGT": f"{wbgt_val} °C",
            "Crew Strain Status": risk_level,
            "Active Work/Rest Ratio": "30m Work / 15m Shade" if risk_level == "AMBER" else ("15m Work / 30m Cooling" if risk_level == "RED" else "50m Work / 10m Rest"),
            "Recommended Action": "IMMEDIATE SHADE ROTATION + COOLING STATION" if risk_level == "RED" else ("Schedule 15-min staggered shade break within 10 mins" if risk_level == "AMBER" else "Normal operations")
        },
        {
            "Crew Zone": "Zone C — Logistics & Crane Staging (14 Workers)",
            "Mean Zone WBGT": "31.6 °C",
            "Crew Strain Status": "AMBER",
            "Active Work/Rest Ratio": "35m Work / 15m Shade",
            "Recommended Action": "Deploy electrolyte cart; rotate crane riggers"
        }
    ])

    st.dataframe(crew_data, use_container_width=True, hide_index=True)

    st.markdown("### 📅 Smart Break & Shift Rotation Schedule (Auto-Generated)")
    now = datetime.now()
    schedule_rows = []
    for idx, row in crew_data.iterrows():
        status = row["Crew Strain Status"]
        start_time = now + timedelta(minutes=5 if status == "RED" else (12 if status == "AMBER" else 25))
        duration = "20 mins (Cooling Tent + Medical Check)" if status == "RED" else ("15 mins (Shaded Rest + 500ml Water)" if status == "AMBER" else "10 mins (Standard Hydration)")
        schedule_rows.append({
            "Target Crew": row["Crew Zone"],
            "Trigger Priority": "P0 — URGENT" if status == "RED" else ("P1 — PREVENTIVE" if status == "AMBER" else "P2 — ROUTINE"),
            "Scheduled Break Window": start_time.strftime("%H:%M") + " – " + (start_time + timedelta(minutes=15)).strftime("%H:%M"),
            "Intervention Protocol": duration
        })
    st.table(pd.DataFrame(schedule_rows))

    if risk_level == "RED":
        with st.expander("🚑 View Consent-Gated Clinical Escalation Packet (FHIR R4 Summary)", expanded=True):
            st.json({
                "resourceType": "Observation",
                "status": "preliminary",
                "category": "occupational-heat-strain-alert",
                "zone": "Zone B - South Concrete Pour",
                "anonymizedNodeToken": "PUCK-ZNB-042-EPHEMERAL",
                "computedWBGT_C": wbgt_val,
                "predictedCoreTemp20m_C": tcore_20m,
                "consentVerified": consent_hr,
                "recommendedClinicalProtocol": "Active evaporative cooling, oral/IV hydration assessment, 30-min vitals observation prior to return-to-duty."
            })

# === TAB 3: HARDWARE & PRIVACY ARCHITECTURE ===
with tab3:
    st.subheader("Subtractive Hardware & Edge-to-Cloud Architecture")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 1. Sub-$18 Ruggedized Edge Micro-Puck
        * **MCU:** Espressif ESP32-S3 (Dual-core Xtensa LX7, ultra-low-power BLE 5.0).
        * **Microclimate Sensor:** Bosch BME280 (Ambient Temp, RH%, Barometric Pressure).
        * **Optional Biometric:** Consented optical PPG sensor (processed strictly in volatile SRAM).
        * **Actuators:** Coin ERM haptic motor + waterproof piezo buzzer for high-noise construction zones.
        * **Resilience:** IP65 dust-tight vented enclosure, offline-first inference, 5-day battery life.
        """)
    with c2:
        st.markdown("""
        #### 2. Strict Worker-First Governance
        * **Hardware Privacy Boundary:** Raw heart-rate waveforms are overwritten in SRAM every 10 seconds after updating the Kalman state vector.
        * **Zero Employer Surveillance:** No individual worker biometric history is ever stored or transmitted to employer dashboards.
        * **Abu Dhabi DoH Alignment:** Maps directly to UAE Ministerial Decree No. 401 (Midday Break) and Abu Dhabi Occupational Safety and Health System Framework (OSHAD-SF).
        """)