import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Instamart Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"

# ----------------------------------
# CUSTOM DESIGN
# ----------------------------------


st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --orange: #ff7417;
    --orange-light: #ff9b42;
    --background: #080c14;
    --card: #151d2d;
    --text: #f8fafc;
    --muted: #9aa8bc;
}

.stApp {
    background:
        radial-gradient(ellipse at 80% 0%, #34200f 0%, transparent 35%),
        linear-gradient(135deg, #080c14, #101827);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #080c14);
    border-right: 1px solid #ff741733;
}

/* MAIN HEADINGS */

h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #ffffff, #ff8a35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* KPI CARDS */

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #202b40, #111827);
    border: 1px solid #ff8a3544;
    border-radius: 20px;
    padding: 25px;
    box-shadow:
        0 12px 25px #00000066,
        inset 0 1px 1px #ffffff15;
    transition: all 0.35s ease;
    transform-style: preserve-3d;
    animation: cardAppear 0.8s ease both;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-8px) rotateX(3deg);
    border-color: #ff8a35;
    box-shadow:
        0 20px 35px #ff741722,
        0 0 18px #ff741733;
}

[data-testid="stMetricLabel"] {
    color: #bdc9da !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #ff963d !important;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
}

/* BUTTONS */

.stButton button,
.stDownloadButton button {
    background: linear-gradient(135deg, #ff7417, #ff9b42);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px #ff741744;
}

.stButton button:hover,
.stDownloadButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px #ff741766;
    color: white;
}

/* CHART CONTAINERS */

[data-testid="stPlotlyChart"] {
    background: linear-gradient(145deg, #1a263a, #101827);
    border: 1px solid #ffffff14;
    border-radius: 18px;
    padding: 10px;
    box-shadow: 0 12px 30px #00000055;
    transition: all 0.3s ease;
}

[data-testid="stPlotlyChart"]:hover {
    border-color: #ff8a3577;
    transform: translateY(-4px);
}

/* DATA TABLE */

[data-testid="stDataFrame"] {
    border: 1px solid #ff8a3533;
    border-radius: 15px;
    overflow: hidden;
}

/* NAVIGATION */

[data-testid="stSidebar"] label {
    color: #e5e7eb !important;
    font-weight: 600;
}

/* SEPARATORS */

hr {
    border-color: #ff8a3533;
}

/* ANIMATIONS */

@keyframes cardAppear {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ----------------------------------
# LOAD PROJECT DATA
# ----------------------------------

@st.cache_data
def load_data(filename):
    file_path = OUTPUT_DIR / filename

    if file_path.exists():
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


segments = load_data("customer_segments.csv")
retention = load_data("retention_list.csv")
triage = load_data("triage_results.csv")
policy = load_data("policy_eval_results.csv")
profile = load_data("segment_profile.csv")


# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------

def find_column(df, keywords):
    for col in df.columns:
        if any(word in col.lower() for word in keywords):
            return col
    return None


def show_table(df):
    if df.empty:
        st.info("No data available yet.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def chart_layout(fig):
    fig.update_layout(
        paper_bgcolor="#1d2b42",
        plot_bgcolor="#1d2b42",
        font_color="white",
        margin=dict(l=20, r=20, t=45, b=20)
    )
    return fig


# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:

    st.markdown("# 🛒 Instamart")

    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "👥 Customer Analytics",
            "📊 Lapse Prediction",
            "🎧 Support Assistant",
            "📚 Policy Assistant"
        ]
    )

    st.markdown("---")

    st.caption("VERSION 2.0")
    st.caption("Powered by Python & Machine Learning")


# ----------------------------------
# MAIN DASHBOARD
# ----------------------------------

if page == "🏠 Dashboard":

    st.title("Welcome to Instamart Intelligence 👋")

    st.write(
        "Your central hub for customer insights, retention "
        "analytics and intelligent customer support."
    )

    st.markdown("---")

    # KPI CARDS

    total_customers = len(segments)
    at_risk_customers = len(retention)
    total_tickets = len(triage)

    flagged_tickets = 0

    if not triage.empty:
        status_col = find_column(triage, ["status"])
        if status_col:
            flagged_tickets = triage[status_col].astype(
                str
            ).str.contains("flagged|invalid", case=False).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Customers", f"{total_customers:,}")
    col2.metric("⚠️ At-Risk Customers", f"{at_risk_customers:,}")
    col3.metric("🎫 Support Tickets", f"{total_tickets:,}")
    col4.metric("🚩 Flagged Tickets", f"{flagged_tickets:,}")

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("Customer Segments")

        if not segments.empty:

            segment_col = find_column(
                segments,
                ["segment", "cluster", "group"]
            )

            if segment_col:

                counts = segments[segment_col].astype(
                    str
                ).value_counts().reset_index()

                counts.columns = ["Segment", "Customers"]

                fig = px.pie(
                    counts,
                    names="Segment",
                    values="Customers",
                    hole=0.55,
                    title="Customer Distribution"
                )

                st.plotly_chart(
                    chart_layout(fig),
                    use_container_width=True
                )

            else:
                st.info("Segment column not found in the data.")

        else:
            st.info("Run the project pipeline to generate customer data.")

    with right:

        st.subheader("Customer Retention")

        if not retention.empty:

            fig = px.bar(
                x=["Customers to Retain"],
                y=[len(retention)],
                labels={"x": "", "y": "Number of Customers"},
                title="Retention Opportunity"
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        else:
            st.info("No retention data available.")

    st.markdown("---")

    st.subheader("Recent Customer Records")

    show_table(segments.head(10))


# ----------------------------------
# CUSTOMER ANALYTICS
# ----------------------------------

elif page == "👥 Customer Analytics":

    st.title("👥 Customer Analytics")

    st.write("Understand customer behaviour and segmentation.")

    st.markdown("---")

    if not segments.empty:

        st.subheader("Customer Segment Details")

        show_table(segments)

        segment_col = find_column(
            segments,
            ["segment", "cluster", "group"]
        )

        if segment_col:

            counts = segments[segment_col].astype(
                str
            ).value_counts().reset_index()

            counts.columns = ["Segment", "Customers"]

            fig = px.bar(
                counts,
                x="Segment",
                y="Customers",
                color="Segment",
                title="Customer Segment Comparison"
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

    else:
        st.warning("Customer segmentation data is not available.")


# ----------------------------------
# LAPSE PREDICTION
# ----------------------------------

elif page == "📊 Lapse Prediction":

    st.title("📊 Customer Lapse Prediction")

    st.write(
        "Identify customers who may stop ordering "
        "and discover retention opportunities."
    )

    st.markdown("---")

    st.metric(
        "Customers in Retention List",
        len(retention)
    )

    st.subheader("Retention Recommendations")

    show_table(retention)

    if not retention.empty:

        st.download_button(
            "⬇️ Download Retention List",
            retention.to_csv(index=False),
            file_name="instamart_retention_list.csv",
            mime="text/csv"
        )


# ----------------------------------
# SUPPORT ASSISTANT
# ----------------------------------

elif page == "🎧 Support Assistant":

    st.title("🎧 Customer Support Intelligence")

    st.write(
        "Explore customer support ticket classifications "
        "and their processing results."
    )

    st.markdown("---")

    if not triage.empty:

        st.metric("Total Processed Tickets", len(triage))

        status_col = find_column(triage, ["status"])

        if status_col:

            counts = triage[status_col].astype(
                str
            ).value_counts().reset_index()

            counts.columns = ["Status", "Tickets"]

            fig = px.bar(
                counts,
                x="Status",
                y="Tickets",
                color="Status",
                title="Ticket Processing Status"
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        st.subheader("Ticket Classification Records")

        show_table(triage)

    else:
        st.warning("No support ticket data available.")


# ----------------------------------
# POLICY ASSISTANT
# ----------------------------------

elif page == "📚 Policy Assistant":

    st.title("📚 Policy & Refund Intelligence")

    st.write(
        "Explore policy evaluation results from "
        "the offline retrieval system."
    )

    st.markdown("---")

    if not policy.empty:

        st.metric("Policy Evaluation Records", len(policy))

        st.subheader("Policy Evaluation Results")

        show_table(policy)

    else:
        st.warning("No policy evaluation data available.")

    st.info(
        "Refund processing in this capstone is simulated. "
        "No real payments or refunds are processed."
    )


# ----------------------------------
# FOOTER
# ----------------------------------

st.markdown("---")

st.caption(
    "Instamart Customer Intelligence & Support Copilot | "
    "Built with Python, Streamlit, Plotly & Machine Learning"
)