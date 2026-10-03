"""Interactive carbon-credit portfolio calculator."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import streamlit as st

st.set_page_config(page_title="Carbon Portfolio | Optiver Challenge", page_icon="◉", layout="wide")

import portfolio_model as model

GREEN = "#087f70"
INK = "#182d29"
GOLD = "#ce9d4b"
RED = "#ad5361"
BAND_COLORS = {"Low": GREEN, "Medium": GOLD, "High": RED}
SOURCE_HASH = hashlib.sha256(model.DATA.read_bytes()).hexdigest()
MODEL_HASH = hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()

st.markdown("""
<style>
  .block-container { max-width: 1240px; padding-top: 2.4rem; padding-bottom: 4rem; }
  [data-testid="stHeader"] { background: transparent; }
  h1, h2, h3 { color: #182d29; letter-spacing: -.035em; }
  h1 { font-family: Georgia, serif; font-weight: 400 !important; font-size: 3.3rem !important; line-height: 1.08 !important; }
  h2 { font-size: 1.85rem !important; font-weight: 500 !important; padding-top: .8rem !important; }
  h3 { font-size: 1.15rem !important; }
  p { line-height: 1.65; }
  .eyebrow { color: #087f70; font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 1.4rem; }
  .intro { max-width: 710px; color: #5a6b64; font-size: 1.07rem; margin: .5rem 0 1.6rem; }
  .target { border-left: 2px solid #087f70; padding: .6rem 0 .6rem 1.4rem; margin-top: .8rem; }
  .target strong { font-family: Georgia, serif; font-size: 2.4rem; font-weight: 400; }
  .target span { color: #5a6b64; font-size: .87rem; }
  [data-testid="stForm"] { background: #fff; border: 1px solid #dbe3db; border-radius: 12px; padding: 1.4rem 1.6rem; }
  [data-testid="stMetricLabel"] { color: #5a6b64; font-size: .85rem; }
  [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 500; letter-spacing: -.04em; }
  .step-number { font: 400 1.7rem Georgia,serif; color: #087f70; padding-top: .6rem; border-top: 1px solid #ccd9ce; }
  .footnote { font-size: .82rem; color: #68786e; line-height: 1.6; }
  [data-testid="stTabs"] button { font-size: .94rem; }
  @media (max-width: 700px) { h1 { font-size: 2.4rem !important; } .block-container { padding: 1.4rem; } }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=24)
def calculate(appetite, budget, source_hash, model_hash):
    """Results are keyed by user inputs and source/model content, never read from old exports."""
    result = model.optimize_portfolio(appetite=appetite, budget=budget)
    if result["success"]:
        result["checks"] = model.audit(result)
        result["stats"], result["retained"] = model.evaluate(result)
    return result


@st.cache_data(show_spinner=False, max_entries=24)
def simulations(portfolio, target, source_hash, model_hash):
    u = model.correlated_uniforms(portfolio, 2000, model.SEED + 400)
    t = np.linspace(0, 1, 13)
    paths = np.column_stack([
        (1 - (u < (1 - (1 - portfolio.p_fail.to_numpy()) ** point)) *
         (1 - portfolio.recovery.to_numpy())) @ portfolio.tonnes.to_numpy() for point in t
    ])
    return t, paths


@st.cache_data(show_spinner=False, max_entries=12)
def benchmarks(appetite, budget, source_hash, model_hash):
    rows = []
    for label, diversify, tail in [("Expected delivery only", False, False),
                                    ("Tail protection", False, True),
                                    ("Tail + diversification", True, True)]:
        result = model.optimize_portfolio(appetite=appetite, budget=budget, diversified=diversify, tail=tail)
        if result["success"]:
            stats, _ = model.evaluate(result, seed=model.SEED + 200)
            rows.append({"Strategy": label, "Cost (USD)": result["cost"],
                         "Target success": stats["target_success_probability"],
                         "Worst 5% mean (tCO₂e)": stats["worst_5pct_mean_tonnes"]})
        else:
            rows.append({"Strategy": label, "Status": "Infeasible within budget"})
    return pd.DataFrame(rows)


def style(fig, height=340):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif", color=INK, size=12),
                      margin=dict(l=0, r=12, t=20, b=20), hoverlabel=dict(bgcolor="white"),
                      legend=dict(orientation="h", y=1.15, x=0), colorway=[GREEN, GOLD, RED])
    fig.update_xaxes(gridcolor="#e4eae2", zeroline=False)
    fig.update_yaxes(gridcolor="#e4eae2", zeroline=False)
    return fig


def chart(fig, key=None):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def iso3(country):
    aliases = {"DRC": "COD", "Taiwan": "TWN", "South Korea": "KOR", "North Korea": "PRK"}
    if country in aliases:
        return aliases[country]
    try:
        return pycountry.countries.lookup(country).alpha_3
    except LookupError:
        return None


def spending_bar(p, column):
    group = p.groupby(column, as_index=False).spend_usd.sum().sort_values("spend_usd")
    group["Share of spending"] = group.spend_usd / p.spend_usd.sum()
    fig = px.bar(group, x="Share of spending", y=column, orientation="h",
                 color=column if column == "risk_band" else None,
                 color_discrete_map=BAND_COLORS, color_discrete_sequence=[GREEN],
                 hover_data={"spend_usd": ":$,.0f", "Share of spending": ":.1%"})
    style(fig, max(230, len(group) * 42))
    fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="Share of spending")
    fig.update_xaxes(tickformat=".0%")
    return fig


st.markdown('<div class="eyebrow">OPTIVER CHALLENGE / ADAHACK 2026</div>', unsafe_allow_html=True)
hero, target_note = st.columns([3.2, 1], gap="large")
with hero:
    st.title("A carbon portfolio.\nBuilt to withstand failure.")
    st.markdown('<p class="intro">Find the lowest-cost mix of carbon credits for your risk appetite. '
                'Diversify the exposure. Test what survives.</p>', unsafe_allow_html=True)
with target_note:
    st.markdown('<div class="target"><span>THE CARBON TARGET</span><br><strong>100,000</strong>'
                '<br><span>tonnes CO₂e retained<br>Up to $1m available to spend</span></div>', unsafe_allow_html=True)

with st.form("calculator"):
    st.subheader("Build your portfolio")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        appetite = st.slider("Risk appetite", min_value=0, max_value=100, value=50,
                             help="0 favours AAA–A credits; 100 allows more B–CCC exposure. This changes spending targets, not a guaranteed failure probability.")
        st.caption("0 · More conservative                                      100 · More risk-tolerant")
    with right:
        budget = st.number_input("Maximum budget (USD)", min_value=10_000, max_value=1_000_000,
                                 value=1_000_000, step=10_000, format="%d")
        st.caption("We minimize cost. Any unused budget stays unspent.")
    submitted = st.form_submit_button("Calculate portfolio →", type="primary", width="stretch")

if submitted or "result" not in st.session_state:
    with st.spinner("Finding the least-cost allocation and testing 10,000 fresh scenarios…"):
        st.session_state.result = calculate(appetite, budget, SOURCE_HASH, MODEL_HASH)
result = st.session_state.result

if not result["success"]:
    st.error("No portfolio meets all constraints within this budget. Increase the budget and calculate again.")
    st.caption(f"Attempted budget: ${result['budget']:,.0f} · Risk appetite: {result['appetite']}/100. No constraints were relaxed.")
    with st.expander("Solver details"):
        st.write(result["message"])
else:
    p, stats = result["portfolio"], result["stats"]
    st.markdown("## Your portfolio")
    st.caption(f"Risk appetite {result['appetite']}/100 · Budget ${result['budget']:,.0f} · "
               f"{len(p)} projects across {p.country.nunique()} countries and {p.scope.nunique()} sectors")
    cols = st.columns(4, gap="large")
    cols[0].metric("Acquisition cost", f"${result['cost']:,.0f}")
    cols[1].metric("Modelled target success", f"{stats['target_success_probability']:.2%}")
    cols[2].metric("Worst 5% average", f"{stats['worst_5pct_mean_tonnes']:,.0f} t")
    cols[3].metric("Unspent budget", f"${result['budget'] - result['cost']:,.0f}")
    st.caption(f"{stats['nominal_tonnes']:,.0f} tCO₂e purchased · "
               f"{stats['analytic_expected_tonnes']:,.0f} expected to remain · "
               f"Success sampling interval: {stats['success_ci_low']:.1%}–{stats['success_ci_high']:.1%}. "
               "Estimates depend on the stated failure model; they are not guarantees.")
    if stats["worst_5pct_mean_tonnes"] < result["target"]:
        st.warning("The allocation satisfies the training constraints, but the independent worst-5% average falls below the carbon target. This portfolio has not passed the independent tail check.")

    allocation_tab, stress_tab, holdings_tab = st.tabs(["Geography & allocation", "Survival & stress tests", "Projects & downloads"])
    with allocation_tab:
        geo, breakdown = st.columns([1.5, 1], gap="large")
        with geo:
            st.subheader("Diversification across the globe")
            grouped = p.groupby("country", as_index=False).agg(spend_usd=("spend_usd", "sum"), tonnes=("tonnes", "sum"))
            grouped["iso3"] = grouped.country.map(iso3)
            fig = px.choropleth(grouped.dropna(subset=["iso3"]), locations="iso3", color="spend_usd",
                                hover_name="country", color_continuous_scale=["#d9ebe0", GREEN],
                                hover_data={"spend_usd": ":$,.0f", "tonnes": ":,.0f", "iso3": False},
                                labels={"spend_usd": "Spending (USD)", "tonnes": "Purchased tCO₂e"}, projection="natural earth")
            style(fig, 415)
            fig.update_geos(showframe=False, showcoastlines=False, showland=True, landcolor="#e6eae3", bgcolor="rgba(0,0,0,0)")
            fig.update_coloraxes(colorbar=dict(title="USD", thickness=10, len=.55))
            chart(fig)
            st.caption("Country shading shows acquisition spending. Hover to see purchased tonnes.")
            if grouped.iso3.isna().any():
                st.caption("Not mapped: " + ", ".join(grouped.loc[grouped.iso3.isna(), "country"]) + ". Included in portfolio totals.")
        with breakdown:
            st.subheader("Allocation by risk rating")
            chart(spending_bar(p, "risk_band"))
            st.caption("Low: AAA–A · Medium: BBB, BB and unrated · High: B–CCC")
            st.subheader("Allocation by project sector")
            chart(spending_bar(p, "scope"))
        with st.expander("Check the risk policy and concentration limits"):
            policy = result["policy"].copy()
            policy["actual"] = p.groupby("risk_band").spend_usd.sum().reindex(model.BANDS, fill_value=0) / result["cost"]
            st.dataframe(policy.rename(columns=str.title).style.format("{:.1%}"), width="stretch")
            st.write(f"All {len(result['checks'])} spending, inventory, allocation and training-carbon checks passed.")
            st.caption(f"Effective project count: {stats['effective_projects_spend']:.1f} by spending, "
                       f"{stats['effective_projects_tonnes']:.1f} by tonnes. Effective counts do not imply independent risk.")

    with stress_tab:
        st.subheader("What survives when projects fail?")
        times, paths = simulations(p, result["target"], SOURCE_HASH, MODEL_HASH)
        fig = go.Figure()
        q05, median, q95 = np.quantile(paths, [.05, .5, .95], axis=0)
        fig.add_trace(go.Scatter(x=times, y=q95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=times, y=q05, mode="lines", line=dict(width=0), fill="tonexty",
                                fillcolor="rgba(8,127,112,.15)", name="5th–95th percentile", hoverinfo="skip"))
        for line in paths[:20]:
            fig.add_trace(go.Scatter(x=times, y=line, mode="lines", line=dict(color="rgba(8,127,112,.12)", width=1), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=times, y=median, mode="lines", line=dict(color=GREEN, width=3), name="Median retained carbon"))
        fig.add_hline(y=result["target"], line_color=RED, line_dash="dash", annotation_text="100,000 t target")
        style(fig, 360)
        fig.update_layout(xaxis_title="Fraction of challenge horizon", yaxis_title="Retained carbon (tCO₂e)")
        chart(fig)
        st.caption("2,000 simulated paths. Timing is a constant-hazard interpolation over an unspecified challenge horizon, not historical performance.")
        st.subheader("10,000 independent outcomes")
        fig = go.Figure(go.Histogram(x=result["retained"], nbinsx=45, marker_color=GREEN))
        fig.add_vline(x=result["target"], line_color=RED, line_dash="dash")
        style(fig, 300)
        fig.update_layout(xaxis_title="Retained credits (tCO₂e)", yaxis_title="Simulated outcomes")
        chart(fig)
        if st.button("Compare with simpler portfolios", key="benchmarks"):
            with st.spinner("Solving the benchmark portfolios…"):
                st.session_state.benchmark_result = (result["appetite"], result["budget"], benchmarks(result["appetite"], result["budget"], SOURCE_HASH, MODEL_HASH))
        previous = st.session_state.get("benchmark_result")
        if previous and previous[:2] == (result["appetite"], result["budget"]):
            st.dataframe(previous[2], hide_index=True, width="stretch")
            st.caption("Same risk policy and budget; fresh comparison simulations. Added protection can cost more and buy more carbon, so this is not a pure causal estimate of diversification.")

    with holdings_tab:
        st.subheader("The complete allocation")
        holdings = p.sort_values("spend_usd", ascending=False)
        columns = ["project_name", "country", "scope", "risk_rating", "tonnes", "spend_usd", "weight"]
        shown = holdings[columns].rename(columns={"project_name": "Project", "country": "Country", "scope": "Sector", "risk_rating": "Rating", "tonnes": "Tonnes CO₂e", "spend_usd": "Spending (USD)", "weight": "Weight"})
        st.dataframe(shown.style.format({"Tonnes CO₂e": "{:,.1f}", "Spending (USD)": "${:,.0f}", "Weight": "{:.1%}"}), hide_index=True, width="stretch", height=430)
        payload = {"appetite": result["appetite"], "budget": result["budget"], "target": result["target"], "stats": stats,
                   "source_sha256": SOURCE_HASH, "model_sha256": MODEL_HASH,
                   "assumptions": {"factor_variance_shares": model.FACTOR_SHARES, "global_spending_caps": model.GLOBAL_CAPS,
                                   "within_band_spending_caps": model.SLEEVE_CAPS, "training_scenarios": model.TRAIN_SCENARIOS,
                                   "test_scenarios": model.TEST_SCENARIOS, "tail_alpha": model.TAIL_ALPHA,
                                   "training_cushion": model.SAFETY_MARGIN, "seed": model.SEED}}
        d1, d2 = st.columns(2)
        d1.download_button("Download project allocation · CSV", holdings.to_csv(index=False), "carbon_portfolio.csv", "text/csv", width="stretch")
        d2.download_button("Download metrics & assumptions · JSON", json.dumps(payload, indent=2, default=float), "portfolio_summary.json", "application/json", width="stretch")

st.divider()
st.markdown('<div class="eyebrow">THE METHODOLOGY</div>', unsafe_allow_html=True)
st.header("A cheaper portfolio is only useful if it delivers.")
st.write("The objective is simple: minimize acquisition cost while preserving the carbon target. "
         "The optimizer considers project inventory, rating-based risk, shared exposures and recovery after failure.")
steps = [
    ("01", "Start with the evidence", "4,355 registry projects, grouped into ten existing sectors. Synthetic ratings become failure probabilities; previous reversals and buffer pools adjust the potential losses."),
    ("02", "Set the allocation", "Your appetite sets spending ranges across low, medium and high risk. Country, sector and developer caps apply within each band as well as across the portfolio."),
    ("03", "Protect the carbon target", "A linear program finds the least-cost allocation. The worst 5% of 1,024 training scenarios must average at least 112,000 retained tonnes, adding a 12% cushion."),
    ("04", "Test on fresh scenarios", "10,000 new correlated failure scenarios measure how often the portfolio retains 100,000 tonnes. The independent outcomes show how much carbon remains when projects fail together."),
]
for col, (number, title, body) in zip(st.columns(4, gap="large"), steps):
    with col:
        st.markdown(f'<div class="step-number">{number}</div>', unsafe_allow_html=True)
        st.subheader(title)
        st.write(body)

with st.expander("The mathematics, constraints and assumptions"):
    st.markdown("We choose purchased tonnes $q_i$ to minimize $\\sum_i c_i q_i$. In scenario $s$, retained carbon is:")
    st.latex(r"D_s = \sum_i q_i\,[1-(1-r_i)F_{si}]")
    st.write("Here c is price per tonne, r is the recovered fraction and F is the project failure indicator. "
             "The lower-tail CVaR constraint protects the mean of the worst 5% of training outcomes:")
    st.latex(r"\eta - \frac{1}{0.05S}\sum_s z_s \geq 1.12T,\quad z_s\geq\eta-D_s,\quad z_s\geq0")
    left, right = st.columns(2)
    with left:
        st.markdown("**Allocation rules**")
        st.write("Appetite 0: 80% Low / 20% Medium / 0% High. Appetite 100: 20% Low / 30% Medium / 50% High. "
                 "Targets interpolate linearly with ±5 percentage points of flexibility, except a zero High target permits no High spending.")
        st.write("Global spending caps: project 8%, country 35%, sector 60%, developer 15%, registry 70%. "
                 "Within each risk band: project 20%, country 55%, sector 75%, developer 35%.")
        st.caption("These percentages apply to money actually spent. Policy choices are explicit assumptions, not estimated optimal weights.")
    with right:
        st.markdown("**Failure rules**")
        st.write("AAA 1% · AA 2% · A 4% · BBB 7% · BB 12% · B 20% · CCC 35% · Unrated 15%. "
                 "Recorded reversal: multiply probability by 1.5, capped at 100%. Buffer pool: recover 50% of a failed credit's value in carbon tonnes.")
        st.write("The Gaussian dependence model shares latent variance across global (5%), country (15%), sector (5%), "
                 "project type (10%), developer (15%) and registry (5%) factors. The remaining 45% is project-specific.")
        st.caption("These variance shares are not observed default correlations. Failure probabilities refer to an unspecified challenge horizon.")
    st.write("The 12% carbon cushion is calibrated on validation scenarios, separate from the final test scenarios. "
             "A higher appetite does not necessarily produce cheaper portfolios or a monotonic realised failure probability.")

with st.expander("Explore the source data"):
    st.caption("Organiser snapshot · Berkeley registry projects · synthetic prices and risk ratings")
    a, b, c, d = st.columns(4)
    a.metric("Projects", f"{len(model.df):,}")
    b.metric("Country / location labels", model.df.country.nunique())
    c.metric("Project sectors", model.df.scope.nunique())
    d.metric("Unrated projects", int(model.df.risk_rating.eq("Unrated").sum()))
    left, right = st.columns(2, gap="large")
    with left:
        category = st.selectbox("Explore a project characteristic", ["country", "scope", "risk_rating", "registry", "project_type", "region", "developer", "status", "reduction_or_removal", "has_buffer_pool", "had_reversal"], format_func=lambda x: x.replace("_", " ").title())
        counts = model.df[category].value_counts().head(12).sort_values()
        fig = px.bar(x=counts.values, y=counts.index, orientation="h", color_discrete_sequence=[GREEN])
        style(fig, 420); fig.update_layout(xaxis_title="Number of projects", yaxis_title=None)
        chart(fig)
        st.caption("Top 12 categories by project count. Counts describe the available universe, not the selected allocation.")
    with right:
        numeric = st.selectbox("Explore a numeric field", ["price_usd_per_t", "available_tonnes", "vintage_year"], format_func=lambda x: x.replace("_", " ").title())
        log = numeric != "vintage_year"
        values = np.log10(model.df[numeric]) if log else model.df[numeric]
        labels = {"price_usd_per_t": "log₁₀ price (USD / tCO₂e)", "available_tonnes": "log₁₀ available credits (tCO₂e)", "vintage_year": "Vintage year"}
        fig = px.histogram(x=values, nbins=35, color_discrete_sequence=[GREEN])
        style(fig, 420); fig.update_layout(xaxis_title=labels[numeric], yaxis_title="Number of projects")
        chart(fig)
    st.dataframe(model.raw, hide_index=True, width="stretch", height=300)

with st.expander("What the results do—and do not—tell us"):
    st.write("These are carbon-credit purchases with delivery risk. No coupons, redemption values, resale prices or return histories are supplied. "
             "Monte Carlo results are scenario tests, not historical backtests. A carbon purchase does not itself reduce the buyer's gross emissions.")
    st.write("Diversification reduces concentration but cannot eliminate common failures. Modelled success and sampling intervals depend on assumed probabilities and correlations; "
             "they do not certify additionality, permanence or future outcomes. The source's project-sector field is not corporate GHG Scope 1/2/3.")
    st.write("The target is 100,000 tonnes CO₂e. No financial return is estimated because the dataset contains no credit-price history or project cash flows.")

st.divider()
st.markdown("**Source & attribution**")
st.caption("Project snapshot retrieved 3 October 2026. Quartson et al. (2026), Voluntary Registry Offsets Database v2026-06, Berkeley Carbon Trading Project, UC Berkeley (CC BY 4.0). Prices and ratings were added by the hackathon organiser.")
st.markdown("[Organiser database](https://docs.google.com/spreadsheets/d/1d3YKqXgVyYUkYA9aYVlmE6ryc5MrrasY/edit) · "
            "[Berkeley registry database](https://gspp.berkeley.edu/berkeley-carbon-trading-project/offsets-database)")
