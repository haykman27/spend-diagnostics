# Procurement Diagnostics — Final (Auto FX → EUR)
# Full app with robust quantity parsing, hardened spend, fixed donut,
# and clear top-suppliers bar chart.

import io, re
import numpy as np
import pandas as pd
import streamlit as st

# Charts
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False

from rapidfuzz import process, fuzz

st.set_page_config(page_title="Procurement Diagnostics — Final", layout="wide")
st.title("Procurement Diagnostics — Final (Auto FX → EUR)")

st.caption(
    "Upload your Excel spend cube. The app auto-detects columns, parses numbers, converts to EUR "
    "(latest ECB FX), and shows diagnostics in € k. Use the **sidebar** for data setup and to switch "
    "between the **Dashboard** and **Deep Dives**."
)

# ---------- Styles ----------
st.markdown(
    """
    <style>
      .kpi-card { background:#fff;border:1px solid #EEE;border-radius:14px;padding:16px 18px;
                  box-shadow:0 1px 3px rgba(0,0,0,.04); }
      .kpi-title { font-size:.95rem;color:#6b7280;margin-bottom:6px;}
      .kpi-value { font-size:2rem;font-weight:700;letter-spacing:-.02rem; }
      .pill{display:inline-block;background:#f1f5f9;border:1px solid #e5e7eb;padding:6px 10px;border-radius:999px;margin-right:8px;}
      .mt-16{margin-top:16px;} .mt-24{margin-top:24px;}
      [data-testid="stSidebar"]{min-width:360px;max-width:400px;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE = "EUR"

# ---------- Helpers ----------
def normalize_headers(cols):
    return [re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()) for c in cols]

def parse_number_robust(x):
    """Parse price/quantity strings like '1 234,56', '1,234', '1 234 pcs' → float."""
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x)
    # Strip units/letters, keep digits, comma, dot, minus
    s = re.sub(r"[^\d,\.\-]", "", s)
    # European vs. US separators
    if "," in s and "." in s:
        # Assume comma is thousands
        s = s.replace(",", "")
    elif "," in s and s.count(",")==1 and len(s.split(",")[-1]) in (2,3):
        # '1,23' or '1,234' → decimal comma
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

def ensure_numeric_spend(series: pd.Series) -> pd.Series:
    """Force spend to clean float; if boolean, convert; coerce object; fillna 0."""
    if series.dtype == bool:
        s = series.astype(float)
    elif series.dtype.kind in ("i","u","f"):
        s = series.astype(float)
    else:
        s = pd.to_numeric(series, errors="coerce")
    return s.fillna(0.0).astype(float)

CURRENCY_SYMBOL_MAP = {"€":"EUR","$":"USD","£":"GBP","¥":"JPY","₩":"KRW","₹":"INR","₺":"TRY","R$":"BRL","S$":"SGD"}
ISO_3 = {
    "EUR","USD","GBP","JPY","CNY","CHF","SEK","NOK","DKK","PLN","HUF","CZK","RON","AUD","NZD","CAD","MXN",
    "BRL","ZAR","AED","SAR","HKD","SGD","INR","TRY","KRW","TWD","THB","PHP","ILS","NGN","VND","RUB"
}
def detect_iso_from_text(text):
    if text is None or (isinstance(text,float) and np.isnan(text)): return None
    s = str(text).upper().strip()
    alias = {"RMB":"CNY","YUAN":"CNY","CN¥":"CNY","ZŁ":"PLN","ZL":"PLN","KČ":"CZK","LEI":"RON","RUR":"RUB","РУБ":"RUB"}
    m = re.search(r"\b([A-Z]{3})\b", s)
    if m:
        c = m.group(1)
        if c in ISO_3: return c
        if c in alias: return alias[c]
    for k,v in alias.items():
        if k in s: return v
    for sym in sorted(CURRENCY_SYMBOL_MAP, key=len, reverse=True):
        if sym in s: return CURRENCY_SYMBOL_MAP[sym]
    return None

@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_latest_ecb():
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"
    fx_wide = pd.read_csv(url)
    fx_wide.rename(columns={"Date":"date"}, inplace=True)
    fx_wide["date"] = pd.to_datetime(fx_wide["date"], errors="coerce")
    latest = fx_wide["date"].max()
    row = fx_wide.loc[fx_wide["date"]==latest].melt(id_vars=["date"], var_name="currency", value_name="eur_to_cur")
    row["currency"] = row["currency"].str.upper().str.strip()
    row["eur_to_cur"] = pd.to_numeric(row["eur_to_cur"], errors="coerce")
    row = row.dropna(subset=["eur_to_cur"])
    row["rate_to_eur"] = 1.0 / row["eur_to_cur"]
    eur = pd.DataFrame([{"currency":"EUR","rate_to_eur":1.0}])
    fx = pd.concat([row[["currency","rate_to_eur"]], eur]).drop_duplicates(subset=["currency"])
    return fx.reset_index(drop=True)

def apply_fx_latest(df, fx):
    df["currency_iso"] = df["currency"].astype(str).apply(lambda x: detect_iso_from_text(x) or BASE)
    df = df.merge(fx, left_on="currency_iso", right_on="currency", how="left")
    df["rate_to_eur"] = df["rate_to_eur"].fillna(1.0)
    return df

# -------- Column mapping ----------
TARGETS = {
    "category": ["category","category name","commodity","spend category","material group","item family","item family group"],
    "supplier": ["supplier","supplier name","vendor","vendor name","seller","payee"],
    "currency": ["currency","ccy","currency code","iso currency","transaction currency","base curr"],
    "unit_price": ["unit price","unit cost","net price","unit po price","unit price (base curr)","unit price (global curr)"],
    "quantity": ["quantity","qty","order qty","qty ordered","units","volume","po qty","po quantity","ordered qty"],
    "amount": ["amount","purchase amount","line amount","total value","net value","extended price"]
}
def suggest_columns(df):
    cols = df.columns.tolist()
    norm = normalize_headers(cols)
    back = {n:o for n,o in zip(norm, cols)}
    out = {}
    for k, syns in TARGETS.items():
        best, score = None, -1
        for s in syns:
            m = process.extractOne(s, norm, scorer=fuzz.token_sort_ratio)
            if m and m[1] > score:
                best, score = m[0], m[1]
        if best: out[k] = back[best]
    return out

# -------- Spend detection ----------
SPEND_NAME_CUES = ["purchase amount","po amount","line total","total value","net value","extended price","base curr","global curr","reporting curr"]
def detect_spend_column(df):
    hits = [c for c in df.columns if any(k in c.lower() for k in SPEND_NAME_CUES)]
    if not hits: return None
    if len(hits)==1: return hits[0]
    med = {c: pd.to_numeric(df[c].apply(parse_number_robust), errors="coerce").median(skipna=True) for c in hits}
    return max(med, key=med.get)

# -------- Savings & VAVE ----------
SAVINGS = {"stamp":"(0.10,0.15)","tube":"(0.08,0.14)","plast":"(0.08,0.18)","steel":"(0.05,0.12)","logist":"(0.08,0.15)"}
def savings_range(cat_val):
    c = "" if (cat_val is None or (isinstance(cat_val,float) and np.isnan(cat_val))) else str(cat_val).lower()
    if "stamp" in c: return (0.10,0.15)
    if "tube" in c:  return (0.08,0.14)
    if "plast" in c: return (0.08,0.18)
    if "steel" in c: return (0.05,0.12)
    if "log" in c:  return (0.08,0.15)
    return (0.05,0.10)

def fmt_k(s: pd.Series) -> pd.Series:
    return (s/1_000.0).round(0)

# -------- Optional item col ----------
def detect_item_col(df):
    cues=["part","material","item","description","desc","sku"]
    hits=[c for c in df.columns if any(k in c.lower() for k in cues)]
    return sorted(hits, key=lambda x:len(x), reverse=True)[0] if hits else None

# ===== Session state =====
if "dash_cat_focus" not in st.session_state: st.session_state.dash_cat_focus=None
if "dash_sup_focus" not in st.session_state: st.session_state.dash_sup_focus=None

# ================= Sidebar upload =================
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])

if not uploaded:
    st.info("Use the **sidebar** to upload an Excel file.")
    st.stop()

raw = pd.read_excel(uploaded)
raw.columns = [str(c) for c in raw.columns]

# ---- Column mapping (sidebar) ----
mapping = suggest_columns(raw)
with st.sidebar:
    st.subheader("Column mapping")
    if "category" not in mapping:
        st.warning("Category column not auto-detected.")
        mapping["category"] = st.selectbox("Category / Family / Group", options=raw.columns, key="cat_pick")
    if "supplier" not in mapping:
        st.warning("Supplier column not auto-detected.")
        mapping["supplier"] = st.selectbox("Supplier / Vendor", options=raw.columns, key="sup_pick")
    if "currency" not in mapping:
        st.warning("Currency column not auto-detected.")
        mapping["currency"] = st.selectbox("Currency", options=raw.columns, key="cur_pick")
    if "unit_price" not in mapping:
        st.warning("Unit Price column not auto-detected.")
        mapping["unit_price"] = st.selectbox("Unit Price (per item)", options=raw.columns, key="up_pick")
    if "quantity" not in mapping:
        st.warning("Quantity column not auto-detected.")
        mapping["quantity"] = st.selectbox("Quantity", options=raw.columns, key="qty_pick")

df = raw.rename(columns={v:k for k,v in mapping.items() if v}).copy()

# ---- Spend column detection
spend_col = detect_spend_column(raw)
with st.sidebar:
    if spend_col:
        st.success(f"Detected **'{spend_col}'** as spend column.")
    else:
        st.warning("No clear spend column found; you can use Unit×Qty×FX instead.")

# ---- Parse numerics (robust for both price & qty)
if "unit_price" in df.columns: df["unit_price"] = df["unit_price"].apply(parse_number_robust)
if "quantity"   in df.columns: df["quantity"]   = df["quantity"].apply(parse_number_robust)

# ---- FX
fx = load_latest_ecb()
df["currency"] = df.get("currency","")
df = apply_fx_latest(df, fx)

# ---- Spend modes
def _is_global_header(c): return any(k in c.lower() for k in ["base curr","base currency","global curr","reporting curr"])
def _find_base_ccy_col(df_raw, spend_col_name):
    cands=[c for c in df_raw.columns if c!=spend_col_name and any(k in c.lower() for k in
        ["base curr code","base currency code","base curr","base currency","reporting curr","global curr"])]
    return cands[0] if cands else None

# Unit price EUR for diagnostics
df["unit_price_eur"] = pd.to_numeric(df.get("unit_price"), errors="coerce") * df["rate_to_eur"]

# 1) Detected spend → EUR
spend_detected = pd.Series(np.nan, index=df.index)
if spend_col:
    spend_vals = raw[spend_col].apply(parse_number_robust)
    if _is_global_header(spend_col):
        base_col = _find_base_ccy_col(raw, spend_col)
        if base_col:
            base_iso = raw[base_col].astype(str).apply(detect_iso_from_text).fillna("EUR")
            fx_map  = dict(zip(fx["currency"], fx["rate_to_eur"]))
            base_rt = base_iso.map(lambda c: fx_map.get(c,1.0))
            spend_detected = pd.to_numeric(spend_vals, errors="coerce") * base_rt
        else:
            spend_detected = pd.to_numeric(spend_vals, errors="coerce")
    else:
        spend_detected = pd.to_numeric(spend_vals, errors="coerce") * df["rate_to_eur"]

# 2) Unit×Qty×FX
spend_calc = (
    pd.to_numeric(df.get("unit_price"), errors="coerce") *
    pd.to_numeric(df.get("quantity"), errors="coerce") *
    df["rate_to_eur"]
)

# Sidebar mode
with st.sidebar:
    st.subheader("Spend source")
    mode = st.radio("Choose the spend source:",
                    ["Use detected spend column","Use Unit×Qty×FX","Auto-validate"],
                    index=2)

# Resolve mode (with guard if calc all zeros)
def _mostly_zero(x): 
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nansum(x)) == 0.0

if mode == "Use detected spend column":
    df["_spend_eur"] = spend_detected.fillna(0.0)
elif mode == "Use Unit×Qty×FX":
    df["_spend_eur"] = spend_calc.fillna(0.0)
else:
    use_calc = False
    if spend_detected.notna().sum() >= 50 and spend_calc.notna().sum() >= 50:
        m = spend_detected.notna() & spend_calc.notna()
        a, b = spend_detected[m].values, spend_calc[m].values
        corr = np.corrcoef(a,b)[0,1] if len(a)>10 else np.nan
        total_detected = np.nansum(spend_detected)
        total_calc     = np.nansum(spend_calc)
        diff_ratio     = abs(total_detected-total_calc)/max(1.0,total_calc)
        if (np.isnan(corr) or corr < 0.85) or (diff_ratio>0.30):
            use_calc = True
    else:
        use_calc = spend_detected.notna().sum() < spend_calc.notna().sum()
    if _mostly_zero(spend_calc):   # don’t pick a dead calc
        use_calc = False
    if use_calc:
        with st.sidebar:
            st.warning("Auto-validate switched to **Unit×Qty×FX** because the detected spend column was inconsistent.")
        df["_spend_eur"] = spend_calc.fillna(0.0)
    else:
        df["_spend_eur"] = spend_detected.fillna(0.0)

# Enforce clean float spend
df["_spend_eur"] = ensure_numeric_spend(df["_spend_eur"])

with st.sidebar:
    st.caption(
        f"Detected total: € {np.nansum(spend_detected):,.0f} • "
        f"Unit×Qty×FX total: € {np.nansum(spend_calc):,.0f}"
    )

# ---------- Aggregates ----------
cat = (df.groupby("category", dropna=False)
       .agg(spend_eur=("_spend_eur","sum"), lines=("category","count"),
            suppliers=("supplier", pd.Series.nunique))
       .reset_index()
       .rename(columns={"category":"Category","lines":"# PO Lines","suppliers":"# Suppliers"}))
cat["Spend (€ k)"] = fmt_k(cat["spend_eur"])

sup_tot = (df.groupby("supplier", dropna=False)
           .agg(spend_eur=("_spend_eur","sum"), lines=("supplier","count"),
                categories=("category", pd.Series.nunique))
           .reset_index()
           .rename(columns={"supplier":"Supplier"}))
sup_tot["Spend (€ k)"] = fmt_k(sup_tot["spend_eur"])

sup_cat = (df.groupby(["supplier","category"], dropna=False)["_spend_eur"]
             .sum().reset_index()
             .rename(columns={"supplier":"Supplier","category":"Category","_spend_eur":"Spend_EUR"}))

# ---------- Navigation ----------
page = st.sidebar.radio("Navigation", ["Dashboard","Deep Dives"], index=0)

# =================== DASHBOARD ===================
if page == "Dashboard":
    st.subheader("Dashboard")
    total_spend = float(df["_spend_eur"].sum())
    total_lines = int(len(df))
    total_suppliers = int(df["supplier"].nunique())
    total_categories = int(df["category"].nunique())

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Spend</div><div class="kpi-value">€ {total_spend/1_000_000:,.1f} M</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Suppliers</div><div class="kpi-value">{total_suppliers:,}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Categories</div><div class="kpi-value">{total_categories:,}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-card"><div class="kpi-title">PO Lines</div><div class="kpi-value">{total_lines:,}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="mt-16"></div>', unsafe_allow_html=True)
    opts = st.columns([1,1])
    with opts[0]:
        top_cat_n = st.slider("How many categories in donut", 5, 15, 10, step=1)
    with opts[1]:
        sup_n = st.slider("Top suppliers in bar", 10, 30, 20, step=5)

    # Current filters
    pill = []
    if st.session_state.dash_cat_focus: pill.append(f'<span class="pill">Category: {st.session_state.dash_cat_focus}</span>')
    if st.session_state.dash_sup_focus: pill.append(f'<span class="pill">Supplier: {st.session_state.dash_sup_focus}</span>')
    if pill:
        st.markdown("Active selection: " + " ".join(pill), unsafe_allow_html=True)
        if st.button("Clear selections"): 
            st.session_state.dash_cat_focus=None
            st.session_state.dash_sup_focus=None

    # Data subset for bar
    df_for_bar = df[df["category"]==st.session_state.dash_cat_focus] if st.session_state.dash_cat_focus else df
    sup_tot_local = (df_for_bar.groupby("supplier", dropna=False)
                     .agg(spend_eur=("_spend_eur","sum"), lines=("supplier","count"), categories=("category", pd.Series.nunique))
                     .reset_index().rename(columns={"supplier":"Supplier"})
                     .sort_values("spend_eur", ascending=False))

    # ---- Charts
    donut_col, bar_col = st.columns([1.1, 1.4])

    # DONUT
    with donut_col:
        st.markdown("**Spend by Category**")
        donut_raw = (df.groupby("category", dropna=False)["_spend_eur"].sum()
                       .reset_index().rename(columns={"category":"Category","_spend_eur":"spend_eur"}))
        donut_raw["spend_eur"] = ensure_numeric_spend(donut_raw["spend_eur"])
        donut_raw = donut_raw[donut_raw["spend_eur"]>0].sort_values("spend_eur", ascending=False)

        # Degenerate block
        if donut_raw.empty or donut_raw["spend_eur"].nunique(dropna=True)<=1:
            st.error(
                "Category spend values look degenerate (all equal/zero). "
                "Most often this means Quantity was parsed as text. The app now parses Qty robustly—"
                "but if your file has special formats, please check the Qty column."
            )
            st.dataframe(donut_raw.rename(columns={"spend_eur":"Spend (EUR)"}), use_container_width=True)
        else:
            topN = donut_raw.head(top_cat_n)
            other_val = float(donut_raw["spend_eur"].iloc[top_cat_n:].sum())
            donut_df = pd.concat([topN, pd.DataFrame([{"Category":"Other","spend_eur":other_val}]) if other_val>0 else pd.DataFrame()], ignore_index=True)
            if PLOTLY_AVAILABLE:
                fig = px.pie(donut_df, names="Category", values="spend_eur",
                             hole=.45, color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=120),
                                  legend=dict(orientation="h", y=-0.15))
                if PLOTLY_EVENTS_AVAILABLE:
                    res = plotly_events(fig, select_event=True, click_event=True, hover_event=False, override_height=520)
                    st.caption("Tip: Click a wedge to drill into that category.")
                    if res:
                        lab = res[0].get("label")
                        if lab and lab!="Other":
                            st.session_state.dash_cat_focus = lab
                            st.session_state.dash_sup_focus = None
                else:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(donut_df, use_container_width=True)

    # BARS
    with bar_col:
        st.markdown("**Top Suppliers by Spend**")
        top_sup = sup_tot_local.head(sup_n).copy()
        if PLOTLY_AVAILABLE and not top_sup.empty:
            # Show in € M, simple color, generous right margin for labels
            top_sup["Spend_M"] = top_sup["spend_eur"] / 1_000_000.0
            fig2 = px.bar(
                top_sup, x="Spend_M", y="Supplier", orientation="h",
                text=top_sup["Spend_M"].map(lambda v: f"€ {v:,.1f} M"),
                labels={"Spend_M":"Spend (€ M)"},
                color_discrete_sequence=["#14b8a6"],  # teal
            )
            fig2.update_traces(textposition="outside", cliponaxis=False)
            fig2.update_layout(
                height=520,
                margin=dict(l=10, r=220, t=10, b=10),
                yaxis=dict(categoryorder="total ascending", automargin=True, ticksuffix="  "),
                xaxis=dict(title="", showgrid=True, zeroline=False),
            )
            if PLOTLY_EVENTS_AVAILABLE:
                res2 = plotly_events(fig2, select_event=True, click_event=True, hover_event=False, override_height=520)
                st.caption("Tip: Click a supplier bar to focus the mix below.")
                if res2:
                    sup_label = res2[0].get("y")
                    if not sup_label:
                        sup_label = top_sup.iloc[res2[0].get("pointIndex", 0)]["Supplier"]
                    st.session_state.dash_sup_focus = sup_label
            else:
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No positive supplier spend to plot yet.")

    st.markdown('<div class="mt-24"></div>', unsafe_allow_html=True)

    # ---------------- Supplier × Category Mix ----------------
    st.markdown("**Supplier × Category Mix**")
    top_list = ([st.session_state.dash_sup_focus] if st.session_state.dash_sup_focus in sup_tot_local["Supplier"].values else []
                ) if st.session_state.dash_sup_focus else top_sup["Supplier"].tolist()

    sc = sup_cat[sup_cat["Supplier"].isin(top_list)].copy()
    if st.session_state.dash_cat_focus:
        sc = sc[sc["Category"] == st.session_state.dash_cat_focus]
    sc["Category_filt"] = sc["Category"]

    view = st.radio("View", ["% mix","EUR"], horizontal=True, index=0)
    piv = sc.groupby(["Supplier","Category_filt"])["Spend_EUR"].sum().reset_index()
    if view == "% mix":
        total_by_sup = piv.groupby("Supplier")["Spend_EUR"].transform("sum")
        piv["value"] = np.where(total_by_sup>0, piv["Spend_EUR"]/total_by_sup, 0.0)
        value_label="Share (%)"; hover_tpl="%{y} • %{legendgroup}<br>%{x:.1%}"
        xaxis_cfg=dict(tickformat=".0%", range=[0,1])
    else:
        piv["value"]=piv["Spend_EUR"]; value_label="Spend (EUR)"; hover_tpl="%{y} • %{legendgroup}<br>€ %{x:,.0f}"
        xaxis_cfg=dict()

    if PLOTLY_AVAILABLE and not piv.empty:
        order_y = top_list[::-1]
        fig3 = px.bar(piv, x="value", y="Supplier", color="Category_filt",
                      orientation="h", labels={"value":value_label,"Category_filt":"Category"},
                      color_discrete_sequence=px.colors.qualitative.Set3)
        fig3.update_traces(hovertemplate=hover_tpl)
        fig3.update_layout(height=550, barmode="stack", margin=dict(l=10,r=10,t=60,b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0),
                           xaxis=xaxis_cfg, yaxis=dict(automargin=True, ticksuffix="  "))
        fig3.update_yaxes(categoryorder="array", categoryarray=order_y)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No data to plot for the current mix selection.")

# =================== DEEP DIVES (unchanged from last version except minor names) ===================
else:
    st.subheader("Deep Dives")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Category Overview", "Supplier Drill-Down", "VAVE Ideas",
        "Consistency Check", "Outliers & Opportunities", "Price vs Volume"
    ])

    with tab1:
        rngs = [savings_range(c) for c in cat["Category"]]
        cat["Savings Range (%)"] = [f"{int(a*100)}–{int(b*100)}" for a,b in rngs]
        cat["Potential Min (€ k)"] = (cat["spend_eur"] * [a for a,b in rngs] / 1_000).round(0)
        cat["Potential Max (€ k)"] = (cat["spend_eur"] * [b for a,b in rngs] / 1_000).round(0)
        st.dataframe(
            cat[["Category","Spend (€ k)","Savings Range (%)","Potential Min (€ k)","Potential Max (€ k)","# PO Lines","# Suppliers"]],
            use_container_width=True,
            column_config={
                "Spend (€ k)": st.column_config.NumberColumn(format="€ %d k"),
                "Potential Min (€ k)": st.column_config.NumberColumn(format="€ %d k"),
                "Potential Max (€ k)": st.column_config.NumberColumn(format="€ %d k"),
            }
        )

    with tab2:
        chosen_cat = st.selectbox("Choose Category:", options=cat["Category"], key="dd_cat")
        sup = (df[df["category"]==chosen_cat].groupby("supplier", dropna=False)
               .agg(spend_eur=("_spend_eur","sum")).reset_index()
               .assign(**{"Spend (€ k)":lambda x: fmt_k(x["spend_eur"])})
               .rename(columns={"supplier":"Supplier"}).sort_values("Spend (€ k)", ascending=False))
        st.dataframe(sup[["Supplier","Spend (€ k)"]], use_container_width=True,
                     column_config={"Spend (€ k)": st.column_config.NumberColumn(format="€ %d k")})

    with tab3:
        vave_map = {
            "stamp": ["Standardize steel grades","Improve sheet nesting","Relax tolerances","Bundle volumes"],
            "tube": ["Standardize diameters","Use HF-welded","Loosen length tolerance"],
            "plast": ["Consolidate resins","Family molds","Standard inserts"],
            "steel": ["Standardize specs","Index-linked pricing"],
            "log": ["Optimize mode mix","Increase load factor"],
        }
        def vave_for(c):
            s = "" if pd.isna(c) else str(c).lower()
            for k,v in vave_map.items():
                if k in s: return v
            return ["Standardize specs","Consolidate variants","Bundle volumes"]
        st.dataframe(pd.DataFrame([{"Category":c, "Example VAVE Ideas":" • ".join(vave_for(c))} for c in cat["Category"]]),
                     use_container_width=True)

    with tab4:
        dbg = df[["supplier","category","_spend_eur"]].copy()
        dbg["calc_spend"] = (pd.to_numeric(df.get("unit_price"), errors="coerce") *
                             pd.to_numeric(df.get("quantity"), errors="coerce") *
                             df["rate_to_eur"])
        with np.errstate(divide="ignore", invalid="ignore"):
            dbg["consistency_ratio"] = np.where(dbg["calc_spend"]>0, dbg["_spend_eur"]/dbg["calc_spend"], np.nan)
        dbg["Spend (€ k)"] = fmt_k(dbg["_spend_eur"])
        dbg["Calc (€ k)"]  = fmt_k(dbg["calc_spend"])
        st.dataframe(dbg[["supplier","category","Spend (€ k)","Calc (€ k)","consistency_ratio"]].head(300),
                     use_container_width=True)

    with tab5:
        col_a, col_b, col_c = st.columns([1,1,1])
        with col_a:  premium_threshold = st.slider("Min premium over baseline (%)", 5, 60, 20, step=1)
        with col_b:  min_line_opportunity = st.slider("Min line opportunity (€)", 0, 20000, 1000, step=500)
        with col_c:  baseline_method = st.selectbox("Baseline method", ["Median (P50)", "Trimmed mean (10%–90%)"])

        df_bp = df.copy()
        if "unit_price_eur" not in df_bp.columns:
            df_bp["unit_price_eur"] = pd.to_numeric(df_bp.get("unit_price"), errors="coerce") * df_bp["rate_to_eur"]
        def _baseline(series):
            s = pd.to_numeric(series, errors="coerce").dropna()
            if s.empty: return np.nan
            if baseline_method.startswith("Median"): return float(np.median(s))
            lo,hi = np.percentile(s,[10,90]); s=s[(s>=lo)&(s<=hi)]
            return float(s.mean()) if len(s) else np.nan
        baselines = df_bp.groupby("category")["unit_price_eur"].apply(_baseline).rename("baseline_price_eur").reset_index()
        df_bp = df_bp.merge(baselines, on="category", how="left")
        df_bp["premium_pct"] = (df_bp["unit_price_eur"]-df_bp["baseline_price_eur"])/df_bp["baseline_price_eur"]
        df_bp["line_opportunity_eur"] = ((df_bp["unit_price_eur"]-df_bp["baseline_price_eur"]).clip(lower=0) *
                                         pd.to_numeric(df_bp.get("quantity"), errors="coerce")).fillna(0.0)
        outliers = df_bp[(df_bp["premium_pct"]>=premium_threshold/100.0) &
                         (df_bp["line_opportunity_eur"]>=min_line_opportunity)].copy()
        item_col = detect_item_col(raw)
        display_cols = [item_col] if item_col and item_col in outliers.columns else []
        outliers_display = outliers.assign(**{
            "Premium (%)": (outliers["premium_pct"]*100).round(1),
            "Unit Price (EUR)": outliers["unit_price_eur"].round(4),
            "Baseline (EUR)": outliers["baseline_price_eur"].round(4),
            "Opportunity (€ k)": (outliers["line_opportunity_eur"]/1000).round(0),
        })
        show_cols = ["category","supplier"] + display_cols + ["Unit Price (EUR)","Baseline (EUR)","Premium (%)","quantity","Opportunity (€ k)"]
        st.dataframe(outliers_display[show_cols].sort_values("Opportunity (€ k)", ascending=False), use_container_width=True)

    with tab6:
        st.markdown("**Price vs Volume Scatter**")
        if not PLOTLY_AVAILABLE:
            st.warning("Add `plotly>=5.24` to requirements.txt to enable this chart.")
        else:
            df_bp = df.copy()
            if "unit_price_eur" not in df_bp.columns:
                df_bp["unit_price_eur"] = pd.to_numeric(df_bp.get("unit_price"), errors="coerce") * df_bp["rate_to_eur"]
            baselines = df_bp.groupby("category")["unit_price_eur"].median().rename("baseline_price_eur").reset_index()
            df_bp = df_bp.merge(baselines, on="category", how="left")
            df_bp["premium_pct"] = (df_bp["unit_price_eur"]-df_bp["baseline_price_eur"])/df_bp["baseline_price_eur"]
            df_bp["line_opportunity_eur"] = ((df_bp["unit_price_eur"]-df_bp["baseline_price_eur"]).clip(lower=0) *
                                             pd.to_numeric(df_bp.get("quantity"), errors="coerce")).fillna(0.0)

            sc = df_bp[["category","supplier","unit_price_eur","quantity","premium_pct","line_opportunity_eur"]].copy()
            sc = sc[(sc["unit_price_eur"].notna()) & (sc["quantity"].notna()) & (sc["quantity"]>0)]
            cats = ["All"] + sorted([c for c in sc["category"].dropna().unique()])
            cat_choice = st.selectbox("Category:", cats, index=0)
            sups = ["All"] + (sorted(sc.loc[sc["category"]==cat_choice,"supplier"].dropna().unique()) if cat_choice!="All"
                               else sorted(sc["supplier"].dropna().unique()))
            sup_choice = st.selectbox("Supplier:", sups, index=0)
            if cat_choice!="All": sc=sc[sc["category"]==cat_choice]
            if sup_choice!="All": sc=sc[sc["supplier"]==sup_choice]
            if sc.empty:
                st.info("No rows match the current filters.")
            else:
                q_low,q_high = np.nanpercentile(sc["quantity"], [0,99])
                p_low,p_high = np.nanpercentile(sc["unit_price_eur"], [0,99])
                if q_low==q_high: q_high=q_low+1
                if p_low==p_high: p_high=p_low+1
                use_log_x = st.checkbox("Log scale for Quantity", value=False)
                fig = px.scatter(sc, x="quantity", y="unit_price_eur", color="premium_pct", size="line_opportunity_eur",
                                 hover_data={"category":True,"supplier":True,"quantity":":,.0f","unit_price_eur":":,.4f","premium_pct":".1%","line_opportunity_eur":":,.0f"},
                                 color_continuous_scale="RdYlGn_r",
                                 labels={"quantity":"Quantity","unit_price_eur":"Unit price (EUR)","premium_pct":"Premium %"})
                fig.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0))
                fig.update_xaxes(range=[q_low,q_high], type="log" if use_log_x else "linear")
                fig.update_yaxes(range=[p_low,p_high])
                st.plotly_chart(fig, use_container_width=True)

# -------- Download pack --------
st.divider()
st.markdown("#### Download full dataset & summaries (XLSX)")
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    df.to_excel(w, index=False, sheet_name="Lines")
    cat.to_excel(w, index=False, sheet_name="Categories")
    sup_tot.to_excel(w, index=False, sheet_name="Suppliers")
    sup_cat.to_excel(w, index=False, sheet_name="Supplier_x_Category")
st.download_button("Download results.xlsx", buf.getvalue(), "procurement_diagnostics_results.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
