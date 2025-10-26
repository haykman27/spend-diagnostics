# app.py — ProcureIQ — Spend Explorer
# -----------------------------------------------------------------------------
# This is the full working app with:
#  - Upload + column mapping in sidebar (category source selector)
#  - Spend source selection (Detected spend / Unit×Qty×FX / Auto-validate)
#  - KPIs (HORIZONTAL banners — the only visual change vs previous build)
#  - Donut: Spend by Category
#  - Bar: Top 20 Suppliers by Spend (auto x-scale)
#  - Supplier × Category Mix (Top 20, same order, same colors)
#  - Data Quality badges + issues table
# -----------------------------------------------------------------------------

import io
import re
import math
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False

from rapidfuzz import process, fuzz

# ----------------------------- PAGE CONFIG ------------------------------------
st.set_page_config(page_title="ProcureIQ — Spend Explorer", layout="wide")

# ----------------------------- STYLE / THEME ----------------------------------
st.markdown("""
<style>
  [data-testid="stSidebar"] { min-width:360px; max-width:420px; }
  .banner {
    background: linear-gradient(135deg, rgba(14,165,233,.10), rgba(139,92,246,.06));
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 20px 22px 18px 22px;
    margin: 6px 0 8px 0;
    box-shadow: 0 2px 6px rgba(2,8,23,.04);
  }
  .app-title { font-size: 32px; font-weight: 800; letter-spacing:-.02rem; margin:0; color:#0f172a; }
  .app-sub   { color:#475569; font-size: 14px; margin:6px 0 0 0; }
  .block-title { font-weight: 800; font-size: 1.05rem; margin: 6px 0 8px 6px; color: #0f172a; }
  .spacer-12 { margin-top: 12px; }
  .spacer-16 { margin-top: 16px; }
  .spacer-24 { margin-top: 24px; }

  .dq-row { display:flex; flex-wrap:wrap; gap:10px; margin: 6px 0 12px 0; }
  .dq-pill {
    display:flex; align-items:center; gap:10px;
    padding: 11px 13px; border-radius: 12px; border:1px solid #e2e8f0;
    background:#fff; min-width: 240px; height: 50px;
    box-shadow:0 1px 2px rgba(2,8,23,.03);
  }
  .ok   { border-color:#bbf7d0; background:#ecfdf5; }
  .bad  { border-color:#fecaca; background:#fef2f2; }
  .dq-lbl { font-size: 13px; color: #475569; }
  .dq-val { font-weight: 800; font-size: 16px; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- HEADER -----------------------------------------
st.markdown("""
<div class="banner">
  <div class="app-title">ProcureIQ — Spend Explorer</div>
  <div class="app-sub">Upload your spend cube, map columns in the sidebar, pick the category source, and explore.</div>
</div>
""", unsafe_allow_html=True)

BASE_CCY = "EUR"

# ----------------------------- HELPERS ----------------------------------------
def norm_headers(cols):
    return [re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()) for c in cols]

def parse_number_robust(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = re.sub(r"[^\d,\.\-]", "", str(x))
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and s.count(",")==1 and len(s.split(",")[-1]) in (2,3):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try: return float(s)
    except: return np.nan

def ensure_float(s: pd.Series) -> pd.Series:
    if s.dtype.kind in ("i","u","f","b"): s = s.astype(float)
    else: s = pd.to_numeric(s, errors="coerce")
    return s.fillna(0.0).astype(float)

def pick_best_col(candidates, cols):
    cols_norm = norm_headers(cols)
    for cand in candidates:
        for i, c in enumerate(cols_norm):
            if cand.lower() == c:
                return cols[i]
    # fuzzy
    best, score = None, 0
    for i, c in enumerate(cols_norm):
        for cand in candidates:
            sc = fuzz.token_sort_ratio(cand, c)
            if sc > score:
                score, best = sc, cols[i]
    return best

# Static FX (fallback only). Your last version used ECB fetch with cache;
# here we keep a safe, static table to avoid internet dependency.
ECB_RATES = {
    "EUR": 1.0, "USD": 0.92, "GBP": 1.10, "CHF": 1.02, "CNY": 0.13,
    "PLN": 0.23, "CZK": 0.040, "HUF": 0.0026, "TRY": 0.028, "INR": 0.0114,
    "MXN": 0.054, "MAD": 0.092, "TWD": 0.029
}

def fx_to_eur(series_ccy):
    c = str(series_ccy).upper() if isinstance(series_ccy, str) else ""
    return ECB_RATES.get(c, np.nan)

# ----------------------------- SIDEBAR: Upload + Mapping ----------------------
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])
    if not up:
        st.info("Upload a file to begin.")
        st.stop()

df0 = pd.read_excel(up)
df0.columns = [str(c) for c in df0.columns]
cols = df0.columns.tolist()
cols_norm = norm_headers(cols)
raw = df0.copy()

# --- Detect useful columns (heuristics like in last build) ---
supplier_col = pick_best_col(
    ["supplier","vendor","supplier name","vendor name"], cols
) or cols[0]
category_cols = {
    "Category": pick_best_col(["category","cat","part category","material group"], cols),
    "Item Family": pick_best_col(["item family","family","commodity"], cols),
    "Family Group": pick_best_col(["item family group","family group"], cols),
}

qty_col = pick_best_col(["quantity","qty","order qty","po qty"], cols)
price_col = pick_best_col(["unit price","price","po unit price","unit cost"], cols)
ccy_col = pick_best_col(["currency","ccy","iso currency","curr"], cols)

spend_candidates = [
    "purchase amount (base curr)", "purchase amount (global curr)",
    "spend", "amount", "po amount","line amount"
]
spend_col = pick_best_col(spend_candidates, cols)

# Make working dataframe with normalized numeric fields
df = raw.copy()
for c in [qty_col, price_col, spend_col]:
    if c and c in df.columns:
        df[c] = df[c].map(parse_number_robust)

if ccy_col and ccy_col in df.columns:
    df[ccy_col] = df[ccy_col].astype(str).str.strip().str.upper()

# -------------- Sidebar: Category source selection ---------------------------
with st.sidebar:
    st.header("Column mapping")
    st.success(f"Detected '{spend_col}' as spend column.") if spend_col else st.warning("No spend column found.")
    st.subheader("Category source")
    cat_choice = st.radio(
        "Use this as Category:",
        ["Category", "Item Family", "Family Group"],
        index=0 if category_cols["Category"] else (1 if category_cols["Item Family"] else 2)
    )

# Resolve category based on choice
category_resolved_col = category_cols.get(cat_choice) or category_cols["Category"] or cols[0]
df["category_resolved"] = df[category_resolved_col].astype(str).str.strip().replace({"": "Unspecified"})

# Supplier canonical
df["supplier_canon"] = df[supplier_col].astype(str).str.strip().replace({"": "Unspecified"})

# -------------- Spend source selection ---------------------------------------
def detect_spend_from_column(_df):
    if spend_col and spend_col in _df.columns:
        s = ensure_float(_df[spend_col])
        return s, "detected"
    return pd.Series(np.nan, index=_df.index), "missing"

def compute_unitqtyfx(_df):
    amt = pd.Series(0.0, index=_df.index)
    if qty_col and price_col:
        qty = ensure_float(_df[qty_col]) if qty_col in _df.columns else 0.0
        pr  = ensure_float(_df[price_col]) if price_col in _df.columns else 0.0
        amt = qty * pr
    if ccy_col and ccy_col in _df.columns:
        rates = _df[ccy_col].map(fx_to_eur).fillna(1.0)
        amt = amt * rates
    return amt

with st.sidebar:
    st.subheader("Spend source")
    spend_choice = st.radio(
        "Choose the spend source:",
        ["Use detected spend column", "Use Unit×Qty×FX", "Auto-validate"],
        index=2
    )

detected_spend, det_status = detect_spend_from_column(df)
uqfx_spend = compute_unitqtyfx(df)

if spend_choice == "Use detected spend column":
    spend = detected_spend
    note = f"Detected spend total: € {detected_spend.sum():,.0f}"
elif spend_choice == "Use Unit×Qty×FX":
    spend = uqfx_spend
    note = f"Unit×Qty×FX total: € {uqfx_spend.sum():,.0f}"
else:
    # Auto-validate: if mismatch > 15%, use Unit×Qty×FX
    if spend_col and det_status == "detected":
        if detected_spend.sum() <= 0 or abs(detected_spend.sum() - uqfx_spend.sum())/max(1,abs(detected_spend.sum())) > 0.15:
            spend = uqfx_spend
            note = "Auto-validate switched to Unit×Qty×FX because detected spend was inconsistent."
        else:
            spend = detected_spend
            note = "Auto-validate kept detected spend column."
    else:
        spend = uqfx_spend
        note = "Auto-validate used Unit×Qty×FX (no detected spend)."

with st.sidebar:
    st.info(note)
df["_spend_eur"] = ensure_float(spend)

# ----------------------------- KPIs (HORIZONTAL) ------------------------------
total_spend = df["_spend_eur"].sum()
total_categories = df["category_resolved"].nunique()
total_suppliers = df["supplier_canon"].nunique()

# Part numbers: use any likely column; fallback to count of unique item-like fields
part_cols = [
    pick_best_col(["part number","part no","item","item number","item code","sku","material","material number","material code"], cols),
]
part_cols = [c for c in part_cols if c]
if part_cols:
    part_count = df[part_cols[0]].astype(str).str.strip().replace({"":np.nan}).nunique()
else:
    # conservative fallback: unique on (category, supplier, unit price)
    if price_col and price_col in df.columns:
        part_count = df[["category_resolved","supplier_canon",price_col]].dropna().drop_duplicates().shape[0]
    else:
        part_count = df[["category_resolved","supplier_canon"]].drop_duplicates().shape[0]

total_lines = len(df)

# ---- Horizontal banners (only change vs. last version) ----
c1, c2, c3, c4, c5 = st.columns(5, gap="small")
card_style = (
    "background-color:#ffffff; border:1px solid #e5e7eb; border-radius:12px; "
    "padding:12px 14px; height:100px; display:flex; flex-direction:column; "
    "justify-content:center; text-align:center; box-shadow:0 1px 2px rgba(0,0,0,0.03);"
)
label_style = "color:#64748b; font-size:13px; margin-bottom:4px;"
value_style = "color:#0f172a; font-size:22px; font-weight:700; letter-spacing:-0.5px;"

with c1:
    st.markdown(f"<div style='{card_style}'><div style='{label_style}'>Total Spend</div><div style='{value_style}'>€ {total_spend/1_000_000:,.1f} M</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div style='{card_style}'><div style='{label_style}'>Categories</div><div style='{value_style}'>{total_categories:,}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div style='{card_style}'><div style='{label_style}'>Suppliers</div><div style='{value_style}'>{total_suppliers:,}</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div style='{card_style}'><div style='{label_style}'>Part Numbers</div><div style='{value_style}'>{part_count:,}</div></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div style='{card_style}'><div style='{label_style}'>PO Lines</div><div style='{value_style}'>{total_lines:,}</div></div>", unsafe_allow_html=True)

st.markdown("<div class='spacer-12'></div>", unsafe_allow_html=True)

# ----------------------------- CATEGORY DONUT + TOP 20 BAR --------------------
left, right = st.columns([0.60, 0.40])  # more space left for donut legend

# color palette
palette = px.colors.qualitative.Set3 if PLOTLY else None
cats = df["category_resolved"].fillna("Unspecified").astype(str)
cat_tot = df.groupby("category_resolved")["_spend_eur"].sum().reset_index().sort_values("_spend_eur", ascending=False)

if PLOTLY and not cat_tot.empty:
    with left:
        st.markdown("<div class='block-title'>Spend by Category</div>", unsafe_allow_html=True)

        # Top-N in donut with “Other”
        top_n = st.slider("Top categories in donut", 3, min(15, len(cat_tot)), value=min(10, len(cat_tot)), key="topN")
        head = cat_tot.head(top_n)
        tail_sum = cat_tot["_spend_eur"].iloc[top_n:].sum()
        donut_df = head.copy()
        if tail_sum > 0:
            donut_df = pd.concat([head, pd.DataFrame([{"category_resolved":"Other","_spend_eur":tail_sum}])], ignore_index=True)

        fig_d = px.pie(
            donut_df, names="category_resolved", values="_spend_eur",
            hole=0.50, color="category_resolved",
            color_discrete_sequence=palette
        )
        fig_d.update_traces(textposition="inside", textinfo="label+percent", hovertemplate="%{label}<br>€ %{value:,.0f}")
        fig_d.update_layout(
            height=520, margin=dict(l=0,r=0,t=10,b=0), showlegend=True,
            legend=dict(orientation="h", y=-0.18)
        )
        st.plotly_chart(fig_d, use_container_width=True)

# Top 20 suppliers bar (auto x-scale)
supp_tot = df.groupby("supplier_canon")["_spend_eur"].sum().reset_index()
supp_top20 = supp_tot.sort_values("_spend_eur", ascending=False).head(20).copy()
top20_order = supp_top20["supplier_canon"].tolist()  # keep for mix plot

if PLOTLY and not supp_top20.empty:
    with right:
        st.markdown("<div class='block-title'>Top Suppliers by Spend</div>", unsafe_allow_html=True)
        fig_b = px.bar(
            supp_top20.sort_values("_spend_eur"),  # ascending to have largest on top after orientation
            x="_spend_eur", y="supplier_canon", orientation="h",
            text=[f"€ {v/1_000_000:.1f} M" for v in supp_top20.sort_values("_spend_eur")["_spend_eur"]],
        )
        max_val = supp_top20["_spend_eur"].max()
        rng = math.ceil(max_val/1_000_000)+0.2
        fig_b.update_layout(
            height=520, margin=dict(l=5,r=20,t=6,b=10),
            xaxis=dict(title="", range=[0, rng*1_000_000], showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="Supplier")
        )
        fig_b.update_traces(marker_color="#10a5a7", hovertemplate="%{y}<br>€ %{x:,.0f}")
        st.plotly_chart(fig_b, use_container_width=True)

# ----------------------------- SUPPLIER × CATEGORY MIX ------------------------
st.markdown('<div class="spacer-12"></div>', unsafe_allow_html=True)
st.markdown('<div class="block-title">Supplier × Category Mix (Top 20 suppliers)</div>', unsafe_allow_html=True)

mix = pd.DataFrame()
if top20_order:
    mix = (
        df[df["supplier_canon"].isin(top20_order)]
        .groupby(["supplier_canon","category_resolved"])["_spend_eur"].sum()
        .reset_index()
        .rename(columns={"supplier_canon":"Supplier"})
    )

if PLOTLY and not mix.empty:
    totals = mix.groupby("Supplier")["_spend_eur"].transform("sum")
    mix["share_pct"] = np.where(totals>0, (mix["_spend_eur"]/totals)*100.0, 0.0)

    # enforce SAME order (top -> bottom like in bar)
    supplier_order_for_plot = list(reversed(top20_order))
    mix["Supplier"] = pd.Categorical(mix["Supplier"], categories=supplier_order_for_plot, ordered=True)

    # color map matching donut
    cat_list = donut_df["category_resolved"].tolist() if 'donut_df' in locals() else sorted(df["category_resolved"].unique().tolist())
    base_colors = px.colors.qualitative.Set3
    color_map = {c: base_colors[i % len(base_colors)] for i, c in enumerate(cat_list)}

    fig_m = px.bar(
        mix, x="share_pct", y="Supplier", color="category_resolved",
        orientation="h", barmode="stack", color_discrete_map=color_map
    )
    fig_m.update_layout(
        height=max(560, len(top20_order)*26 + 180),
        margin=dict(l=10, r=40, t=10, b=120),
        legend=dict(orientation="h", y=-0.22, x=0, title="Category"),
        xaxis=dict(title="Share (%)", ticksuffix="%", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Supplier", automargin=True),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    st.plotly_chart(fig_m, use_container_width=True)
else:
    st.info("Mix chart will appear once the Top-20 suppliers are available.")

# ----------------------------- DATA QUALITY -----------------------------------
st.markdown('<div class="spacer-16"></div>', unsafe_allow_html=True)
st.markdown('<div class="block-title">Data Quality</div>', unsafe_allow_html=True)

# basic checks like last build
unknown_ccy = 0
if ccy_col and ccy_col in df.columns:
    iso_ok = set(ECB_RATES.keys())
    unknown_ccy = int((~df[ccy_col].isin(iso_ok)).sum())

missing_price = int(df[price_col].isna().sum()) if price_col and price_col in df.columns else 0
missing_qty   = int(df[qty_col].isna().sum())   if qty_col and qty_col in df.columns   else 0

zero_neg_price = int((df[price_col] <= 0).sum()) if price_col and price_col in df.columns else 0
zero_neg_qty   = int((df[qty_col]   <= 0).sum()) if qty_col and qty_col in df.columns   else 0

blank_supplier = int((df["supplier_canon"].astype(str).str.strip()=="").sum())
blank_category = int((df["category_resolved"].astype(str).str.strip()=="").sum())

# pills
def pill(label, value, bad=False):
    cls = "bad" if bad and value>0 else "ok"
    st.markdown(
        f"<div class='dq-pill {cls}'><div class='dq-lbl'>{label}</div>"
        f"<div class='dq-val'>{value}</div></div>", unsafe_allow_html=True
    )

cqa, cqb, cqc = st.columns(3)
with cqa:
    st.markdown("<div class='dq-row'>", unsafe_allow_html=True)
    pill("Unknown currency:", unknown_ccy, bad=True)
    pill("Missing unit price:", missing_price, bad=True)
    pill("Missing quantity:", missing_qty, bad=True)
    st.markdown("</div>", unsafe_allow_html=True)
with cqb:
    st.markdown("<div class='dq-row'>", unsafe_allow_html=True)
    pill("Zero/negative price:", zero_neg_price, bad=True)
    pill("Zero/negative qty:", zero_neg_qty, bad=True)
    st.markdown("</div>", unsafe_allow_html=True)
with cqc:
    st.markdown("<div class='dq-row'>", unsafe_allow_html=True)
    pill("Blank supplier:", blank_supplier, bad=True)
    pill("Blank category:", blank_category, bad=True)
    st.markdown("</div>", unsafe_allow_html=True)

# issues table
issues_mask = pd.Series(False, index=df.index)
if price_col and price_col in df.columns: issues_mask |= df[price_col].isna() | (df[price_col] <= 0)
if qty_col and qty_col in df.columns:     issues_mask |= df[qty_col].isna()  | (df[qty_col] <= 0)
if ccy_col and ccy_col in df.columns:     issues_mask |= ~df[ccy_col].isin(list(ECB_RATES.keys()))

issues = df.loc[issues_mask, [
    "supplier_canon","category_resolved",
    ccy_col if ccy_col else df.columns[0],
    price_col if price_col else df.columns[0],
    qty_col if qty_col else df.columns[0],
    "_spend_eur"
]].copy()

issues.columns = ["Supplier","Category","Currency","Unit Price","Quantity","Spend (EUR)"]

if not issues.empty:
    st.dataframe(issues.head(1000), use_container_width=True)
else:
    st.success("No blocking data quality issues detected in key fields.")
