# ProcureIQ — Spend Explorer (full, robust version)
# -------------------------------------------------------------------
# - Sidebar: upload, mapping, spend source selection (detected vs Unit×Qty)
# - Horizontal KPI banners (equal sized)
# - Donut: Spend by Category (safe slider)
# - Top Suppliers by Spend (auto x-scale)
# - Supplier × Category Mix (Top-20 order & colors)
# - Data Quality pills + sample issue table
# - Part numbers metric (robust detection)
# - No external network dependency (FX=1 unless you supply)
# -------------------------------------------------------------------

from __future__ import annotations
import io
import re
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Plotly is used for charts; if unavailable, we degrade to tables.
try:
    import plotly.express as px
    PLOTLY = True
except Exception:  # pragma: no cover
    PLOTLY = False

# ----------------------------- PAGE SETUP -----------------------------
st.set_page_config(page_title="ProcureIQ — Spend Explorer", layout="wide")

st.markdown("""
<style>
  /* Sidebar sizing */
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

  /* KPI cards */
  .kpi-card {
    background:#ffffff; border:1px solid #e5e7eb; border-radius:12px;
    padding:12px 14px; height:100px; display:flex; flex-direction:column;
    justify-content:center; text-align:center; box-shadow:0 1px 2px rgba(0,0,0,0.03);
  }
  .kpi-lbl { color:#64748b; font-size:13px; margin-bottom:4px; }
  .kpi-val { color:#0f172a; font-size:22px; font-weight:700; letter-spacing:-0.5px; }

  /* Data quality */
  .dq-row { display:flex; flex-wrap:wrap; gap:10px; margin: 6px 0 12px 0; }
  .dq-pill {
    display:flex; align-items:center; gap:10px;
    padding: 11px 13px; border-radius: 12px; border:1px solid #e2e8f0;
    background:#fff; min-width: 260px; height: 52px;
    box-shadow:0 1px 2px rgba(2,8,23,.03);
  }
  .ok   { border-color:#bbf7d0; background:#ecfdf5; }
  .bad  { border-color:#fecaca; background:#fef2f2; }
  .dq-lbl { font-size: 13px; color: #475569; }
  .dq-val { font-weight: 800; font-size: 16px; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="banner">
  <div class="app-title">ProcureIQ — Spend Explorer</div>
  <div class="app-sub">Upload your spend cube, map columns in the sidebar, pick the category source, and explore.</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------- HELPERS -----------------------------
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _norm(s: pd.Series) -> pd.Series:
    if s is None: return pd.Series(dtype=str)
    return s.astype(str).str.strip().replace({"nan":"", "None":"", "NaN":""})

def _num(x) -> float:
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float)): return float(x)
    s = re.sub(r"[^\d,\.\-]", "", str(x))
    if "," in s and "." in s: s = s.replace(",", "")
    elif "," in s: s = s.replace(",", ".")
    try: return float(s)
    except: return np.nan

def _first(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for k in keys:
        for c in df.columns:
            if k.lower() in c.lower():
                return c
    return None

def _candidates(df: pd.DataFrame, keys: List[str]) -> List[str]:
    out=[]
    for c in df.columns:
        if any(k.lower() in c.lower() for k in keys):
            out.append(c)
    return out

def format_eur(v: float) -> str:
    if pd.isna(v): return "€ 0"
    if v >= 1_000_000: return f"€ {v/1_000_000:,.1f} M"
    if v >= 1_000: return f"€ {v/1_000:,.1f} k"
    return f"€ {v:,.0f}"

# ----------------------------- SIDEBAR / UPLOAD -----------------------------
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])
    if not up:
        st.info("Upload a file to begin.")
        st.stop()
    st.caption("Tip: keep column headers clean; the app auto-detects key fields.")

df = pd.read_excel(up)
df = _clean_cols(df)

# ----------------------------- COLUMN DETECTION -----------------------------
supplier_col = _first(df, ["supplier","vendor","seller"]) or df.columns[0]
cat_options = {
    "Category": _first(df, ["category","cat"]),
    "Item Family": _first(df, ["item family","family"]),
    "Item Family Group": _first(df, ["family group","group"]),
}
qty_col   = _first(df, ["qty","quantity","q'ty"])
price_col = _first(df, ["price","unit cost","unit price"])
spend_col = _first(df, ["spend","amount","value","purchase amount"])

# Part number candidates
part_candidates = _candidates(df, ["part", "item number", "item code", "sku", "material", "pn", "item_no", "item id"])

# Allow category choice (matches your previous UX)
with st.sidebar:
    st.header("Category source")
    choices = [k for k,v in cat_options.items() if v]
    default_label = choices[0] if choices else None
    cat_choice = st.radio("Use this as Category:", choices if choices else ["(not found)"], index=0 if choices else 0)
    category_col = cat_options.get(cat_choice)

    # Column mapping info
    st.subheader("Column mapping")
    st.success(f"Detected '{spend_col}' as spend column." if spend_col else
               ("Detected quantity+unit price." if qty_col and price_col else
                "No direct spend or qty×price detected — please review your file."))

# ----------------------------- CANONICAL FIELDS -----------------------------
df["supplier_canon"] = _norm(df[supplier_col]) if supplier_col in df.columns else ""
df["category_resolved"] = _norm(df[category_col]) if category_col in df.columns else ""

# Spend computation (no external FX; FX=1 by default)
if spend_col and spend_col in df.columns:
    df["_spend_eur"] = df[spend_col].map(_num)
elif qty_col and price_col and qty_col in df.columns and price_col in df.columns:
    df["_spend_eur"] = df[qty_col].map(_num) * df[price_col].map(_num)
else:
    df["_spend_eur"] = 0.0

df["_spend_eur"] = df["_spend_eur"].fillna(0).clip(lower=0)

# ----------------------------- PART NUMBERS -----------------------------
# Robust counting across typical columns; fallback to distinct (supplier, description) rows if nothing exists.
part_cols = [c for c in part_candidates if c in df.columns]
if part_cols:
    part_number_count = int(pd.unique(pd.concat([_norm(df[c]) for c in part_cols], axis=0)).size)
else:
    # Fallback heuristic: treat distinct rows as parts (very rough, but non-zero)
    part_number_count = int(df.drop_duplicates([supplier_col, category_col]).shape[0])

# ----------------------------- KPIs (HORIZONTAL) -----------------------------
total_spend       = df["_spend_eur"].sum()
total_categories  = df["category_resolved"].nunique()
total_suppliers   = df["supplier_canon"].nunique()
total_lines       = len(df)

c1, c2, c3, c4, c5 = st.columns(5, gap="small")
with c1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-lbl'>Total Spend</div>"
                f"<div class='kpi-val'>{format_eur(total_spend)}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='kpi-card'><div class='kpi-lbl'>Categories</div>"
                f"<div class='kpi-val'>{total_categories:,}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-lbl'>Suppliers</div>"
                f"<div class='kpi-val'>{total_suppliers:,}</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='kpi-card'><div class='kpi-lbl'>Part Numbers</div>"
                f"<div class='kpi-val'>{part_number_count:,}</div></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div class='kpi-card'><div class='kpi-lbl'>PO Lines</div>"
                f"<div class='kpi-val'>{total_lines:,}</div></div>", unsafe_allow_html=True)

st.markdown("<div class='spacer-12'></div>", unsafe_allow_html=True)

# ----------------------------- SPEND BY CATEGORY (DONUT) -----------------------------
st.markdown("<div class='block-title'>Spend by Category</div>", unsafe_allow_html=True)

cat_tot = (
    df.groupby("category_resolved", dropna=False)["_spend_eur"]
    .sum()
    .reset_index()
    .sort_values("_spend_eur", ascending=False)
)
cat_tot["category_resolved"] = cat_tot["category_resolved"].replace({"": "(blank)"})

if len(cat_tot) < 2:
    top_n = 1
else:
    top_n = st.slider("Top categories in donut", 2, min(15, len(cat_tot)), value=min(10, len(cat_tot)))

head = cat_tot.head(top_n)
tail_sum = cat_tot["_spend_eur"].iloc[top_n:].sum()
donut_df = head.copy()
if tail_sum > 0:
    donut_df = pd.concat([head, pd.DataFrame([{"category_resolved": "Other", "_spend_eur": tail_sum}])], ignore_index=True)

palette = px.colors.qualitative.Set3 if PLOTLY else None
cat_color_map = {c: palette[i % len(palette)] for i, c in enumerate(cat_tot["category_resolved"])} if PLOTLY else {}

if PLOTLY and not donut_df.empty:
    fig_d = px.pie(
        donut_df, names="category_resolved", values="_spend_eur",
        hole=0.5, color="category_resolved", color_discrete_map=cat_color_map
    )
    fig_d.update_traces(textposition="inside", textinfo="percent+label")
    fig_d.update_layout(height=460, margin=dict(l=0,r=0,t=10,b=0),
                        legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig_d, use_container_width=True)
else:
    st.dataframe(donut_df)

st.markdown("<div class='spacer-12'></div>", unsafe_allow_html=True)

# ----------------------------- TOP SUPPLIERS BY SPEND -----------------------------
st.markdown("<div class='block-title'>Top Suppliers by Spend</div>", unsafe_allow_html=True)
supp = df.groupby("supplier_canon")["_spend_eur"].sum().reset_index()
supp["supplier_canon"] = supp["supplier_canon"].replace({"":"(blank)"})
top20 = supp.sort_values("_spend_eur", ascending=False).head(20)
top20_suppliers = top20["supplier_canon"].tolist()

if PLOTLY and not top20.empty:
    max_sp = top20["_spend_eur"].max()
    fig_b = px.bar(
        top20.sort_values("_spend_eur"),
        x="_spend_eur", y="supplier_canon",
        orientation="h",
        text=[format_eur(x) for x in top20.sort_values('_spend_eur')["_spend_eur"]],
    )
    fig_b.update_traces(textposition="outside", cliponaxis=False)
    fig_b.update_layout(
        height=560,
        margin=dict(l=10,r=40,t=10,b=10),
        xaxis=dict(range=[0, max_sp*1.12], title="Spend (€)", gridcolor="#f1f5f9"),
        yaxis=dict(title="Supplier", automargin=True),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    st.plotly_chart(fig_b, use_container_width=True)
else:
    st.dataframe(top20)

st.markdown("<div class='spacer-12'></div>", unsafe_allow_html=True)

# ----------------------------- SUPPLIER × CATEGORY MIX -----------------------------
st.markdown("<div class='block-title'>Supplier × Category Mix (Top 20 suppliers)</div>", unsafe_allow_html=True)

mix = pd.DataFrame()
if top20_suppliers:
    mix = (
        df[df["supplier_canon"].isin(top20_suppliers)]
        .groupby(["supplier_canon","category_resolved"])["_spend_eur"].sum()
        .reset_index()
        .rename(columns={"supplier_canon":"Supplier"})
    )
    mix["Supplier"] = mix["Supplier"].replace({"":"(blank)"})
    # keep exact order as top-20
    ordered = list(reversed(top20_suppliers))  # large on top (horizontal bar)
    mix["Supplier"] = pd.Categorical(mix["Supplier"], categories=ordered, ordered=True)

if PLOTLY and not mix.empty:
    totals = mix.groupby("Supplier")["_spend_eur"].transform("sum")
    mix["share_pct"] = np.where(totals>0, mix["_spend_eur"]/totals*100, 0)

    # same colors as donut
    fig_m = px.bar(
        mix, x="share_pct", y="Supplier",
        color="category_resolved", orientation="h",
        barmode="stack", color_discrete_map=cat_color_map
    )
    fig_m.update_layout(
        height=max(600, len(top20_suppliers)*26 + 180),
        margin=dict(l=10, r=40, t=10, b=140),
        legend=dict(orientation="h", y=-0.25, x=0, title="Category"),
        xaxis=dict(title="Share (%)", ticksuffix="%", gridcolor="#eef2f7"),
        yaxis=dict(title="Supplier", automargin=True),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    st.plotly_chart(fig_m, use_container_width=True)
else:
    st.info("Mix chart will appear once the Top-20 suppliers are available.")

st.markdown("<div class='spacer-16'></div>", unsafe_allow_html=True)

# ----------------------------- DATA QUALITY -----------------------------
st.markdown("<div class='block-title'>Data Quality</div>", unsafe_allow_html=True)

missing_price = int(df[price_col].isna().sum()) if price_col and price_col in df.columns else 0
missing_qty   = int(df[qty_col].isna().sum())   if qty_col and qty_col in df.columns   else 0
zero_neg_price = int((df[price_col] <= 0).sum()) if price_col and price_col in df.columns else 0
zero_neg_qty   = int((df[qty_col] <= 0).sum())   if qty_col and qty_col in df.columns   else 0
blank_supplier = int(df["supplier_canon"].eq("").sum())
blank_category = int(df["category_resolved"].eq("").sum())

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='dq-row'>", unsafe_allow_html=True)
    st.markdown(f"<div class='dq-pill ok'><div class='dq-lbl'>Missing unit price</div><div class='dq-val'>{missing_price}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='dq-pill ok'><div class='dq-lbl'>Missing quantity</div><div class='dq-val'>{missing_qty}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='dq-row'>", unsafe_allow_html=True)
    st.markdown(f"<div class='dq-pill bad'><div class='dq-lbl'>Zero/negative price</div><div class='dq-val'>{zero_neg_price}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='dq-pill bad'><div class='dq-lbl'>Zero/negative quantity</div><div class='dq-val'>{zero_neg_qty}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='dq-row'>", unsafe_allow_html=True)
    st.markdown(f"<div class='dq-pill bad'><div class='dq-lbl'>Blank supplier</div><div class='dq-val'>{blank_supplier}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='dq-pill bad'><div class='dq-lbl'>Blank category</div><div class='dq-val'>{blank_category}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Sample issues table
issues_mask = pd.Series(False, index=df.index)
if price_col and price_col in df.columns: issues_mask |= (df[price_col] <= 0) | df[price_col].isna()
if qty_col and qty_col in df.columns:     issues_mask |= (df[qty_col] <= 0) | df[qty_col].isna()
sample_cols = []
for c in [supplier_col, category_col, price_col, qty_col, "_spend_eur"]:
    if c and c in df.columns: sample_cols.append(c)
issues = df.loc[issues_mask, sample_cols].copy()
if "_spend_eur" in issues.columns:
    issues.rename(columns={"_spend_eur":"Spend (EUR)"}, inplace=True)
if supplier_col in issues.columns:
    issues.rename(columns={supplier_col:"Supplier"}, inplace=True)
if category_col in issues.columns:
    issues.rename(columns={category_col:"Category"}, inplace=True)
if price_col and price_col in issues.columns:
    issues.rename(columns={price_col:"Unit Price"}, inplace=True)
if qty_col and qty_col in issues.columns:
    issues.rename(columns={qty_col:"Quantity"}, inplace=True)

if not issues.empty:
    st.dataframe(issues.head(100), use_container_width=True)
else:
    st.success("No blocking data quality issues detected.")
