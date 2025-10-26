# app.py
# ProcureIQ — Spend Explorer (clean, modern, and consistent)
# --------------------------------------------------------------------
import re
import numpy as np
import pandas as pd
import streamlit as st

# Plotly may not be installed locally; we'll guard imports to keep the app resilient
try:
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False

# =========================
# ---------- THEME --------
# =========================
st.set_page_config(
    page_title="ProcureIQ — Spend Explorer",
    page_icon="💎",
    layout="wide",
)

# Minimal CSS polish for KPIs and section titles
st.markdown(
    """
    <style>
      :root {
        --card-bg: #ffffff;
        --muted: #64748b;
        --ink: #0f172a;
        --accent: #06b6d4;
        --border: #e2e8f0;
      }
      .big-hero {
        padding: 18px 22px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: linear-gradient(180deg,#f0fbff, #f8fbff);
        margin-bottom: 6px;
      }
      .kpi-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 16px;
        margin: 10px 0 6px 0;
      }
      .kpi {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 18px 22px 18px;
        height: 120px;
      }
      .kpi .label {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 8px;
      }
      .kpi .value {
        color: var(--ink);
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.6px;
      }
      .block-title {
        font-size: 20px;
        font-weight: 700;
        margin: 8px 0 4px 0;
      }
      .spacer-12 { height: 12px; }
      .spacer-24 { height: 24px; }
      .badge-row {
        display:flex; gap:14px; flex-wrap: wrap; margin: 4px 0 10px 0;
      }
      .badge {
        border: 1px solid var(--border);
        background: #f8fafc;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 13px;
      }
      .badge.bad {
        background: #fff1f2;
        border-color: #fecdd3;
        color: #be123c;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================================
# ------- SMALL HELPER UTILS --------
# ===================================

def _norm_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.replace({"nan": "", "None": "", "NaN": ""})
    return s

def _to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    # remove thousand separators and normalize decimals
    s = s.replace("’", "").replace("'", "")
    s = s.replace(" ", "").replace(",", ".")
    # keep only valid numeric pattern (for safety)
    m = re.match(r"^-?\d+(\.\d+)?$", s)
    try:
        return float(m.group(0)) if m else np.nan
    except Exception:
        return np.nan

def pick_best_col(df, candidates, exclude=set()):
    # return first existing candidate not in exclude
    cols = [c for c in df.columns if c.lower() not in exclude]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map and lower_map[k] not in exclude:
            return lower_map[k]
    # loose contains
    for c in df.columns:
        lc = c.lower()
        if lc in exclude:
            continue
        if any(k in lc for k in candidates):
            return c
    return None

def pick_best_part_column(df, chosen_cat_col=None):
    # Don't pick columns that are clearly category-like
    forbidden = set()
    if chosen_cat_col:
        forbidden.add(chosen_cat_col.lower())
    for k in ["item family", "item_family", "itemfamily", "family", "family group", "item family group"]:
        for c in df.columns:
            if k in c.lower():
                forbidden.add(c.lower())

    keywords = ["material", "mat.", "mat_", "item", "sku", "part", "pn", "code", "number", "no."]
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if lc in forbidden:
            continue
        if any(k in lc for k in keywords):
            s = _norm_text(df[c])
            # keep only columns with reasonable presence of digits
            mask = s.str.contains(r"\d", regex=True, na=False)
            if mask.mean() > 0.3:  # at least 30% look like codes
                candidates.append((c, s.nunique(dropna=True)))

    if not candidates:
        return None, 0
    # choose the most granular one
    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0][0]
    count = df[best].astype(str).str.strip().replace({"": np.nan}).nunique(dropna=True)
    return best, int(count)

# ===================================
# --------- SIDEBAR / DATA ----------
# ===================================

with st.sidebar:
    st.subheader("Data")
    up = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx", "xls"])
    st.caption("The app auto-detects key columns. You can override picks below if needed.")

    st.markdown("### Column mapping")

if up is None:
    st.markdown(
        """
        <div class="big-hero">
          <h2 style="margin:0 0 6px 0;">ProcureIQ — Spend Explorer</h2>
          <div style="color:#475569">Upload your spend cube in the sidebar and explore categories, suppliers, and savings opportunities.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Upload your Excel file in the sidebar to begin.")
    st.stop()

# read excel
raw = pd.read_excel(up)
raw.columns = [str(c).strip() for c in raw.columns]

# Normalized/numeric helpers
df = raw.copy()

# --- Detect Supplier
supplier_col = pick_best_col(df, ["supplier", "supplier name", "vendor", "vendor name"])
if supplier_col is None:
    # choose a text column with many distincts as fallback
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    supplier_col = text_cols[0] if text_cols else df.columns[0]
df["supplier"] = _norm_text(df[supplier_col])

# Canonical supplier for consistent naming
df["supplier_canon"] = _norm_text(df["supplier"])

with st.sidebar:
    st.write("**Supplier column:**", supplier_col)

# --- Detect Category
cat_candidates = ["category", "item family", "family", "item family group", "family group", "item group", "category name"]
category_col = pick_best_col(df, cat_candidates)
if category_col is None:
    # pick a column with medium distinct count (not too many, not 1)
    distincts = [(c, df[c].nunique()) for c in df.columns]
    distincts.sort(key=lambda x: x[1], reverse=True)
    category_col = distincts[len(distincts)//2][0]
df["category_resolved"] = _norm_text(df[category_col])

with st.sidebar:
    st.write("**Category column:**", category_col)

# --- Detect Quantity / Unit Price / Spend
qty_col = pick_best_col(df, ["quantity", "qty"])
price_col = pick_best_col(df, ["unit price", "price", "unit cost", "unit net price", "po price", "net price"])
spend_col = pick_best_col(df, ["purchase amount", "amount", "total", "gross amount", "value", "ext price", "extended price"])

if qty_col:   df["_qty"] = df[qty_col].map(_to_float)
else:         df["_qty"] = np.nan

if price_col: df["_unit_price"] = df[price_col].map(_to_float)
else:         df["_unit_price"] = np.nan

if spend_col: df["_spend_raw"] = df[spend_col].map(_to_float)
else:         df["_spend_raw"] = np.nan

# Currency (not used for FX here, but shown in DQ)
ccy_col = pick_best_col(df, ["currency", "curr", "iso", "ccy"])

# Spend source switch (auto-validate like behavior)
with st.sidebar:
    st.markdown("### Spend source")
    src_choice = st.radio(
        "Choose spend source:",
        options=["Auto (prefer spend column)", "Use detected spend column", "Use Unit×Qty"],
        index=0
    )

# Decide spend in EUR (no FX here -> treat as base)
if src_choice == "Use Unit×Qty" or (src_choice == "Auto (prefer spend column)" and df["_spend_raw"].isna().all()):
    df["_spend_eur"] = df["_unit_price"] * df["_qty"]
    spend_source_msg = "Unit×Qty"
else:
    df["_spend_eur"] = df["_spend_raw"]
    spend_source_msg = "Detected spend column"

# Clean negatives to keep charts sane; still visible in DQ
df["_spend_eur_clean"] = df["_spend_eur"].where(df["_spend_eur"] >= 0, 0)

# ===================================
# ---------- KPIs / HEADER ----------
# ===================================

# KPIs
total_spend = df["_spend_eur_clean"].sum()
n_categories = df["category_resolved"].nunique(dropna=True)
n_suppliers = df["supplier_canon"].nunique(dropna=True)
po_lines = len(df)

# Part numbers
best_part_col, part_count = pick_best_part_column(df, chosen_cat_col=category_col)
with st.sidebar:
    st.subheader("Part numbers")
    if best_part_col:
        st.info(f"Detected: **{best_part_col}**  •  uniques: **{part_count:,}**")
    else:
        st.warning("No obvious part-number column detected.")
    pn_override = st.selectbox("Override (optional):", ["(auto)"] + list(raw.columns), index=0)
    if pn_override != "(auto)":
        best_part_col = pn_override
        s = _norm_text(raw[best_part_col])
        part_count = s.replace({"": np.nan}).nunique(dropna=True)

# HERO + KPIs
st.markdown(
    """
    <div class="big-hero">
      <h2 style="margin:0 0 2px 0;">ProcureIQ — Spend Explorer</h2>
      <div style="color:#475569">Explore category mix, supplier concentration, and quality signals. Spend source: <b>"""
    + spend_source_msg +
    """</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="kpi-row">', unsafe_allow_html=True)

def _fmt_eur_m(eur):
    return f"€ {eur/1_000_000:,.1f} M".replace(",", " ").replace(".0", "")

kpis = [
    ("Total Spend", _fmt_eur_m(total_spend)),
    ("Categories", f"{n_categories:,}"),
    ("Suppliers", f"{n_suppliers:,}"),
    ("Part Numbers", f"{part_count:,}"),
    ("PO Lines", f"{po_lines:,}"),
]
for label, val in kpis:
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value">{val}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)  # end kpi-row

st.markdown('<div class="spacer-12"></div>', unsafe_allow_html=True)

# ===================================
# --------- DONUT & TOP-20 ----------
# ===================================

# Spend by Category (donut)
cat_agg = (
    df.groupby("category_resolved", dropna=False)["_spend_eur_clean"].sum().reset_index()
    .rename(columns={"_spend_eur_clean": "spend_eur"})
)
cat_agg = cat_agg.sort_values("spend_eur", ascending=False)
cat_agg["share"] = cat_agg["spend_eur"] / cat_agg["spend_eur"].sum() * 100

# COLOR MAP (consistent across donut and mix)
if PLOTLY:
    base_palette = px.colors.qualitative.Set3
else:
    base_palette = ["#60a5fa","#fb7185","#34d399","#fbbf24","#a78bfa","#f472b6","#22d3ee","#f59e0b","#10b981","#ef4444"]
cats_sorted = cat_agg["category_resolved"].fillna("Other").tolist()
color_map = {c: base_palette[i % len(base_palette)] for i, c in enumerate(cats_sorted)}

col_donut, col_bar = st.columns((1.05, 1.4))

with col_donut:
    st.markdown('<div class="block-title">Spend by Category</div>', unsafe_allow_html=True)
    top_n = st.slider("Top categories in donut", min_value=4, max_value=min(20, len(cat_agg)), value=min(10, len(cat_agg)))
    donut_df = cat_agg.head(top_n).copy()
    other = cat_agg.iloc[top_n:]["spend_eur"].sum()
    if other > 0:
        donut_df = pd.concat([donut_df, pd.DataFrame({"category_resolved":["Other"], "spend_eur":[other]})], ignore_index=True)

    if PLOTLY:
        fig_d = px.pie(
            donut_df,
            values="spend_eur",
            names="category_resolved",
            color="category_resolved",
            color_discrete_map=color_map,
            hole=0.55
        )
        fig_d.update_traces(textposition='inside', textinfo='percent+label')
        fig_d.update_layout(margin=dict(l=4,r=4,t=10,b=10), height=420, showlegend=True, legend=dict(orientation="h"))
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.dataframe(donut_df)

with col_bar:
    st.markdown('<div class="block-title">Top Suppliers by Spend</div>', unsafe_allow_html=True)
    sup_tot = (
        df.groupby("supplier_canon", dropna=False)
          .agg(spend_eur=("_spend_eur_clean", "sum"),
               lines=("supplier_canon","count"),
               categories=("category_resolved", pd.Series.nunique))
          .reset_index()
          .rename(columns={"supplier_canon":"Supplier"})
    )
    top_sup = sup_tot.sort_values("spend_eur", ascending=False).head(20).copy()
    top_sup["Spend (M€)"] = top_sup["spend_eur"] / 1_000_000
    top20_suppliers = top_sup["Supplier"].tolist()

    if PLOTLY and not top_sup.empty:
        xmax = max(0.5, float(top_sup["Spend (M€)"].max()) * 1.12)
        fig_b = px.bar(
            top_sup.sort_values("Spend (M€)", ascending=True),
            x="Spend (M€)", y="Supplier",
            orientation="h",
            text=top_sup.sort_values("Spend (M€)", ascending=True)["Spend (M€)"].map(lambda v: f"€ {v:.1f} M"),
            color_discrete_sequence=["#14b8a6"],
        )
        fig_b.update_traces(textposition="outside", cliponaxis=False)
        fig_b.update_layout(
            height=540,
            margin=dict(l=10,r=20,t=10,b=10),
            xaxis=dict(range=[0, xmax], gridcolor="#eef2f7"),
            yaxis=dict(automargin=True),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.dataframe(top_sup[["Supplier","spend_eur"]])

# ===================================
# ----- SUPPLIER × CATEGORY MIX -----
# ===================================

st.markdown('<div class="spacer-24"></div>', unsafe_allow_html=True)
st.markdown('<div class="block-title">Supplier × Category Mix (Top 20 suppliers)</div>', unsafe_allow_html=True)

mix = pd.DataFrame()
if top20_suppliers:
    mix = (
        df[df["supplier_canon"].isin(top20_suppliers)]
        .groupby(["supplier_canon", "category_resolved"])["_spend_eur_clean"]
        .sum()
        .reset_index()
        .rename(columns={"supplier_canon":"Supplier"})
    )

if PLOTLY and not mix.empty:
    totals = mix.groupby("Supplier")["_spend_eur_clean"].transform("sum")
    mix["share_pct"] = np.where(totals > 0, (mix["_spend_eur_clean"] / totals) * 100.0, 0.0)

    # force same order (top-20 bar is top -> bottom; for horizontal stacked we reverse)
    supplier_order_for_plot = list(reversed(top20_suppliers))
    mix["Supplier"] = pd.Categorical(mix["Supplier"], categories=supplier_order_for_plot, ordered=True)

    fig3 = px.bar(
        mix,
        x="share_pct",
        y="Supplier",
        color="category_resolved",
        orientation="h",
        barmode="stack",
        color_discrete_map=color_map,
    )
    # Legend below chart, generous margins to avoid overlap.
    fig3.update_layout(
        height=max(560, len(top20_suppliers)*26 + 180),
        margin=dict(l=10, r=40, t=10, b=120),
        legend=dict(orientation="h", y=-0.20, x=0, title="Category"),
        xaxis=dict(title="Share (%)", ticksuffix="%", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Supplier", automargin=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Mix chart will appear once Top-20 suppliers are available.")

# ===================================
# ----------- DATA QUALITY ----------
# ===================================

st.markdown('<div class="spacer-24"></div>', unsafe_allow_html=True)
st.markdown('<div class="block-title">Data Quality</div>', unsafe_allow_html=True)

unknown_ccy = int(df[ccy_col].isna().sum()) if ccy_col else 0
missing_price = int(df["_unit_price"].isna().sum())
missing_qty = int(df["_qty"].isna().sum())
zero_neg_price = int((df["_unit_price"] <= 0).fillna(False).sum())
zero_neg_qty = int((df["_qty"] <= 0).fillna(False).sum())
blank_supplier = int(df["supplier_canon"].eq("").sum())
blank_category = int(df["category_resolved"].eq("").sum())

st.markdown(
    f"""
    <div class="badge-row">
      <div class="badge">Unknown currency: {unknown_ccy}</div>
      <div class="badge">Missing unit price: {missing_price}</div>
      <div class="badge">Missing quantity: {missing_qty}</div>
      <div class="badge">Zero/negative price: {zero_neg_price}</div>
      <div class="badge bad">Zero/negative qty: {zero_neg_qty}</div>
      <div class="badge">Blank supplier: {blank_supplier}</div>
      <div class="badge">Blank category: {blank_category}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

issues_mask = (
    (df["_unit_price"].isna()) |
    (df["_qty"].isna()) |
    (df["_unit_price"] <= 0) |
    (df["_qty"] <= 0) |
    (df["supplier_canon"] == "") |
    (df["category_resolved"] == "")
)
dq_cols = ["supplier", "category_resolved"]
if ccy_col: dq_cols.append(ccy_col)
dq_cols += ["_unit_price","_qty","_spend_eur"]
issues = df.loc[issues_mask, dq_cols].rename(columns={
    "supplier":"Supplier", "category_resolved":"Category",
    "_unit_price":"Unit Price", "_qty":"Quantity", "_spend_eur":"Spend (EUR)"
})
if issues.empty:
    st.success("No blocking data quality issues detected.")
else:
    st.dataframe(issues, use_container_width=True)
