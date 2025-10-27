# ──────────────────────────────────────────────────────────────────────────────
# ProcureIQ — Spend Explorer (No-API keys; Yahoo Finance + EDGAR + Google News)
# ──────────────────────────────────────────────────────────────────────────────

import io, os, re, math, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import streamlit as st
import requests

# Charts
try:
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False

# Yahoo Finance (no key)
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

from rapidfuzz import process, fuzz

st.set_page_config(page_title="ProcureIQ — Spend Explorer", layout="wide")

# ============================== THEME / CSS (unchanged) =======================
P_PRIMARY = "#06b6d4"   # cyan-500
P_DEEP    = "#0ea5e9"   # sky-500
P_ACCENT  = "#8b5cf6"   # violet-500
P_TEXT    = "#0f172a"   # slate-900
P_TEXT2   = "#475569"   # slate-600
P_BORDER  = "#e2e8f0"   # slate-200
P_SOFTBG  = "#f8fafc"   # slate-50

st.markdown(
    f"""
    <style>
      .banner {{
        background: linear-gradient(135deg, rgba(14,165,233,.10), rgba(139,92,246,.06));
        border: 1px solid {P_BORDER};
        border-radius: 18px;
        padding: 20px 22px 18px 22px;
        margin: 6px 0 10px 0;
        box-shadow: 0 2px 6px rgba(2,8,23,.04);
      }}
      .app-title {{ font-size: 32px; font-weight: 800; letter-spacing: -.02rem; margin: 0; color: {P_TEXT}; }}
      .app-sub   {{ color: {P_TEXT2}; font-size: 14px; margin: 6px 0 0 0; }}

      .kpi-grid {{ display:grid; grid-template-columns: repeat(5, 1fr); gap:14px; margin:12px 0 0 0; }}
      .kpi-card {{
        background:#fff;border:1px solid {P_BORDER};border-radius:14px;padding:16px 18px;
        box-shadow:0 1px 3px rgba(0,0,0,.05); min-height:110px;
        display:flex; flex-direction:column; justify-content:center;
      }}
      .kpi-title {{ font-size:.95rem;color:{P_TEXT2};margin-bottom:8px; }}
      .kpi-value {{ font-size:1.8rem;font-weight:800;letter-spacing:-.02rem; white-space:nowrap; }}
      .kpi-unit  {{ font-weight:700; font-size:1.1rem; color:{P_TEXT2}; margin-left:.3rem; }}

      .block-title {{ font-weight:800; font-size:1.05rem; margin:6px 0 8px 6px; color:{P_TEXT}; }}
      .spacer-16 {{ margin-top:16px; }}
      .spacer-24 {{ margin-top:24px; }}

      .dq-row {{ display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 12px 0; }}
      .dq-pill {{
        display:flex; align-items:center; gap:10px;
        padding: 11px 13px; border-radius: 12px; border:1px solid {P_BORDER};
        background:#fff; min-width: 240px; height: 50px;
        box-shadow:0 1px 2px rgba(2,8,23,.03);
      }}
      .ok   {{ border-color:#bbf7d0; background:#ecfdf5; }}
      .warn {{ border-color:#fed7aa; background:#fff7ed; }}
      .bad  {{ border-color:#fecaca; background:#fef2f2; }}
      .dq-lbl {{ font-size: 13px; color: {P_TEXT2}; }}
      .dq-val {{ font-weight: 800; font-size: 16px; color: {P_TEXT}; }}

      .idea-bullet {{ line-height:1.45rem; }}
      .idea-tag {{
        display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px;
        border:1px solid {P_BORDER}; margin-right:6px; color:{P_TEXT2}
      }}

      [data-testid="stSidebar"] {{ min-width: 360px; max-width: 400px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================== HEADER (unchanged) ============================
st.markdown(
    """
    <div class="banner">
      <div class="app-title">ProcureIQ — Spend Explorer</div>
      <div class="app-sub">Upload your spend cube, map columns in the sidebar, pick the category source, and explore.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

BASE = "EUR"

# ============================== HELPERS (unchanged) ===========================
def normalize_headers(cols): return [re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()) for c in cols]
def parse_number_robust(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = re.sub(r"[^\d,\.\-]", "", str(x))
    if "," in s and "." in s: s = s.replace(",", "")
    elif "," in s and s.count(",")==1 and len(s.split(",")[-1]) in (2,3): s = s.replace(",", ".")
    else: s = s.replace(",", "")
    try: return float(s)
    except: return np.nan
def ensure_numeric_spend(s: pd.Series) -> pd.Series:
    if s.dtype == bool: s = s.astype(float)
    elif s.dtype.kind in ("i","u","f"): s = s.astype(float)
    else: s = pd.to_numeric(s, errors="coerce")
    return s.fillna(0.0).astype(float)

CURRENCY_SYMBOL_MAP = {"€":"EUR","$":"USD","£":"GBP","¥":"JPY","₩":"KRW","₹":"INR","₺":"TRY","R$":"BRL","S$":"SGD"}
ISO_3 = {"EUR","USD","GBP","JPY","CNY","CHF","SEK","NOK","DKK","PLN","HUF","CZK","RON","AUD","NZD","CAD","MXN","BRL","ZAR","AED","SAR","HKD","SGD","INR","TRY","KRW","TWD","THB","PHP","ILS","VND","NGN","RUB"}
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

TARGETS = {
    "category":   ["category","category name","spend category","commodity","material group","item family","item family group"],
    "cat_family": ["item family","family","family name"],
    "cat_group":  ["item family group","family group","group","group name","material group"],
    "supplier":   ["supplier","supplier name","vendor","vendor name","seller","payee"],
    "currency":   ["currency","ccy","currency code","iso currency","transaction currency","base curr"],
    "unit_price": ["unit price","unit cost","net price","unit po price","unit price (base curr)","unit price (global curr)"],
    "quantity":   ["quantity","qty","order qty","qty ordered","units","volume","po qty","po quantity","ordered qty"],
    "amount":     ["amount","purchase amount","line amount","total value","net value","extended price"]
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

SPEND_NAME_CUES = ["purchase amount","po amount","line total","total value","net value","extended price","base curr","global curr","reporting curr"]
def detect_spend_column(df):
    hits = [c for c in df.columns if any(k in c.lower() for k in SPEND_NAME_CUES)]
    if not hits: return None
    if len(hits)==1: return hits[0]
    med = {c: pd.to_numeric(df[c].apply(parse_number_robust), errors="coerce").median(skipna=True) for c in hits}
    return max(med, key=med.get)

def fmt_k(s: pd.Series) -> pd.Series: return (s/1_000.0).round(0)

def detect_part_number_cols(df_in):
    norm = {c: c.lower() for c in df_in.columns}
    cues = ["item", "item number", "item no", "material", "material number", "sku", "code", "part", "pn", "material code"]
    hits = []
    for c, n in norm.items():
        if any(k in n for k in cues):
            hits.append(c)
    hits = [c for c in hits if c not in ["category_resolved"]]
    hits = sorted(hits, key=lambda x: 0 if "item" in x.lower() else (1 if "material" in x.lower() else 2))
    return hits

# ============================== UPLOAD (unchanged) ============================
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])

if not uploaded:
    st.info("Upload a file in the sidebar to start.")
    st.stop()

raw = pd.read_excel(uploaded)
raw.columns = [str(c) for c in raw.columns]

# ============================== MAPPING (unchanged) ===========================
mapping = suggest_columns(raw)
with st.sidebar:
    st.subheader("Column mapping")
    if "supplier" not in mapping:
        mapping["supplier"] = st.selectbox("Supplier / Vendor", options=raw.columns, key="sup_pick")
    if "currency" not in mapping:
        mapping["currency"] = st.selectbox("Currency", options=raw.columns, key="cur_pick")
    if "unit_price" not in mapping:
        mapping["unit_price"] = st.selectbox("Unit Price (per item)", options=raw.columns, key="up_pick")
    if "quantity" not in mapping:
        mapping["quantity"] = st.selectbox("Quantity", options=raw.columns, key="qty_pick")

# Optional ticker mapping upload (no changes elsewhere)
with st.sidebar:
    st.caption("Optional: upload supplier→ticker map (CSV: supplier,ticker)")
    map_file = st.file_uploader("supplier_tickers.csv", type=["csv"], key="sup_ticker_map", accept_multiple_files=False)

def load_user_ticker_map(file):
    try:
        m = pd.read_csv(file)
        m.columns = [c.strip().lower() for c in m.columns]
        if "supplier" in m.columns and "ticker" in m.columns:
            m["supplier_norm"] = m["supplier"].astype(str).str.strip().str.lower()
            return dict(zip(m["supplier_norm"], m["ticker"].astype(str).str.strip()))
    except Exception:
        pass
    return {}
USER_TICKER_MAP = load_user_ticker_map(map_file) if map_file else {}

df = raw.copy()
for k in ["supplier","currency","unit_price","quantity","category","cat_family","cat_group"]:
    if k in mapping:
        df[k] = raw[mapping[k]]

# Category candidates (unchanged)
cat_cands, label_to_col = [], {}
def _add_cand(label, colname):
    if colname and colname in df.columns and df[colname].notna().any():
        cat_cands.append(label); label_to_col[label]=colname
_add_cand("Category", mapping.get("category"))
_add_cand("Item Family", mapping.get("cat_family"))
_add_cand("Item Family Group", mapping.get("cat_group"))
if mapping.get("cat_family") in df.columns and mapping.get("cat_group") in df.columns:
    fam = df[mapping["cat_family"]].astype(str).str.strip().replace({"nan":""})
    grp = df[mapping["cat_group"]].astype(str).str.strip().replace({"nan":""})
    df["__family_plus_group__"] = (fam + " - " + grp).str.strip(" -")
    if df["__family_plus_group__"].replace("", np.nan).notna().any():
        cat_cands.append("Family + Group")
        label_to_col["Family + Group"] = "__family_plus_group__"

with st.sidebar:
    if not cat_cands:
        st.warning("No category-like columns detected. Pick one:")
        user_cat = st.selectbox("Pick a Category column", options=raw.columns, key="cat_any")
        label_to_col["Category"] = user_cat
        cat_cands = ["Category"]

with st.sidebar:
    st.subheader("Category source")
    cat_choice = st.radio("Use this as Category:", cat_cands, index=0)
resolved_cat_col = label_to_col[cat_choice]
df["category_resolved"] = df[resolved_cat_col].copy()

# ============================== FX + SPEND (unchanged) ========================
df["unit_price"] = df["unit_price"].apply(parse_number_robust)
df["quantity"]   = df["quantity"].apply(parse_number_robust)

fx = load_latest_ecb()
df["currency"] = df["currency"].fillna("")
df = apply_fx_latest(df, fx)
df["unit_price_eur"] = pd.to_numeric(df["unit_price"], errors="coerce") * df["rate_to_eur"]

spend_col = detect_spend_column(raw)
with st.sidebar:
    if spend_col: st.success(f"Detected **'{spend_col}'** as spend column.")
    else: st.warning("No clear spend column found; you can use Unit×Qty×FX instead.")

def _is_global_header(c):
    return any(k in c.lower() for k in ["base curr","base currency","global curr","reporting curr"])
def _find_base_ccy_col(df_raw, spend_col_name):
    cands=[c for c in df_raw.columns if c!=spend_col_name and any(k in c.lower() for k in
        ["base curr code","base currency code","base curr","base currency","reporting curr","global curr"])]
    return cands[0] if cands else None

spend_detected = pd.Series(np.nan, index=df.index)
if spend_col:
    s = raw[spend_col].apply(parse_number_robust)
    if _is_global_header(spend_col):
        base_col = _find_base_ccy_col(raw, spend_col)
        if base_col:
            base_iso = raw[base_col].astype(str).apply(detect_iso_from_text).fillna("EUR")
            fx_map  = dict(zip(fx["currency"], fx["rate_to_eur"]))
            base_rt = base_iso.map(lambda c: fx_map.get(c,1.0))
            spend_detected = pd.to_numeric(s, errors="coerce") * base_rt
        else:
            spend_detected = pd.to_numeric(s, errors="coerce")
    else:
        spend_detected = pd.to_numeric(s, errors="coerce") * df["rate_to_eur"]

spend_calc = (pd.to_numeric(df["unit_price"], errors="coerce") *
              pd.to_numeric(df["quantity"], errors="coerce") *
              df["rate_to_eur"])

with st.sidebar:
    st.subheader("Spend source")
    mode = st.radio("Choose:", ["Use detected spend column","Use Unit×Qty×FX","Auto-validate"], index=2)

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
    if _mostly_zero(spend_calc): use_calc = False
    df["_spend_eur"] = (spend_calc if use_calc else spend_detected).fillna(0.0)

df["_spend_eur"] = ensure_numeric_spend(df["_spend_eur"])

# ============================== AGGREGATES (unchanged) ========================
cat = (df.groupby("category_resolved", dropna=False)
       .agg(spend_eur=("_spend_eur","sum"),
            lines=("category_resolved","count"),
            suppliers=("supplier", pd.Series.nunique))
       .reset_index()
       .rename(columns={"category_resolved":"Category","lines":"# PO Lines","suppliers":"# Suppliers"}))
cat["Spend (€ k)"] = fmt_k(cat["spend_eur"])

sup_tot = (df.groupby("supplier", dropna=False)
           .agg(spend_eur=("_spend_eur","sum"),
                lines=("supplier","count"),
                categories=("category_resolved", pd.Series.nunique))
           .reset_index()
           .rename(columns={"supplier":"Supplier"}))
sup_tot["Spend (€ k)"] = fmt_k(sup_tot["spend_eur"])

# Part numbers (unchanged)
part_cols = detect_part_number_cols(raw)
part_count = 0
part_id_col = None
if part_cols:
    for c in part_cols:
        if raw[c].notna().sum() > 0:
            part_id_col = c
            break
    if part_id_col:
        part_count = int(raw[part_id_col].astype(str).replace({"nan":np.nan,"":np.nan}).nunique())

# ============================== NAV (unchanged) ===============================
page = st.sidebar.radio("Navigation", ["Dashboard","Deep Dives"], index=0)

# ============================== DASHBOARD (unchanged) =========================
if page == "Dashboard":
    total_spend = float(df["_spend_eur"].sum())
    total_lines = int(len(df))
    total_suppliers = int(df["supplier"].nunique())
    total_categories = int(df["category_resolved"].nunique())

    # KPI row
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    st.markdown(f'''
        <div class="kpi-card"><div class="kpi-title">Total Spend</div>
        <div class="kpi-value">€ {total_spend/1_000_000:,.1f}<span class="kpi-unit">M</span></div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="kpi-card"><div class="kpi-title">Categories</div>
        <div class="kpi-value">{total_categories:,}</div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="kpi-card"><div class="kpi-title">Suppliers</div>
        <div class="kpi-value">{total_suppliers:,}</div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="kpi-card"><div class="kpi-title">Part Numbers</div>
        <div class="kpi-value">{part_count:,}</div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="kpi-card"><div class="kpi-title">PO Lines</div>
        <div class="kpi-value">{total_lines:,}</div></div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="spacer-16"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 2.2], gap="large")

    # Donut
    with left:
        st.markdown('<div class="block-title">Spend by Category</div>', unsafe_allow_html=True)
        donut_raw = (df.groupby("category_resolved", dropna=False)["_spend_eur"].sum()
                       .reset_index().rename(columns={"category_resolved":"Category","_spend_eur":"spend_eur"}))
        donut_raw = donut_raw[donut_raw["spend_eur"]>0].sort_values("spend_eur", ascending=False)
        defaultN = 10 if len(donut_raw)>=10 else len(donut_raw)
        st.slider("Top categories in donut", 5, min(15, len(donut_raw)), defaultN, key="donutN")

        if donut_raw.empty or donut_raw["spend_eur"].nunique(dropna=True) <= 1:
            st.error("Category spend looks degenerate. Check **Category source** in the sidebar.")
            st.dataframe(donut_raw.rename(columns={"spend_eur":"Spend (EUR)"}), use_container_width=True)
            color_map = {}
        else:
            topN_df = donut_raw.head(st.session_state.donutN)
            other = float(donut_raw["spend_eur"].iloc[st.session_state.donutN:].sum())
            donut_df = pd.concat([topN_df, pd.DataFrame([{"Category":"Other","spend_eur":other}]) if other>0 else pd.DataFrame()], ignore_index=True)
            if PLOTLY:
                palette = px.colors.qualitative.Set3
                cats_in_palette = donut_df["Category"].tolist()
                color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats_in_palette)}
                fig = px.pie(donut_df, names="Category", values="spend_eur",
                             hole=.45, color="Category", color_discrete_map=color_map)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(height=520, margin=dict(l=0,r=0,t=0,b=0),
                                  legend=dict(orientation="h", y=-0.16))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(donut_df, use_container_width=True)

    # Top suppliers bar
    with right:
        st.markdown('<div class="block-title">Top Suppliers by Spend</div>', unsafe_allow_html=True)
        top_sup = sup_tot.sort_values("spend_eur", ascending=False).head(20).copy()
        if PLOTLY and not top_sup.empty:
            top_sup["Spend_M"] = top_sup["spend_eur"]/1_000_000.0
            fig2 = px.bar(top_sup, x="Spend_M", y="Supplier", orientation="h",
                          text=top_sup["Spend_M"].map(lambda v: f"€ {v:,.1f} M"),
                          labels={"Spend_M":"Spend (€ M)"},
                          color_discrete_sequence=[P_PRIMARY])
            fig2.update_traces(textposition="outside", cliponaxis=False)
            fig2.update_layout(height=520, margin=dict(l=10,r=140,t=0,b=10),
                               yaxis=dict(categoryorder="total ascending", automargin=True, ticksuffix="  "),
                               xaxis=dict(title="", showgrid=True, zeroline=False, gridcolor="#e5e7eb"),
                               plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No supplier spend to plot yet.")

    # Supplier × Category Mix (Top 20)
    st.markdown('<div class="spacer-24"></div>', unsafe_allow_html=True)
    st.markdown('<div class="block-title">Supplier × Category Mix (Top 20 suppliers)</div>', unsafe_allow_html=True)

    top20_suppliers = top_sup["Supplier"].tolist() if 'top_sup' in locals() and not top_sup.empty else []
    if top20_suppliers:
        top20_order_for_mix = (
            sup_tot[sup_tot["Supplier"].isin(top20_suppliers)]
            .sort_values("spend_eur", ascending=True)["Supplier"].tolist()
        )
    else:
        top20_order_for_mix = []

    mix = pd.DataFrame()
    if top20_order_for_mix:
        mix = (
            df[df["supplier"].isin(top20_order_for_mix])]
            .groupby(["supplier","category_resolved"])["_spend_eur"].sum()
            .reset_index()
            .rename(columns={"supplier":"Supplier"})
        )

    if PLOTLY and not mix.empty:
        totals = mix.groupby("Supplier")["_spend_eur"].transform("sum")
        mix["share_pct"] = np.where(totals>0, (mix["_spend_eur"]/totals)*100.0, 0.0)
        mix["Supplier"] = pd.Categorical(mix["Supplier"], categories=top20_order_for_mix, ordered=True)

        if not 'color_map' in locals() or not color_map:
            palette = px.colors.qualitative.Set3
            all_cats = sorted(df["category_resolved"].dropna().unique().tolist())
            color_map = {c: palette[i % len(palette)] for i, c in enumerate(all_cats)}

        fig3 = px.bar(mix, x="share_pct", y="Supplier", color="category_resolved",
                      orientation="h", barmode="stack", color_discrete_map=color_map)
        fig3.update_layout(height=max(560, len(top20_order_for_mix)*26 + 180),
                           margin=dict(l=10, r=40, t=10, b=120),
                           legend=dict(orientation="h", y=-0.20, x=0, title="Category"),
                           xaxis=dict(title="Share (%)", ticksuffix="%", showgrid=True, gridcolor="#f1f5f9"),
                           yaxis=dict(title="Supplier", automargin=True, categoryorder="array",
                                      categoryarray=top20_order_for_mix),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Mix chart will appear once the Top-20 suppliers are available.")

    # Data Quality (unchanged)
    st.markdown('<div class="spacer-24"></div>', unsafe_allow_html=True)
    st.subheader("Data Quality")

    unknown_ccy    = df["currency_iso"].isna() | (df["currency_iso"]=="")
    missing_price  = df["unit_price"].isna()
    missing_qty    = df["quantity"].isna()
    zero_neg_price = (pd.to_numeric(df["unit_price"], errors="coerce")<=0)
    zero_neg_qty   = (pd.to_numeric(df["quantity"], errors="coerce")<=0)
    blank_supplier = (df["supplier"].astype(str).str.strip()=="")
    blank_category = (df["category_resolved"].astype(str).str.strip()=="")

    dq = [
        ("Unknown currency", int(unknown_ccy.sum())),
        ("Missing unit price", int(missing_price.sum())),
        ("Missing quantity", int(missing_qty.sum())),
        ("Zero/negative price", int(zero_neg_price.sum())),
        ("Zero/negative qty", int(zero_neg_qty.sum())),
        ("Blank supplier", int(blank_supplier.sum())),
        ("Blank category", int(blank_category.sum())),
    ]

    st.markdown('<div class="dq-row">', unsafe_allow_html=True)
    for label, val in dq:
        cls = "ok" if val==0 else ("warn" if val<10 else "bad")
        st.markdown(f'''
          <div class="dq-pill {cls}">
            <div class="dq-lbl">{label}</div>
            <div class="dq-val">{val}</div>
          </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    issues_mask = (unknown_ccy | missing_price | missing_qty |
                   zero_neg_price | zero_neg_qty | blank_supplier | blank_category)
    sample_cols = [c for c in ["supplier","category_resolved","currency","unit_price","quantity","_spend_eur"]
                   if c in df.columns]
    issues = df.loc[issues_mask, sample_cols].head(200)
    if not issues.empty:
        st.dataframe(issues.rename(columns={
            "supplier":"Supplier","category_resolved":"Category","currency":"Currency",
            "unit_price":"Unit Price","quantity":"Quantity","_spend_eur":"Spend (EUR)"}), use_container_width=True)
    else:
        st.success("No obvious data quality issues found in key fields.")

# ============================== DEEP DIVES (no-key enrich) ====================
else:
    st.subheader("Deep Dives")

    # Category picker
    categories = sorted([c for c in df["category_resolved"].dropna().unique().tolist() if str(c).strip() != ""])
    if not categories:
        st.info("No categories available.")
        st.stop()
    sel_cat = st.selectbox("Select category", categories, index=0)

    df_cat = df[df["category_resolved"] == sel_cat].copy()
    sup_cat = (df_cat.groupby("supplier", dropna=False)
               .agg(spend_eur=("_spend_eur","sum"), lines=("supplier","count"))
               .reset_index())

    # parts per supplier in this category
    if part_id_col and part_id_col in df_cat.columns:
        pn = (df_cat.groupby("supplier")[part_id_col]
              .nunique().reset_index().rename(columns={part_id_col:"# Parts"}))
        sup_cat = sup_cat.merge(pn, on="supplier", how="left")
    else:
        sup_cat["# Parts"] = 0
    sup_cat["# Parts"] = sup_cat["# Parts"].fillna(0).astype(int)
    sup_cat = sup_cat.sort_values("spend_eur", ascending=False)

    left, right = st.columns([1.7, 1.3], gap="large")

    # ---- left: supplier list bar
    with left:
        st.markdown(f'<div class="block-title">Suppliers in “{sel_cat}” (by spend)</div>', unsafe_allow_html=True)
        if PLOTLY and not sup_cat.empty:
            tmp = sup_cat.copy()
            tmp["Spend_M"] = tmp["spend_eur"]/1_000_000.0
            fig = px.bar(tmp, x="Spend_M", y="supplier", orientation="h",
                         text=tmp["Spend_M"].map(lambda v: f"€ {v:,.1f} M"),
                         labels={"Spend_M":"Spend (€ M)", "supplier":"Supplier"},
                         color_discrete_sequence=[P_PRIMARY])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(height=max(420, 22*len(tmp)+80),
                              margin=dict(l=10,r=140,t=0,b=10),
                              yaxis=dict(categoryorder="total ascending", automargin=True),
                              xaxis=dict(title="", showgrid=True, gridcolor="#e5e7eb"),
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    # ---- right: supplier overview (Yahoo Finance -> EDGAR -> OpenCorporates)
    with right:
        st.markdown('<div class="block-title">Supplier overview</div>', unsafe_allow_html=True)
        sup_list = sup_cat["supplier"].tolist()
        sel_sup = st.selectbox("Select a supplier for details", sup_list, index=0)

        # ---------- financials w/out keys ----------
        def _normalize_name(x:str)->str: return str(x or "").strip().lower()

        def get_ticker_for_supplier(name:str, user_map:dict) -> str|None:
            n = _normalize_name(name)
            if n in user_map: return user_map[n]
            return None  # rely on uploaded mapping for precision

        def _safe_number(x):
            try:
                if x is None or (isinstance(x,float) and math.isnan(x)): return None
                return float(x)
            except Exception:
                return None

        def fx_convert(amount, from_ccy:str) -> float|None:
            if amount is None: return None
            if not from_ccy:  return amount
            c = str(from_ccy).upper().strip()
            row = fx.loc[fx["currency"]==c]
            if row.empty: return amount
            rate = row["rate_to_eur"].iloc[0]
            try: return float(amount) * float(rate)
            except Exception: return amount

        # Yahoo Finance
        def fin_from_yf(ticker:str) -> dict|None:
            if not HAVE_YF: return None
            try:
                t = yf.Ticker(ticker)
                info = t.info or {}
                revenue = _safe_number(info.get("totalRevenue"))
                margin  = _safe_number(info.get("profitMargins"))
                if margin is not None and margin < 1.0:
                    margin = 100.0 * margin
                ccy     = info.get("financialCurrency")
                if revenue is None and margin is None: return None
                return {"provider":"YahooFinance","currency":ccy,"revenue":revenue,"margin_pct":margin}
            except Exception:
                return None

        # SEC EDGAR (no key)
        @st.cache_data(ttl=24*60*60, show_spinner=False)
        def _sec_ticker_table():
            try:
                url = "https://www.sec.gov/files/company_tickers.json"
                r = requests.get(url, headers={"User-Agent":"ProcureIQ/1.0"}, timeout=20)
                r.raise_for_status()
                data = r.json()
                rows = []
                for _, v in data.items():
                    rows.append({"cik": str(v["cik_str"]).zfill(10),
                                 "ticker": str(v["ticker"]).upper().strip(),
                                 "name": v["title"]})
                return pd.DataFrame(rows)
            except Exception:
                return pd.DataFrame(columns=["cik","ticker","name"])

        def _clean_name(n: str) -> str:
            n = re.sub(r"[,\.]", " ", str(n))
            n = re.sub(r"\b(incorporated|inc|ltd|limited|corp|corporation|plc|sa|ag|nv|co|company)\b", "", n, flags=re.I)
            n = re.sub(r"\s+", " ", n).strip()
            return n

        def _closest_sec_match(name: str, sec_df: pd.DataFrame, min_score=84):
            if sec_df is None or sec_df.empty or not isinstance(name, str): return None
            q = _clean_name(name)
            cand_name = process.extractOne(q, (_clean_name(x) for x in sec_df["name"].tolist()), scorer=fuzz.WRatio)
            best = None
            if cand_name and cand_name[1] >= min_score:
                row = sec_df.loc[sec_df["name"].apply(_clean_name) == cand_name[0]].head(1)
                if len(row):
                    r = row.iloc[0]
                    best = {"cik": str(r["cik"]).zfill(10), "ticker": r["ticker"], "name": r["name"]}
            return best

        def fin_from_edgar(company_name: str) -> dict|None:
            try:
                sec = _sec_ticker_table()
                match = _closest_sec_match(company_name, sec)
                if match is None: return None
                cik = match["cik"]
                url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
                headers = {"User-Agent": "ProcureIQ/1.0 (contact: procurement@example.com)"}
                r = requests.get(url, headers=headers, timeout=25)
                if r.status_code != 200: return None
                data = r.json()

                def _latest_for_any(taxonomy: str, concepts: list, preferred_unit="USD"):
                    for concept in concepts:
                        try:
                            units = data["facts"][taxonomy][concept]["units"]
                        except Exception:
                            continue
                        series = None
                        if preferred_unit and preferred_unit in units:
                            series = units[preferred_unit]
                        elif len(units):
                            series = next(iter(units.values()))
                        if not series:
                            continue
                        series = sorted(series, key=lambda x: (x.get("end",""), x.get("fy",""), x.get("period","")))
                        if series:
                            val = series[-1].get("val", None)
                            try: return float(val)
                            except Exception: continue
                    return None

                revenue_candidates = [
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet","Revenues","Revenue","TotalRevenuesAndOtherIncome",
                ]
                income_candidates = [
                    "NetIncomeLoss","ProfitLoss",
                    "NetIncomeLossAvailableToCommonStockholdersBasic",
                    "NetIncomeLossIncludingPortionAttributableToNoncontrollingInterest",
                ]

                rev_usd = _latest_for_any("us-gaap", revenue_candidates, preferred_unit="USD")
                ni_usd  = _latest_for_any("us-gaap", income_candidates,  preferred_unit="USD")
                if rev_usd is None:
                    rev_usd = _latest_for_any("us-gaap", revenue_candidates, preferred_unit=None)
                if ni_usd is None:
                    ni_usd  = _latest_for_any("us-gaap", income_candidates,  preferred_unit=None)

                if rev_usd is None and ni_usd is None: return None
                margin = None
                if (rev_usd is not None) and (ni_usd is not None) and rev_usd != 0:
                    margin = 100.0 * (ni_usd / rev_usd)
                return {"provider":"EDGAR","currency":"USD","revenue":rev_usd,"margin_pct":margin}
            except Exception:
                return None

        def fin_from_opencorps(name:str) -> dict|None:
            try:
                q = requests.get("https://api.opencorporates.com/v0.4/companies/search",
                                 params={"q": name, "per_page":1}, timeout=10)
                if q.status_code != 200: return None
                js = q.json()
                if js.get("results",{}).get("companies"):
                    return {"provider":"OpenCorporates"}
                return None
            except Exception:
                return None

        def enrich_financials_for_supplier(supplier_name:str) -> dict:
            # try explicit ticker mapping first
            ticker = None
            n = str(supplier_name or "").strip().lower()
            if USER_TICKER_MAP and n in USER_TICKER_MAP:
                ticker = USER_TICKER_MAP[n]

            # 1) Yahoo Finance if we have a ticker
            if ticker:
                yf_fin = fin_from_yf(ticker)
                if yf_fin:
                    rev_eur = fx_convert(yf_fin.get("revenue"), yf_fin.get("currency"))
                    return {"provider":"YahooFinance", "ticker":ticker,
                            "revenue_eur_million": (rev_eur/1_000_000.0) if rev_eur is not None else None,
                            "margin_pct": yf_fin.get("margin_pct")}

            # 2) EDGAR by name (for US filers, no key)
            edg = fin_from_edgar(supplier_name)
            if edg:
                rev_eur = fx_convert(edg.get("revenue"), edg.get("currency"))
                return {"provider":"EDGAR", "ticker":None,
                        "revenue_eur_million": (rev_eur/1_000_000.0) if rev_eur is not None else None,
                        "margin_pct": edg.get("margin_pct")}

            # 3) OpenCorporates presence as last resort (no numbers)
            oc = fin_from_opencorps(supplier_name)
            if oc:
                return {"provider":"OpenCorporates", "ticker":None,
                        "revenue_eur_million": None, "margin_pct": None}

            # Nothing
            return {"provider": None, "ticker": None, "revenue_eur_million": None, "margin_pct": None}

        fin = enrich_financials_for_supplier(sel_sup)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Revenue (EUR million)")
            st.markdown(f"### {('N/A' if fin['revenue_eur_million'] is None else f'{fin['revenue_eur_million']:.0f}')}")

        with c2:
            st.caption("Margin (%)")
            st.markdown(f"### {('N/A' if fin['margin_pct'] is None else f'{fin['margin_pct']:.1f}%')}")

        prov = fin.get("provider")
        if prov:
            st.info(f"Financials source: **{prov}**" + (f" · Ticker: `{fin.get('ticker')}`" if fin.get("ticker") else ""))
        else:
            st.warning("No reliable public financials found with keyless sources for this supplier.")

    # ------------------------ Research & Negotiation Ideas (no keys) ----------
    st.markdown('<div class="spacer-24"></div>', unsafe_allow_html=True)
    st.subheader("Research-backed negotiation & cost-reduction ideas")

    # Google News RSS (no key)
    @st.cache_data(ttl=2*60*60, show_spinner=False)
    def google_news(query: str, max_items=8):
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent":"ProcureIQ/1.0"})
            r.raise_for_status()
            root = ET.fromstring(r.content)
            items = []
            for item in root.iterfind(".//item")[:max_items]:
                title = item.findtext("title") or ""
                link  = item.findtext("link") or ""
                items.append({"title": title, "link": link})
            return items
        except Exception:
            return []

    df_sel = df_cat[df_cat["supplier"] == sel_sup].copy()
    parts_n  = int(df_sel[part_id_col].nunique()) if (part_id_col and part_id_col in df_sel.columns) else 0
    avg_price = (df_sel["_spend_eur"].sum() / max(1.0, df_sel["quantity"].sum())) if df_sel["quantity"].sum() else None

    q_news  = f'{sel_sup} {sel_cat} supplier'
    news = google_news(q_news, 8)

    i1, i2 = st.columns([1.25, 1])
    with i1:
        st.markdown("##### Market & price checks")
        if avg_price:
            st.markdown(f"<span class='idea-tag'>internal</span> Approx. **avg unit price** in your data for this supplier/category: **€{avg_price:,.4f}**", unsafe_allow_html=True)
        else:
            st.markdown("<span class='idea-tag'>internal</span> Could not compute average unit price (missing qty).", unsafe_allow_html=True)

        # provide Google search links (no key) the user can click
        def google_link(label, q):
            url = f"https://www.google.com/search?q={requests.utils.quote(q)}"
            st.markdown(f"- [{label}]({url})")

        st.markdown("**Benchmark search links**")
        google_link("Commodity/index pricing", f"{sel_cat} price index benchmark supplier cost")
        google_link("Should-cost estimators", f"{sel_cat} should-cost model manufacturing conversion overhead scrap")
        google_link("Regional price parity", f"{sel_cat} component price EU vs Asia logistics duty")

        st.markdown("##### VAVE levers (search links)")
        google_link("VAVE levers list", f"{sel_cat} value analysis value engineering cost reduction ideas")
        google_link("Material standardization", f"{sel_cat} standardize material grades commonize parts")
        google_link("Design-to-cost", f"{sel_cat} design to cost thickness optimization alternate materials")

    with i2:
        st.markdown("##### Recent news (potential negotiation angles)")
        if news:
            for n in news:
                st.markdown(f"- [{n['title']}]({n['link']})")
        else:
            st.info("No recent Google-News items fetched right now.")

    # Data-driven bullets
    st.markdown("##### Data-driven opportunities (from your cube)")
    bullets = []
    supplier_total = float(df_sel["_spend_eur"].sum())
    cat_total      = float(df_cat["_spend_eur"].sum())
    if supplier_total and cat_total:
        share = 100.0 * supplier_total / cat_total
        bullets.append(f"<span class='idea-tag'>mix</span> Supplier accounts for **{share:,.1f}%** of spend in *{sel_cat}*. Explore step pricing and volume-for-price trades.")
    if parts_n > 0:
        bullets.append(f"<span class='idea-tag'>complexity</span> **{parts_n} part numbers** with {sel_sup}. Check standardisation, consolidation, tooling amortisation.")
    if avg_price:
        bullets.append("<span class='idea-tag'>should-cost</span> Build a should-cost using material index + conversion (machine-hour), scrap & overhead; compare with benchmark search links above.")
    if len(bullets)==0:
        bullets.append("<span class='idea-tag'>hint</span> Add quantities/part IDs to unlock more specific ideas.")
    for b in bullets:
        st.markdown(f"- <span class='idea-bullet'>{b}</span>", unsafe_allow_html=True)

# ============================== DOWNLOADS (unchanged) =========================
st.divider()
st.markdown("#### Download full dataset & summaries (XLSX)")
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    df.to_excel(w, index=False, sheet_name="Lines")
    cat.to_excel(w, index=False, sheet_name="Categories")
    sup_tot.to_excel(w, index=False, sheet_name="Suppliers")
    (df.groupby(["supplier","category_resolved"])["_spend_eur"].sum()
       .reset_index().rename(columns={"supplier":"Supplier","category_resolved":"Category","_spend_eur":"Spend (EUR)"})
       .to_excel(w, index=False, sheet_name="Supplier_x_Category"))
st.download_button(
    "Download results.xlsx", buf.getvalue(), "procurement_diagnostics_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
