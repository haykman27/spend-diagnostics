# ──────────────────────────────────────────────────────────────────────────────
# ProcureIQ — Spend Explorer (keep existing UI; fix Supplier×Category Mix order)
# ──────────────────────────────────────────────────────────────────────────────
 
import io, re
import numpy as np
import pandas as pd
import streamlit as st
 
# Charts
try:
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False
 
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
 
      /* KPI row (same look as before: single horizontal row, equal sizes) */
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
 
      /* Data Quality pills (unchanged) */
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
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"  # ← fixed: quoted string
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
 
def detect_part_number_cols(df):
    norm = {c: c.lower() for c in df.columns}
    cues = ["item", "item number", "item no", "material", "material number", "sku", "code", "part", "pn", "material code"]
    hits = []
    for c, n in norm.items():
        if any(k in n for k in cues):
            hits.append(c)
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
 
# Part numbers (unchanged logic you approved earlier)
def detect_part_number_cols(df_in):
    norm = {c: c.lower() for c in df_in.columns}
    cues = ["item", "item number", "item no", "material", "material number", "sku", "code", "part", "pn", "material code"]
    hits = []
    for c, n in norm.items():
        if any(k in n for k in cues):
            hits.append(c)
    hits = [c for c in hits if c not in [mapping.get("cat_family"), mapping.get("cat_group"), resolved_cat_col]]
    return hits
part_cols = detect_part_number_cols(raw)
part_count = 0
if part_cols:
    chosen = None
    for c in part_cols:
        if raw[c].notna().sum() > 0:
            chosen = c; break
    if chosen:
        part_count = int(raw[chosen].astype(str).replace({"nan":np.nan,"":np.nan}).nunique())
 
# ============================== NAV (unchanged) ===============================
page = st.sidebar.radio("Navigation", ["Dashboard","Deep Dives"], index=0)
 
# ============================== DASHBOARD (unchanged except KPI layout) =======
if page == "Dashboard":
    total_spend = float(df["_spend_eur"].sum())
    total_lines = int(len(df))
    total_suppliers = int(df["supplier"].nunique())
    total_categories = int(df["category_resolved"].nunique())
 
    # === KPI row — CHANGED TO HORIZONTAL EQUAL-WIDTH COLUMNS (only layout change) ===
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1:
        st.markdown(f'''
            <div class="kpi-card">
              <div class="kpi-title">Total Spend</div>
              <div class="kpi-value">€ {total_spend/1_000_000:,.1f}<span class="kpi-unit">M</span></div>
            </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
            <div class="kpi-card">
              <div class="kpi-title">Categories</div>
              <div class="kpi-value">{total_categories:,}</div>
            </div>
        ''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
            <div class="kpi-card">
              <div class="kpi-title">Suppliers</div>
              <div class="kpi-value">{total_suppliers:,}</div>
            </div>
        ''', unsafe_allow_html=True)
    with c4:
        st.markdown(f'''
            <div class="kpi-card">
              <div class="kpi-title">Part Numbers</div>
              <div class="kpi-value">{part_count:,}</div>
            </div>
        ''', unsafe_allow_html=True)
    with c5:
        st.markdown(f'''
            <div class="kpi-card">
              <div class="kpi-title">PO Lines</div>
              <div class="kpi-value">{total_lines:,}</div>
            </div>
        ''', unsafe_allow_html=True)
    # === END KPI change ===
 
    st.markdown('<div class="spacer-16"></div>', unsafe_allow_html=True)
 
    # Donut left, bar right (unchanged titles/separation)
    left, right = st.columns([1.05, 2.2], gap="large")
 
    # -------- Donut (unchanged)
    with left:
        st.markdown('<div class="block-title">Spend by Category</div>', unsafe_allow_html=True)
        donut_raw = (df.groupby("category_resolved", dropna=False)["_spend_eur"].sum()
                       .reset_index().rename(columns={"category_resolved":"Category","_spend_eur":"spend_eur"}))
        donut_raw = donut_raw[donut_raw["spend_eur"]>0].sort_values("spend_eur", ascending=False)
 
        topN_default = 10 if len(donut_raw) >= 10 else len(donut_raw)
        st.slider("Top categories in donut", 5, min(15, len(donut_raw)), topN_default, key="donutN")
 
        if donut_raw.empty or donut_raw["spend_eur"].nunique(dropna=True) <= 1:
            st.error("Category spend looks degenerate. Check **Category source** in the sidebar.")
            st.dataframe(donut_raw.rename(columns={"spend_eur":"Spend (EUR)"}), use_container_width=True)
            color_map = {}
        else:
            topN_df = donut_raw.head(st.session_state.donutN)
            other = float(donut_raw["spend_eur"].iloc[st.session_state.donutN:].sum())
            donut_df = pd.concat([topN_df, pd.DataFrame([{"Category":"Other","spend_eur":other}]) if other>0 else pd.DataFrame()], ignore_index=True)
 
            # color map used ALSO by Supplier×Category Mix (unchanged)
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
 
    # -------- Top suppliers bar (unchanged)
    with right:
        st.markdown('<div class="block-title">Top Suppliers by Spend</div>', unsafe_allow_html=True)
        top_sup = sup_tot.sort_values("spend_eur", ascending=False).head(20).copy()
        if PLOTLY and not top_sup.empty:
            top_sup["Spend_M"] = top_sup["spend_eur"]/1_000_000.0
            fig2 = px.bar(
                top_sup, x="Spend_M", y="Supplier", orientation="h",
                text=top_sup["Spend_M"].map(lambda v: f"€ {v:,.1f} M"),
                labels={"Spend_M":"Spend (€ M)"},
                color_discrete_sequence=[P_PRIMARY],
            )
            fig2.update_traces(textposition="outside", cliponaxis=False)
            fig2.update_layout(
                height=520,
                margin=dict(l=10, r=140, t=0, b=10),
                yaxis=dict(categoryorder="total ascending", automargin=True, ticksuffix="  "),
                xaxis=dict(title="", showgrid=True, zeroline=False, gridcolor="#e5e7eb"),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No supplier spend to plot yet.")
 
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # SUPPLIER × CATEGORY MIX — FIXED to match Top-20 names AND ORDER, legend below
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    st.markdown('<div class="spacer-24"></div>', unsafe_allow_html=True)
    st.markdown('<div class="block-title">Supplier × Category Mix (Top 20 suppliers)</div>', unsafe_allow_html=True)
 
    # 1) exact same Top-20 suppliers (by name) as above
    top20_suppliers = top_sup["Supplier"].tolist() if 'top_sup' in locals() and not top_sup.empty else []
 
    # 2) same order as bar chart (ascending by spend)
    if top20_suppliers:
        top20_order_for_mix = (
            sup_tot[sup_tot["Supplier"].isin(top20_suppliers)]
            .sort_values("spend_eur", ascending=True)["Supplier"].tolist()
        )
    else:
        top20_order_for_mix = []
 
    # 3) build the mix strictly from those Top-20 suppliers (same names)
    mix = pd.DataFrame()
    if top20_order_for_mix:
        mix = (
            df[df["supplier"].isin(top20_order_for_mix)]
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
        fig3 = px.bar(
            mix, x="share_pct", y="Supplier", color="category_resolved",
            orientation="h", barmode="stack", color_discrete_map=color_map
        )
        fig3.update_layout(
            height=max(560, len(top20_order_for_mix)*26 + 180),
            margin=dict(l=10, r=40, t=10, b=120),
            legend=dict(orientation="h", y=-0.20, x=0, title="Category"),
            xaxis=dict(title="Share (%)", ticksuffix="%", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="Supplier", automargin=True, categoryorder="array", categoryarray=top20_order_for_mix),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Mix chart will appear once the Top-20 suppliers are available.")
 
    # ------------------------ DATA QUALITY (unchanged) ------------------------
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
 
# ============================== DEEP DIVES (unchanged) ========================
else:
    st.subheader("Deep Dives")
 
    def savings_range(cat_val):
        s = "" if pd.isna(cat_val) else str(cat_val).lower()
        if "stamp" in s: return (0.10,0.15)
        if "tube"  in s: return (0.08,0.14)
        if "plast" in s: return (0.08,0.18)
        if "steel" in s: return (0.05,0.12)
        if "log"   in s: return (0.08,0.15)
        return (0.05,0.10)
 
    cat2 = (df.groupby("category_resolved", dropna=False)
           .agg(spend_eur=("_spend_eur","sum"), lines=("category_resolved","count"),
                suppliers=("supplier", pd.Series.nunique))
           .reset_index()
           .rename(columns={"category_resolved":"Category","lines":"# PO Lines"}))
    cat2["Spend (€ k)"] = fmt_k(cat2["spend_eur"])
 
    rngs = [savings_range(c) for c in cat2["Category"]]
    cat2["Savings Range (%)"] = [f"{int(a*100)}–{int(b*100)}" for a,b in rngs]
    cat2["Potential Min (€ k)"] = (cat2["spend_eur"] * [a for a,b in rngs] / 1_000).round(0)
    cat2["Potential Max (€ k)"] = (cat2["spend_eur"] * [b for a,b in rngs] / 1_000).round(0)
 
    st.dataframe(
        cat2[["Category","Spend (€ k)","Savings Range (%)","Potential Min (€ k)","Potential Max (€ k)","# PO Lines"]],
        use_container_width=True,
        column_config={
            "Spend (€ k)": st.column_config.NumberColumn(format="€ %d k"),
            "Potential Min (€ k)": st.column_config.NumberColumn(format="€ %d k"),
            "Potential Max (€ k)": st.column_config.NumberColumn(format="€ %d k"),
        }
    )
 
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
