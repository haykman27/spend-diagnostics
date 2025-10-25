# Procurement Diagnostics — Final (Auto FX → EUR)
# Sustainable spend detection, robust ECB FX (CSV), fallback selectors for Category/Supplier/Currency/Unit Price/Quantity,
# Spend Source selector (Detected vs Unit×Qty×FX vs Auto-validate),
# Category & Supplier analytics, VAVE, Consistency check,
# Outlier & Opportunity Finder (baseline vs price),
# and a Price vs Volume scatter with Category/Supplier filters.

import io, re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from rapidfuzz import process, fuzz

# -------------------------- App setup --------------------------
st.set_page_config(page_title="Procurement Diagnostics — Final", layout="wide")
st.title("Procurement Diagnostics — Final (Auto FX → EUR)")
st.caption(
    "Upload your Excel spend cube. The app auto-detects columns, parses numbers, converts to EUR "
    "(latest ECB FX), and shows diagnostics in € k. If a key column isn’t recognized, you can pick it. "
    "Use the Spend Source toggle to control exactly which column/calculation to use. "
    "Includes Outlier & Opportunity Finder and a Price vs Volume scatter."
)

uploaded = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx", "xls"])
BASE = "EUR"

# -------------------------- Helpers ----------------------------
def normalize_headers(cols):
    return [re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()) for c in cols]

def parse_price_to_float(x):
    """Parse numbers robustly (handles symbols, thousands sep, EU/US decimals)."""
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = re.sub(r"[^\d,\.\-]", "", str(x))
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and s.count(",")==1 and len(s.split(",")[-1]) in (2,3):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

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

# -------------------- ECB FX (CSV, robust) ---------------------
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

# -------------------- Column mapping (auto) --------------------
TARGETS = {
    "category": [
        "category","category name","category desc","category description","commodity","spend category",
        "material group","material group desc","gl category","family","family group","group","group desc",
        "group name","sub category","sub category desc","item family","item family group","item group",
        "procurement group","procurement group name","cluster","class"
    ],
    "supplier": ["supplier","supplier name","vendor","vendor name","seller","payee"],
    "currency": ["currency","ccy","curr","currency code","iso currency","tran curr","transaction currency"],
    "unit_price": [
        "unit price","price","unit cost","net price","price per unit","unit gross price",
        "unit po price","po unit price","unit price (base curr)","unit price (local curr)","unit price (global curr)"
    ],
    "quantity": ["quantity","qty","order qty","qty ordered","units","volume","po qty","po quantity","ordered qty"],
    "amount": ["amount","line amount","total","total price","value","net value","extended price","purchase amount"]
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

# ---------------- Spend detection (sustainable) ----------------
SPEND_NAME_CUES = [
    "purchase amount","po amount","line total","line value","total value","net value","gross amount",
    "extended price","spend","base curr","base currency","global curr","local curr","invoice value","reporting curr"
]
def detect_spend_column(df):
    hits = [c for c in df.columns if any(k in c.lower() for k in SPEND_NAME_CUES)]
    if not hits: return None
    if len(hits) == 1: return hits[0]
    med = {c: pd.to_numeric(df[c].apply(parse_price_to_float), errors="coerce").median(skipna=True) for c in hits}
    return max(med, key=med.get)

# ---------------- Savings & VAVE (NaN-safe) -------------------
SAVINGS = {"stampings":(0.10,0.15),"tubes":(0.08,0.14),"plast":(0.08,0.18),"steel":(0.05,0.12),"logist":(0.08,0.15)}
VAVE = {
    "stampings": ["Standardize steel grades","Improve sheet nesting","Relax tolerances","Bundle volumes"],
    "tubes": ["Standardize diameters","Use HF-welded","Loosen length tolerance"],
    "plast": ["Consolidate resins","Family molds","Standard inserts"],
    "steel": ["Standardize material specs","Use index-linked pricing"],
    "logist": ["Optimize mode mix","Increase load factor"]
}
def savings_for(cat_val):
    c = "" if (cat_val is None or (isinstance(cat_val,float) and np.isnan(cat_val))) else str(cat_val).lower()
    for k,v in SAVINGS.items():
        if k in c: return v
    return (0.05, 0.10)
def vave_for(cat_val):
    c = "" if (cat_val is None or (isinstance(cat_val,float) and np.isnan(cat_val))) else str(cat_val).lower()
    for k,v in VAVE.items():
        if k in c: return v
    return ["Standardize specs","Consolidate variants","Bundle volumes"]

def fmt_k(series: pd.Series) -> pd.Series:
    return (series/1_000.0).round(0)

# ---------- Optional: detect a "part/material/desc" column -----
def detect_item_col(df):
    cues = ["part", "material", "item", "description", "desc", "sku"]
    hits = [c for c in df.columns if any(k in c.lower() for k in cues)]
    if not hits: return None
    return sorted(hits, key=lambda x: len(x), reverse=True)[0]

# ============================== MAIN ==============================
if uploaded:
    raw = pd.read_excel(uploaded)
    raw.columns = [str(c) for c in raw.columns]

    # Auto-map & fallback selectors
    mapping = suggest_columns(raw)
    if "category" not in mapping:
        st.warning("Category column not auto-detected. Please choose it below.")
        mapping["category"] = st.selectbox("Choose the Category/Family/Group column:", options=raw.columns, key="cat_pick")
    if "supplier" not in mapping:
        st.warning("Supplier column not auto-detected. Please choose it below.")
        mapping["supplier"] = st.selectbox("Choose the Supplier/Vendor column:", options=raw.columns, key="sup_pick")
    if "currency" not in mapping:
        st.warning("Currency column not auto-detected. Please choose it below.")
        mapping["currency"] = st.selectbox("Choose the Currency column:", options=raw.columns, key="cur_pick")
    if "unit_price" not in mapping:
        st.warning("Unit Price column not auto-detected. Please choose it below (price per item).")
        mapping["unit_price"] = st.selectbox("Choose the Unit Price column (per item):", options=raw.columns, key="up_pick")
    if "quantity" not in mapping:
        st.warning("Quantity column not auto-detected. Please choose it below.")
        mapping["quantity"] = st.selectbox("Choose the Quantity column:", options=raw.columns, key="qty_pick")

    df = raw.rename(columns={v:k for k,v in mapping.items() if v}).copy()

    spend_col = detect_spend_column(raw)
    if spend_col:
        st.success(f"Detected '{spend_col}' as spend column.")
    else:
        st.warning("No clear spend column found; you can use Unit×Qty×FX instead.")

    # Parse numeric inputs
    if "unit_price" in df.columns: df["unit_price"] = df["unit_price"].apply(parse_price_to_float)
    if "quantity" in df.columns: df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Latest ECB FX
    fx = load_latest_ecb()
    df["currency"] = df.get("currency", "")
    df = apply_fx_latest(df, fx)

    # -------- Spend computation (no double FX) + Selector --------
    def _is_global_header(c): return any(k in c.lower() for k in ["base curr","base currency","global curr","reporting curr"])
    def _find_base_ccy_col(df_raw, spend_col_name):
        cands = [c for c in df_raw.columns if c != spend_col_name and any(k in c.lower() for k in
                 ["base curr code","base currency code","base curr","base currency",
                  "reporting curr","reporting currency","global curr","global currency"])]
        return cands[0] if cands else None

    # Unit price in EUR for diagnostics/outliers
    if "unit_price" in df.columns:
        df["unit_price_eur"] = df["unit_price"] * df["rate_to_eur"]
    else:
        df["unit_price_eur"] = np.nan

    # 1) Spend from detected column (converted properly)
    spend_detected = pd.Series(np.nan, index=df.index)
    if spend_col:
        spend_vals = raw[spend_col].apply(parse_price_to_float)
        if _is_global_header(spend_col):
            base_col = _find_base_ccy_col(raw, spend_col)
            if base_col:
                base_iso = raw[base_col].astype(str).apply(detect_iso_from_text).fillna("EUR")
                fx_map = dict(zip(fx["currency"], fx["rate_to_eur"]))
                base_rate = base_iso.map(lambda c: fx_map.get(c, 1.0))
                spend_detected = pd.to_numeric(spend_vals, errors="coerce") * base_rate
            else:
                spend_detected = pd.to_numeric(spend_vals, errors="coerce")  # assume already EUR
        else:
            spend_detected = pd.to_numeric(spend_vals, errors="coerce") * df["rate_to_eur"]

    # 2) Spend calculated from your Excel: Unit × Qty × FX
    spend_calc = (
        pd.to_numeric(df.get("unit_price"), errors="coerce") *
        pd.to_numeric(df.get("quantity"), errors="coerce") *
        df["rate_to_eur"]
    )

    # Spend source selector
    mode = st.radio(
        "Spend source (from your uploaded Excel):",
        ["Use detected spend column", "Use Unit×Qty×FX", "Auto-validate"],
        index=0,
        help="Choose which spend source to use. Auto-validate compares both and picks the consistent one."
    )

    if mode == "Use detected spend column":
        df["_spend_eur"] = spend_detected.fillna(0.0)
    elif mode == "Use Unit×Qty×FX":
        df["_spend_eur"] = spend_calc.fillna(0.0)
    else:
        use_calc = False
        if spend_detected.notna().sum() >= 50 and spend_calc.notna().sum() >= 50:
            mask = spend_detected.notna() & spend_calc.notna()
            a = spend_detected[mask].values
            b = spend_calc[mask].values
            if len(a) > 10:
                corr = np.corrcoef(a, b)[0,1]
                total_detected = np.nansum(spend_detected)
                total_calc = np.nansum(spend_calc)
                diff_ratio = abs(total_detected - total_calc) / max(1.0, total_calc)
                if (np.isnan(corr) or corr < 0.85) or (diff_ratio > 0.30):
                    use_calc = True
        else:
            if spend_detected.notna().sum() < spend_calc.notna().sum():
                use_calc = True
        if use_calc:
            st.warning("Auto-validate switched to **Unit×Qty×FX** because the detected spend column "
                       "was inconsistent with the calculation.")
            df["_spend_eur"] = spend_calc.fillna(0.0)
        else:
            df["_spend_eur"] = spend_detected.fillna(0.0)

    # Transparency
    st.caption(
        f"Detected spend total: € {np.nansum(spend_detected):,.0f}  •  "
        f"Unit×Qty×FX total: € {np.nansum(spend_calc):,.0f}"
    )

    # --------------------- 1) Category Overview ---------------------
    cat = df.groupby("category", dropna=False).agg(
        spend_eur=("_spend_eur","sum"),
        lines=("category","count"),
        suppliers=("supplier", pd.Series.nunique)
    ).reset_index()

    rngs = [savings_for(c) for c in cat["category"]]
    cat["Savings Range (%)"] = [f"{int(r[0]*100)}–{int(r[1]*100)}" for r in rngs]
    cat["Potential Min (€ k)"] = (cat["spend_eur"] * [r[0] for r in rngs] / 1_000).round(0)
    cat["Potential Max (€ k)"] = (cat["spend_eur"] * [r[1] for r in rngs] / 1_000).round(0)
    cat["Spend (€ k)"] = fmt_k(cat["spend_eur"])
    cat.rename(columns={"category":"Category","lines":"# PO Lines","suppliers":"# Suppliers"}, inplace=True)

    st.subheader("1) Category Overview")
    st.dataframe(
        cat[["Category","Spend (€ k)","Savings Range (%)","Potential Min (€ k)","Potential Max (€ k)","# PO Lines","# Suppliers"]],
        use_container_width=True,
        column_config={
            "Spend (€ k)": st.column_config.NumberColumn(format="€ %d k"),
            "Potential Min (€ k)": st.column_config.NumberColumn(format="€ %d k"),
            "Potential Max (€ k)": st.column_config.NumberColumn(format="€ %d k"),
            "# PO Lines": st.column_config.NumberColumn(format="%d"),
            "# Suppliers": st.column_config.NumberColumn(format="%d"),
        }
    )

    # --------------------- 2) Supplier Drill-Down --------------------
    st.subheader("2) Supplier Drill-Down")
    chosen_cat = st.selectbox("Choose Category:", options=cat["Category"])
    sup = (
        df[df["category"] == chosen_cat]
        .groupby("supplier", dropna=False)
        .agg(spend_eur=("_spend_eur","sum"))
        .reset_index()
        .assign(**{"Spend (€ k)": lambda x: fmt_k(x["spend_eur"])})
        .rename(columns={"supplier":"Supplier"})
        .sort_values("Spend (€ k)", ascending=False)
    )
    st.dataframe(
        sup[["Supplier","Spend (€ k)"]],
        use_container_width=True,
        column_config={"Spend (€ k)": st.column_config.NumberColumn(format="€ %d k")}
    )

    # ------------------------ 3) VAVE Ideas --------------------------
    st.subheader("3) Example VAVE Ideas")
    vrows = [{"Category": (c if pd.notna(c) else ""), "Example VAVE Ideas": " • ".join(vave_for(c))} for c in cat["Category"]]
    st.dataframe(pd.DataFrame(vrows), use_container_width=True)

    # -------------------- 4) Consistency Check ----------------------
    st.subheader("4) Consistency Check (Spend / Unit×Qty)")
    dbg = df[["supplier","category","_spend_eur"]].copy()
    dbg["calc_spend"] = spend_calc
    with np.errstate(divide="ignore", invalid="ignore"):
        dbg["consistency_ratio"] = np.where(dbg["calc_spend"]>0, dbg["_spend_eur"]/dbg["calc_spend"], np.nan)
    dbg["Spend (€ k)"] = fmt_k(dbg["_spend_eur"])
    dbg["Calc (€ k)"] = fmt_k(dbg["calc_spend"])
    st.dataframe(
        dbg[["supplier","category","Spend (€ k)","Calc (€ k)","consistency_ratio"]].head(300),
        use_container_width=True,
        column_config={
            "Spend (€ k)": st.column_config.NumberColumn(format="€ %d k"),
            "Calc (€ k)": st.column_config.NumberColumn(format="€ %d k"),
        }
    )

    # -------------------- 5) Outlier & Opportunities -----------------
    st.subheader("5) Outlier & Opportunity Finder (N4)")

    # Controls
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        premium_threshold = st.slider("Min premium over baseline (%)", 5, 60, 20, step=1,
                                      help="Flag lines where Unit Price (EUR) exceeds category baseline by at least this %.")
    with col_b:
        min_line_opportunity = st.slider("Min line opportunity (€)", 0, 20000, 1000, step=500,
                                         help="Ignore tiny lines; show lines with potential saving above this amount.")
    with col_c:
        baseline_method = st.selectbox("Baseline method", ["Median (P50)", "Trimmed mean (10%–90%)"],
                                       help="How the baseline unit price per category is computed.")

    # Compute baseline per category
    df_bp = df.copy()
    if "unit_price_eur" not in df_bp.columns:
        df_bp["unit_price_eur"] = pd.to_numeric(df_bp.get("unit_price"), errors="coerce") * df_bp["rate_to_eur"]

    def _baseline(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty: return np.nan
        if baseline_method.startswith("Median"):
            return float(np.median(s))
        lo, hi = np.percentile(s, [10,90])
        s = s[(s>=lo) & (s<=hi)]
        return float(s.mean()) if len(s) else np.nan

    baselines = (df_bp.groupby("category")["unit_price_eur"]
                 .apply(_baseline)
                 .rename("baseline_price_eur")
                 .reset_index())

    df_bp = df_bp.merge(baselines, on="category", how="left")
    df_bp["premium_pct"] = (df_bp["unit_price_eur"] - df_bp["baseline_price_eur"]) / df_bp["baseline_price_eur"]
    df_bp["line_opportunity_eur"] = (
        (df_bp["unit_price_eur"] - df_bp["baseline_price_eur"]).clip(lower=0) *
        pd.to_numeric(df_bp.get("quantity"), errors="coerce")
    )
    df_bp["line_opportunity_eur"] = df_bp["line_opportunity_eur"].fillna(0.0)

    # Outlier lines filter
    outliers = df_bp[
        (df_bp["premium_pct"] >= premium_threshold/100.0) &
        (df_bp["line_opportunity_eur"] >= min_line_opportunity)
    ].copy()

    # Try to show a part/material/desc column if present
    item_col = detect_item_col(raw)
    if item_col and item_col in outliers.columns:
        display_cols = [item_col]
    else:
        display_cols = []

    # Outlier Lines table
    outliers_display = outliers.assign(
        **{
            "Premium (%)": (outliers["premium_pct"]*100).round(1),
            "Unit Price (EUR)": outliers["unit_price_eur"].round(4),
            "Baseline (EUR)": outliers["baseline_price_eur"].round(4),
            "Opportunity (€ k)": (outliers["line_opportunity_eur"]/1000).round(0),
        }
    )
    show_cols = (["category","supplier"] + display_cols +
                 ["Unit Price (EUR)","Baseline (EUR)","Premium (%)","quantity","Opportunity (€ k)"])
    st.markdown("**Outlier Lines** (price above baseline)")
    st.dataframe(outliers_display[show_cols].sort_values("Opportunity (€ k)", ascending=False),
                 use_container_width=True)

    # Top opportunity by Supplier
    st.markdown("**Top Opportunity by Supplier**")
    opp_sup = (outliers.groupby("supplier", dropna=False)["line_opportunity_eur"]
               .sum().reset_index()
               .assign(**{"Opportunity (€ k)": lambda x: (x["line_opportunity_eur"]/1000).round(0)})
               .rename(columns={"supplier":"Supplier"})
               .sort_values("Opportunity (€ k)", ascending=False))
    st.dataframe(opp_sup[["Supplier","Opportunity (€ k)"]], use_container_width=True,
                 column_config={"Opportunity (€ k)": st.column_config.NumberColumn(format="€ %d k")})

    # Top opportunity by Category
    st.markdown("**Top Opportunity by Category**")
    opp_cat = (outliers.groupby("category", dropna=False)["line_opportunity_eur"]
               .sum().reset_index()
               .assign(**{"Opportunity (€ k)": lambda x: (x["line_opportunity_eur"]/1000).round(0)})
               .rename(columns={"category":"Category"})
               .sort_values("Opportunity (€ k)", ascending=False))
    st.dataframe(opp_cat[["Category","Opportunity (€ k)"]], use_container_width=True,
                 column_config={"Opportunity (€ k)": st.column_config.NumberColumn(format="€ %d k")})

    # ===== Scatter: Unit Price (EUR) vs Quantity (Volume) =====
    st.markdown("**Price vs Volume Scatter**")

    scatter_df = df_bp[[
        "category","supplier","unit_price_eur","quantity","premium_pct","line_opportunity_eur"
    ]].copy()
    scatter_df = scatter_df[
        scatter_df["unit_price_eur"].notna() &
        scatter_df["quantity"].notna() &
        (scatter_df["quantity"] > 0)
    ]

    cat_options = ["All"] + sorted([c for c in scatter_df["category"].dropna().unique()])
    cat_choice = st.selectbox("Category (for scatter):", cat_options, index=0)

    if cat_choice != "All":
        sup_options = ["All"] + sorted([s for s in scatter_df.loc[scatter_df["category"]==cat_choice, "supplier"].dropna().unique()])
    else:
        sup_options = ["All"] + sorted([s for s in scatter_df["supplier"].dropna().unique()])

    sup_choice = st.selectbox("Supplier (for scatter):", sup_options, index=0)

    sc = scatter_df.copy()
    if cat_choice != "All":
        sc = sc[sc["category"] == cat_choice]
    if sup_choice != "All":
        sc = sc[sc["supplier"] == sup_choice]

    if sc.empty:
        st.info("No rows match the current filters.")
    else:
        # Clamp axes to 1st–99th percentiles for readability
        q_low, q_high = np.nanpercentile(sc["quantity"], [0, 99])
        p_low, p_high = np.nanpercentile(sc["unit_price_eur"], [0, 99])
        if q_low == q_high: q_high = q_low + 1
        if p_low == p_high: p_high = p_low + 1

        chart = (
            alt.Chart(sc)
            .mark_circle(opacity=0.7)
            .encode(
                x=alt.X("quantity:Q", title="Quantity (Volume)", scale=alt.Scale(domain=[q_low, q_high])),
                y=alt.Y("unit_price_eur:Q", title="Actual unit price (EUR)", scale=alt.Scale(domain=[p_low, p_high])),
                size=alt.Size("line_opportunity_eur:Q", title="Opportunity (EUR)", scale=alt.Scale(range=[20, 800])),
                color=alt.Color("premium_pct:Q",
                                title="Premium %",
                                scale=alt.Scale(scheme="redyellowgreen", domain=[-0.1, 0.0, 0.5], reverse=True),
                                legend=alt.Legend(format=".0%")),
                tooltip=[
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("supplier:N", title="Supplier"),
                    alt.Tooltip("quantity:Q", title="Quantity", format=",.0f"),
                    alt.Tooltip("unit_price_eur:Q", title="Unit price (EUR)", format=",.4f"),
                    alt.Tooltip("premium_pct:Q", title="Premium %", format=".1%"),
                    alt.Tooltip("line_opportunity_eur:Q", title="Opportunity (€)", format=",.0f"),
                ],
            )
            .properties(height=420)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    # ------------------------- 6) Download pack --------------------
    st.markdown("#### Download full results (XLSX)")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Lines")
        cat.to_excel(w, index=False, sheet_name="Categories")
        sup.to_excel(w, index=False, sheet_name="Suppliers_Selected")
        pd.DataFrame(vrows).to_excel(w, index=False, sheet_name="VAVE_Ideas")
        outliers_display.to_excel(w, index=False, sheet_name="Outlier_Lines")
        opp_sup.to_excel(w, index=False, sheet_name="Opp_by_Supplier")
        opp_cat.to_excel(w, index=False, sheet_name="Opp_by_Category")
    st.download_button(
        "Download results.xlsx",
        buf.getvalue(),
        "procurement_diagnostics_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info(
        "Upload an Excel file to begin. If Category/Supplier/Currency/Unit Price/Quantity aren’t detected automatically, "
        "you’ll be prompted to select the correct columns. All spends are shown in EUR (latest FX) as € k. "
        "Use the Outlier & Opportunity Finder to spot high-price lines and biggest saving levers, and the Price vs Volume scatter to see economies of scale."
    )
