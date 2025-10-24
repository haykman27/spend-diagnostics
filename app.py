# Procurement Diagnostics — Final (Auto FX → EUR, sustainable spend detection)
# - Keeps all features we built: auto-mapping, sustainable spend-detection, latest ECB FX, category savings,
#   supplier drill-down, VAVE ideas, sanity checks, debug tray, download pack.
# - NEW: Robust ECB loader using CSV (no JSONDecodeError). Uses latest available rates.
# - Displays spend as "€ 1,234 k" with interactive tables.

import io, re
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# ------------------- Streamlit setup -------------------
st.set_page_config(page_title="Procurement Diagnostics — Final", layout="wide")
st.title("Procurement Diagnostics — Final (Auto FX → EUR)")
st.caption(
    "Upload your Excel. The app auto-detects key columns, fixes decimals, detects the spend column, "
    "converts to EUR with latest ECB rates, and shows interactive diagnostics in € k."
)
uploaded = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])
BASE = "EUR"

# ------------------- Helpers: header & parsing -------------------
def normalize_headers(cols):
    return [re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()) for c in cols]

def parse_price_to_float(x):
    """Parse numbers that may contain currency symbols, thousands, or EU/US decimals."""
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x)
    # Keep only digits, dot, comma, minus
    s = re.sub(r"[^\d,\.\-]", "", s)
    # If both separators exist, assume comma is thousands sep: drop commas
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        # If there's a single comma and it looks like decimal comma, convert to dot
        if "," in s and s.count(",")==1 and len(s.split(",")[-1]) in (2,3):
            s = s.replace(",", ".")
        else:
            # Otherwise treat commas as thousands separators
            s = s.replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

# ------------------- Currency helpers -------------------
CURRENCY_SYMBOL_MAP = {
    "$":"USD","US$":"USD","A$":"AUD","AU$":"AUD","C$":"CAD","CA$":"CAD",
    "€":"EUR","£":"GBP","¥":"JPY","₩":"KRW","₹":"INR","₺":"TRY",
    "R$":"BRL","S$":"SGD","HK$":"HKD","₪":"ILS","₱":"PHP","₦":"NGN","₫":"VND"
}
ISO_3 = {
    "EUR","USD","GBP","JPY","CNY","CHF","SEK","NOK","DKK","PLN","HUF","CZK","RON",
    "AUD","NZD","CAD","MXN","BRL","ZAR","AED","SAR","HKD","SGD","INR","TRY","KRW",
    "TWD","THB","PHP","ILS","NGN","VND","RUB"
}
def detect_iso_from_text(text: str):
    if text is None or (isinstance(text, float) and np.isnan(text)): return None
    s = str(text).strip().upper()
    if not s: return None
    alias = {"RMB":"CNY","YUAN":"CNY","CN¥":"CNY","元":"CNY","ZŁ":"PLN","ZL":"PLN","KČ":"CZK",
             "LEI":"RON","RUR":"RUB","РУБ":"RUB"}
    # explicit 3-letter code
    m = re.search(r"\b([A-Z]{3})\b", s)
    if m:
        c = m.group(1)
        if c in ISO_3: return c
        if c in alias: return alias[c]
    # aliases embedded
    for k,v in alias.items():
        if k in s: return v
    # symbols
    for sym in sorted(CURRENCY_SYMBOL_MAP.keys(), key=len, reverse=True):
        if sym in s: return CURRENCY_SYMBOL_MAP[sym]
    return None

# ------------------- ECB FX (LATEST from CSV; robust) -------------------
@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_latest_ecb():
    """
    Return DataFrame with ['currency','rate_to_eur'] using the latest available
    row in eurofxref-hist.csv. Gracefully drops NaNs and returns EUR=1 always.
    """
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"
    fx_wide = pd.read_csv(url)
    fx_wide.rename(columns={"Date": "date"}, inplace=True)
    fx_wide["date"] = pd.to_datetime(fx_wide["date"], errors="coerce")
    latest_date = fx_wide["date"].max()
    latest_row = fx_wide.loc[fx_wide["date"] == latest_date].copy()
    fx_long = latest_row.melt(id_vars=["date"], var_name="currency", value_name="eur_to_cur")
    fx_long["currency"] = fx_long["currency"].str.upper().str.strip()
    fx_long["eur_to_cur"] = pd.to_numeric(fx_long["eur_to_cur"], errors="coerce")
    fx_long = fx_long.dropna(subset=["eur_to_cur"])
    fx_long["rate_to_eur"] = 1.0 / fx_long["eur_to_cur"]
    eur_row = pd.DataFrame([{"currency":"EUR","rate_to_eur":1.0}])
    fx = pd.concat([fx_long[["currency","rate_to_eur"]], eur_row], ignore_index=True)
    fx = fx.drop_duplicates(subset=["currency"]).reset_index(drop=True)
    return fx[["currency","rate_to_eur"]]

def apply_fx_latest(df, fx):
    df["currency_iso"] = df["currency"].astype(str).apply(lambda x: detect_iso_from_text(x) or BASE)
    df = df.merge(fx, left_on="currency_iso", right_on="currency", how="left")
    df["rate_to_eur"] = df["rate_to_eur"].fillna(1.0)
    return df

# ------------------- Auto-mapping -------------------
TARGETS = {
    "po_id":       ["po","po id","po number","purchase order","order id","document","line id","invoice line id"],
    "date":        ["date","posting date","order date","document date","invoice date"],
    "category":    ["category","commodity","spend category","material group","gl category","family","family group","group","item family","item family group"],
    "description": ["description","item","item description","material description","service description","short text","long text"],
    "supplier":    ["supplier","supplier name","vendor","vendor name","seller","payee"],
    "sku":         ["sku","material","material no","material number","item code","product code","part number","pn"],
    "unit_price":  ["unit price","price","unit cost","net price","price per unit","unit gross price"],
    "amount":      ["amount","line amount","total","total price","value","net value","extended price"],
    "quantity":    ["quantity","qty","order qty","qty ordered","units","volume"],
    "currency":    ["currency","ccy","curr","currency code","iso currency"]
}
REQUIRED_MIN = ["supplier","category","currency"]  # we’ll compute others if missing

def suggest_columns(df):
    cols = df.columns.tolist()
    norm = normalize_headers(cols)
    back = {n: orig for n, orig in zip(norm, cols)}
    suggestions = {}
    for field, synonyms in TARGETS.items():
        best = None; best_score = -1
        for syn in synonyms:
            match = process.extractOne(syn, norm, scorer=fuzz.token_sort_ratio)
            if match and match[1] > best_score:
                best, best_score = match[0], match[1]
        if best is not None:
            suggestions[field] = back[best]
    # prefer item family/group as category if present
    for pref in ["item family group","item family"]:
        for n, c in zip(norm, cols):
            if n == pref:
                suggestions["category"] = c
    # ensure minimal required presence
    for r in REQUIRED_MIN:
        if r not in suggestions:
            st.error(f"Missing a required column for '{r}'. Please include it in your file.")
            st.stop()
    return suggestions

# ------------------- Sustainable spend-column detection -------------------
SPEND_NAME_CUES = [
    "purchase amount","po amount","line total","line value","total value",
    "net value","gross amount","extended price","spend","base curr","global curr"
]
def detect_spend_column(df):
    hits = [c for c in df.columns if any(k in c.lower() for k in SPEND_NAME_CUES)]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    # choose the one with largest median numeric value (line totals >> unit price)
    medians = {}
    for c in hits:
        medians[c] = pd.to_numeric(df[c], errors="coerce").median(skipna=True)
    return max(medians, key=medians.get)

# ------------------- Savings ranges & VAVE ideas -------------------
SAVINGS_LIBRARY = {
    "stampings": (0.10, 0.15), "tubes": (0.08, 0.14), "machin": (0.07, 0.14),
    "cast": (0.06, 0.12), "forg": (0.05, 0.10), "plast": (0.08, 0.18),
    "fasten": (0.05, 0.10), "steel": (0.05, 0.12), "logist": (0.08, 0.15),
    "electronics": (0.06, 0.12), "packag": (0.05, 0.15), "mro": (0.05, 0.12)
}
VAVE_IDEAS = {
    "stampings": ["Standardize steel grades", "Improve sheet nesting", "Relax tolerances", "Bundle volumes"],
    "tubes": ["Standardize diameters", "Use HF-welded", "Loosen length tolerance"],
    "machin": ["Shift to near-net", "Reduce setups", "Use commercial finishes"],
    "plast": ["Consolidate resins", "Family molds", "Standard inserts"],
    "steel": ["Standardize material specs", "Use index-linked pricing"],
    "logist": ["Optimize mode mix", "Increase load factor"]
}
def savings_for(cat: str):
    c = (cat or "").lower()
    for k, rng in SAVINGS_LIBRARY.items():
        if k in c: return rng
    if any(w in c for w in ["pcb","electro","electronics"]): return (0.06, 0.12)
    return (0.05, 0.10)
def vave_for(cat: str):
    c = (cat or "").lower()
    for k, ideas in VAVE_IDEAS.items():
        if k in c: return ideas
    return ["Standardize specs", "Consolidate variants", "Bundle volumes"]

# ------------------- Outlier baseline flags -------------------
def add_baseline_and_flags(df, iqr_multiplier=1.5):
    if "unit_price_eur" not in df.columns:
        return df.assign(price_outlier=False, baseline_price=np.nan, potential_eur=np.nan)
    key = ["category"]
    grp = df.groupby(key, dropna=False)["unit_price_eur"]
    base = grp.median().rename("baseline_price").reset_index()
    df = df.merge(base, on=key, how="left")
    if "quantity" in df.columns:
        df["potential_eur"] = ((df["unit_price_eur"] - df["baseline_price"]).clip(lower=0)) * df["quantity"]
    else:
        df["potential_eur"] = np.nan
    def iqr_flag(g):
        x = g["unit_price_eur"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 4:
            upper = np.nanpercentile(x, 75) * 1.25 if len(x) else np.nan
        else:
            q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
            upper = q3 + iqr_multiplier * iqr if iqr > 0 else q3 * 1.25
        return g.assign(price_outlier=g["unit_price_eur"] > upper if pd.notna(upper) else False)
    return df.groupby(key, group_keys=False).apply(iqr_flag)

# ------------------- Number formatting helpers -------------------
def fmt_k_eur(series: pd.Series) -> pd.Series:
    """Return numeric thousands (k€); use column_config for display as '€ 1,234 k'."""
    return (series / 1_000.0).round(0)

# ------------------- MAIN -------------------
if uploaded:
    # 1) Read Excel (keep raw text; we’ll parse only needed numeric cols)
    raw = pd.read_excel(uploaded)
    raw.columns = [str(c) for c in raw.columns]

    # 2) Auto-map core columns
    suggestions = suggest_columns(raw)
    df = raw.rename(columns={v: k for k, v in suggestions.items() if v}).copy()

    # 3) Detect spend column sustainably (header cues + median check)
    spend_col_raw = detect_spend_column(raw)
    if spend_col_raw:
        st.success(f"Detected '{spend_col_raw}' as spend column.")
    else:
        st.warning("No clear spend column found; will use Unit Price × Quantity.")

    # 4) Parse numeric columns properly (no global string mangling)
    if "unit_price" in df.columns:
        df["unit_price"] = df["unit_price"].apply(parse_price_to_float)
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # 5) Latest ECB FX → EUR (ignore date for simplicity, per your request)
    fx = load_latest_ecb()
    df["currency"] = df.get("currency", "")
    df = apply_fx_latest(df, fx)

    # 6) Compute unit_price_eur for flags/outliers and a robust spend value in EUR
    if "unit_price" in df.columns:
        df["unit_price_eur"] = df["unit_price"] * df["rate_to_eur"]
    else:
        df["unit_price_eur"] = np.nan

    if spend_col_raw:
        # use provided line-total column, then convert via latest FX
        spend_numeric = pd.to_numeric(raw[spend_col_raw].apply(parse_price_to_float), errors="coerce")
        df["_spend_eur"] = (spend_numeric * df["rate_to_eur"]).fillna(0.0)
    else:
        # fall back to unit × qty × fx
        df["_spend_eur"] = (
            pd.to_numeric(df.get("unit_price"), errors="coerce") *
            pd.to_numeric(df.get("quantity"), errors="coerce") *
            df["rate_to_eur"]
        ).fillna(0.0)

    # 7) Consistency ratio (only when all inputs present)
    if {"unit_price","quantity"}.issubset(df.columns):
        df["calc_spend"] = (
            pd.to_numeric(df["unit_price"], errors="coerce") *
            pd.to_numeric(df["quantity"], errors="coerce") *
            df["rate_to_eur"]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            df["consistency_ratio"] = np.where(df["calc_spend"]>0, df["_spend_eur"]/df["calc_spend"], np.nan)
    else:
        df["calc_spend"] = np.nan
        df["consistency_ratio"] = np.nan

    # 8) Sanity checks
    too_small_prices = (df["unit_price_eur"].dropna() < 0.05).mean() > 0.4
    msgs = []
    if too_small_prices:
        msgs.append("Many unit prices are under €0.05 — check quantity or decimal conventions.")
    if msgs:
        st.warning("Sanity checks:\n- " + "\n- ".join(msgs))

    # 9) Baseline & outlier flags
    if "category" not in df.columns:
        st.error("No category detected. Please include a 'Category' / 'Item family' column.")
        st.stop()
    df = add_baseline_and_flags(df)

    # ------------------- VIEW 1: Category Overview -------------------
    cat = df.groupby("category", dropna=False).agg(
        spend_eur=("_spend_eur", "sum"),
        lines=("po_id", "count") if "po_id" in df.columns else ("_spend_eur","count"),
        suppliers=("supplier", pd.Series.nunique) if "supplier" in df.columns else ("_spend_eur","count")
    ).reset_index()

    # Savings ranges and potentials
    rngs = [savings_for(c) for c in cat["category"]]
    cat["Savings Range (%)"] = [f"{int(r[0]*100)}–{int(r[1]*100)}" for r in rngs]
    cat["Potential Min (€ k)"] = (cat["spend_eur"] * [r[0] for r in rngs] / 1_000).round(0)
    cat["Potential Max (€ k)"] = (cat["spend_eur"] * [r[1] for r in rngs] / 1_000).round(0)

    # Spend display as € k (numeric; use column_config to show "€ … k")
    cat["Spend (€ k)"] = fmt_k_eur(cat["spend_eur"])

    # Friendly headers
    cat.rename(columns={
        "category": "Category",
        "lines": "# PO Lines",
        "suppliers": "# Suppliers",
    }, inplace=True)

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

    # ------------------- VIEW 2: Supplier Drill-Down -------------------
    st.subheader("2) Supplier Drill-Down")
    chosen_cat = st.selectbox("Choose Category:", options=cat["Category"])
    sup = (
        df[df["category"] == chosen_cat]
        .groupby("supplier", dropna=False)
        .agg(spend_eur=("_spend_eur","sum"))
        .reset_index()
        .assign(**{"Spend (€ k)": lambda x: fmt_k_eur(x["spend_eur"])})
        .rename(columns={"supplier": "Supplier"})
        .sort_values("Spend (€ k)", ascending=False)
    )
    st.dataframe(
        sup[["Supplier","Spend (€ k)"]],
        use_container_width=True,
        column_config={"Spend (€ k)": st.column_config.NumberColumn(format="€ %d k")}
    )

    # ------------------- VIEW 3: VAVE Ideas -------------------
    st.subheader("3) Example VAVE Ideas")
    vave_rows = [{"Category": c, "Example VAVE Ideas": " • ".join(vave_for(c))} for c in cat["Category"]]
    st.dataframe(pd.DataFrame(vave_rows), use_container_width=True)

    # ------------------- VIEW 4: Consistency Check -------------------
    st.subheader("4) Consistency Check (Spend / Unit×Qty)")
    dbg_cols = []
    for c in ["supplier","category","_spend_eur","calc_spend","consistency_ratio","unit_price","quantity","currency","rate_to_eur"]:
        if c in df.columns: dbg_cols.append(c)
    dbg = df[dbg_cols].copy()
    dbg["Spend (€ k)"] = fmt_k_eur(dbg["_spend_eur"]) if "_spend_eur" in dbg else np.nan
    dbg["Calc (€ k)"] = fmt_k_eur(dbg["calc_spend"]) if "calc_spend" in dbg else np.nan
    view_cols = [c for c in ["supplier","category","Spend (€ k)","Calc (€ k)","consistency_ratio"] if c in dbg.columns]
    st.dataframe(
        dbg[view_cols].head(300),
        use_container_width=True,
        column_config={
            "Spend (€ k)": st.column_config.NumberColumn(format="€ %d k"),
            "Calc (€ k)": st.column_config.NumberColumn(format="€ %d k"),
        }
    )

    # ------------------- Download Pack -------------------
    st.markdown("#### Download pack (XLSX)")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lines")
        cat.to_excel(writer, index=False, sheet_name="Categories")
        sup.to_excel(writer, index=False, sheet_name="Suppliers_Selected")
        pd.DataFrame(vave_rows).to_excel(writer, index=False, sheet_name="VAVE_Ideas")
    st.download_button(
        "Download results.xlsx",
        data=buffer.getvalue(),
        file_name="procurement_diagnostics_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Upload an Excel file to begin. The app will auto-detect the spend column, apply the latest ECB FX to EUR, "
            "and display interactive category/supplier/VAVE insights with spends formatted as € k.")
