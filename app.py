# === PART 1/3: imports, setup, and core helpers ===
import io, re
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
import requests

st.set_page_config(page_title="Procurement Diagnostics — Final", layout="wide")
st.title("Procurement Diagnostics — Final (Auto FX → EUR)")
st.caption("Upload your Excel. The app auto-detects key columns, converts to EUR using latest ECB rates, "
           "detects the spend column intelligently, fixes decimals, and shows interactive tables in € k.")

uploaded = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])
BASE = "EUR"

# ---------- Global display prefs ----------
pd.options.display.float_format = "{:,.2f}".format

# ---------- Currency helpers ----------
CURRENCY_SYMBOL_MAP = {
    "$":"USD","US$":"USD","A$":"AUD","AU$":"AUD","C$":"CAD","CA$":"CAD","€":"EUR","£":"GBP",
    "¥":"JPY","₩":"KRW","₹":"INR","₺":"TRY","R$":"BRL","S$":"SGD","HK$":"HKD","₪":"ILS",
    "₱":"PHP","₦":"NGN","₫":"VND"
}
ISO_3 = {"EUR","USD","GBP","JPY","CNY","CHF","SEK","NOK","DKK","PLN","HUF","CZK","RON",
         "AUD","NZD","CAD","MXN","BRL","ZAR","AED","SAR","HKD","SGD","INR","TRY","KRW",
         "TWD","THB","PHP","ILS","NGN","VND","RUB"}

def detect_iso_from_text(text):
    if text is None or (isinstance(text,float) and np.isnan(text)): return None
    s = str(text).strip().upper()
    alias = {"RMB":"CNY","YUAN":"CNY","CN¥":"CNY","元":"CNY","ZŁ":"PLN","ZL":"PLN","KČ":"CZK",
             "LEI":"RON","RUR":"RUB","РУБ":"RUB"}
    m = re.search(r"\b([A-Z]{3})\b", s)
    if m:
        c = m.group(1)
        if c in ISO_3: return c
        if c in alias: return alias[c]
    for k,v in alias.items():
        if k in s: return v
    for sym in sorted(CURRENCY_SYMBOL_MAP.keys(), key=len, reverse=True):
        if sym in s: return CURRENCY_SYMBOL_MAP[sym]
    return None

def parse_price_to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float)): return float(x)
    s = str(x)
    s = re.sub(r"[^\d,\.\-]", "", s)
    if "," in s and "." in s: s = s.replace(",", "")
    else:
        if "," in s and s.count(",")==1 and len(s.split(",")[-1]) in (2,3): s = s.replace(",", ".")
        else: s = s.replace(",", "")
    try: return float(s)
    except: return np.nan

# ---------- Normalization helpers ----------
def normalize_headers(cols):
    return [re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()) for c in cols]

# ---------- ECB FX ----------
@st.cache_data(ttl=6*60*60)
def load_latest_ecb():
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref.json"
    data = requests.get(url, timeout=10).json()
    rates = data["rates"]
    latest_date = list(rates.keys())[0]
    rate_dict = rates[latest_date]
    df = pd.DataFrame(rate_dict.items(), columns=["currency","eur_to_cur"])
    df["rate_to_eur"] = 1/pd.to_numeric(df["eur_to_cur"], errors="coerce")
    df.loc[len(df)] = ["EUR",1.0,1.0]
    return df

# ---------- Sustainable spend column detector ----------
def detect_spend_column(df):
    header_hits = [c for c in df.columns if any(k in c.lower() for k in
                    ["purchase amount","po amount","line total","total value",
                     "line value","net value","gross amount","spend","extended price","amount (global)"])]
    if not header_hits:
        return None
    if len(header_hits)==1:
        return header_hits[0]
    # Heuristic: pick the one with largest median numeric value
    candidates = {}
    for c in header_hits:
        s = pd.to_numeric(df[c], errors="coerce")
        candidates[c] = np.nanmedian(s)
    return max(candidates, key=candidates.get)
# === END PART 1/3 ===
# === PART 2/3: FX merge, spend calc, savings, VAVE, analytics helpers ===

# ---------- FX merge ----------
def apply_fx(df, fx):
    df["currency_iso"] = df["currency"].astype(str).apply(lambda x: detect_iso_from_text(x) or BASE)
    df = df.merge(fx[["currency","rate_to_eur"]], left_on="currency_iso", right_on="currency", how="left")
    df["rate_to_eur"] = df["rate_to_eur"].fillna(1.0)
    return df

# ---------- Savings & VAVE ----------
SAVINGS_LIBRARY = {
    "stampings":(0.10,0.15),"tubes":(0.08,0.14),"machin":(0.07,0.14),
    "cast":(0.06,0.12),"forg":(0.05,0.10),"plast":(0.08,0.18),
    "fasten":(0.05,0.10),"steel":(0.05,0.12),"logist":(0.08,0.15)
}
VAVE_IDEAS = {
    "stampings":["Standardize steel grades","Improve sheet nesting","Relax tolerances","Bundle volumes"],
    "tubes":["Standardize diameters","Use HF-welded","Loosen length tolerance"],
    "machin":["Shift to near-net","Reduce setups","Use commercial finishes"],
    "plast":["Consolidate resins","Family molds","Standard inserts"],
    "steel":["Standardize material specs","Use index-linked pricing"],
    "logist":["Optimize mode mix","Increase load factor"]
}
def savings_for(cat):
    c = (cat or "").lower()
    for k,v in SAVINGS_LIBRARY.items():
        if k in c: return v
    return (0.05,0.10)
def vave_for(cat):
    c = (cat or "").lower()
    for k,v in VAVE_IDEAS.items():
        if k in c: return v
    return ["Standardize specs","Consolidate variants","Bundle volumes"]

# ---------- Analytics helpers ----------
def add_baseline_and_flags(df, iqr_multiplier=1.5):
    key = ["category"]
    grp = df.groupby(key, dropna=False)["unit_price_eur"]
    p50 = grp.median(); p25 = grp.quantile(0.25)
    base = p50.rename("baseline_price").reset_index()
    df = df.merge(base, on=key, how="left")
    df["potential_eur"] = ((df["unit_price_eur"] - df["baseline_price"]).clip(lower=0)) * df["quantity"]
    def iqr_flag(g):
        x = g["unit_price_eur"].astype(float).replace([np.inf,-np.inf],np.nan).dropna()
        if len(x)<4: upper=np.nanpercentile(x,75)*1.25 if len(x) else np.nan
        else:
            q1,q3=np.percentile(x,[25,75]); iqr=q3-q1
            upper=q3+iqr_multiplier*iqr if iqr>0 else q3*1.25
        return g.assign(price_outlier=g["unit_price_eur"]>upper if pd.notna(upper) else False)
    return df.groupby(key, group_keys=False).apply(iqr_flag)
# === END PART 2/3 ===
# === PART 3/3: main app flow, display, and download ===
if uploaded:
    # ----- READ & CLEAN -----
    raw = pd.read_excel(uploaded, dtype=str)
    for col in raw.columns:
        raw[col] = raw[col].str.replace(r"[^\d,.\-]", "", regex=True)
        raw[col] = raw[col].str.replace(",", ".", regex=False)
    raw = raw.apply(pd.to_numeric, errors="ignore")

    # detect spend col
    spend_col = detect_spend_column(raw)
    if spend_col:
        st.success(f"Detected '{spend_col}' as spend column.")
    else:
        st.warning("No clear spend column found; will use unit price × quantity.")

    # basic column mapping
    norm = normalize_headers(raw.columns)
    back = {n:o for n,o in zip(norm, raw.columns)}
    mapping = {
        "supplier": next((back[c] for c in norm if "supplier" in c or "vendor" in c), None),
        "category": next((back[c] for c in norm if "item family" in c or "category" in c), None),
        "unit_price": next((back[c] for c in norm if "unit price" in c), None),
        "quantity": next((back[c] for c in norm if "qty" in c or "quantity" in c), None),
        "currency": next((back[c] for c in norm if "currency" in c), None)
    }
    df = raw.rename(columns={v:k for k,v in mapping.items() if v})

    # FX
    fx = load_latest_ecb()
    df = apply_fx(df, fx)

    # compute spend
    if spend_col:
        df["_spend_eur"] = pd.to_numeric(raw[spend_col], errors="coerce") * df["rate_to_eur"]
    else:
        df["unit_price"] = df["unit_price"].apply(parse_price_to_float)
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df["_spend_eur"] = (df["unit_price"] * df["quantity"] * df["rate_to_eur"]).fillna(0.0)

    # consistency ratio
    if {"unit_price","quantity","_spend_eur"}.issubset(df.columns):
        df["calc_spend"] = df["unit_price"]*df["quantity"]*df["rate_to_eur"]
        df["consistency_ratio"] = np.where(df["calc_spend"]>0, df["_spend_eur"]/df["calc_spend"], np.nan)

    # add baseline & flags
    df["unit_price_eur"] = df["unit_price"]*df["rate_to_eur"]
    df = add_baseline_and_flags(df)

    # CATEGORY view
    cat = df.groupby("category", dropna=False).agg(
        spend_eur=("_spend_eur","sum"),
        lines=("quantity","count"),
        suppliers=("supplier","nunique")
    ).reset_index()
    cat["spend_k"] = cat["spend_eur"]/1_000
    cat["Savings Range (%)"] = cat["category"].apply(lambda x: f"{int(savings_for(x)[0]*100)}–{int(savings_for(x)[1]*100)}")
    cat["Potential Min (€ k)"] = (cat["spend_eur"]*cat["category"].apply(lambda x:savings_for(x)[0])/1_000).round(0)
    cat["Potential Max (€ k)"] = (cat["spend_eur"]*cat["category"].apply(lambda x:savings_for(x)[1])/1_000).round(0)
    cat.rename(columns={"category":"Category","spend_k":"Spend (€ k)","lines":"# PO Lines","suppliers":"# Suppliers"}, inplace=True)

    st.subheader("1) Category Overview")
    st.dataframe(cat, use_container_width=True)

    # SUPPLIER drill-down
    st.subheader("2) Supplier Drill-Down")
    chosen = st.selectbox("Choose Category:", cat["Category"])
    sup = (df[df["category"]==chosen].groupby("supplier",dropna=False)
           .agg(spend_eur=("_spend_eur","sum"))
           .reset_index()
           .assign(**{"Spend (€ k)":lambda x:(x["spend_eur"]/1_000).round(0)}))
    sup.rename(columns={"supplier":"Supplier"}, inplace=True)
    st.dataframe(sup[["Supplier","Spend (€ k)"]].sort_values("Spend (€ k)",ascending=False), use_container_width=True)

    # VAVE
    st.subheader("3) Example VAVE Ideas")
    vave_rows=[{"Category":c,"Example VAVE Ideas":" • ".join(vave_for(c))} for c in cat["Category"]]
    st.dataframe(pd.DataFrame(vave_rows), use_container_width=True)

    # Consistency check
    st.subheader("4) Consistency Check (Spend / Unit×Qty)")
    st.dataframe(df[["supplier","category","_spend_eur","calc_spend","consistency_ratio"]].head(50), use_container_width=True)

    # Download
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Lines")
        cat.to_excel(w, index=False, sheet_name="Categories")
        sup.to_excel(w, index=False, sheet_name="Suppliers")
    st.download_button("Download results.xlsx", buf.getvalue(), "results.xlsx")

else:
    st.info("Upload an Excel file to begin. The app will auto-detect the spend column, apply latest ECB FX to EUR, "
            "and display interactive category/supplier/VAVE insights.")
# === END PART 3/3 ===
