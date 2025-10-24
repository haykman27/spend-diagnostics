# Procurement Diagnostics — Final (Auto FX → EUR)
# Sustainable spend detection, robust FX, interactive tables (€ k), and all analytics preserved.

import io, re
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="Procurement Diagnostics — Final", layout="wide")
st.title("Procurement Diagnostics — Final (Auto FX → EUR)")
st.caption(
    "Upload your Excel spend cube. The app auto-detects key columns, fixes decimals, detects spend, "
    "converts to EUR (latest ECB FX), and shows diagnostics in € k."
)
uploaded = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])
BASE = "EUR"

# ---------------- Helpers ----------------
def normalize_headers(cols):
    return [re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()) for c in cols]

def parse_price_to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float)): return float(x)
    s=re.sub(r"[^\d,\.\-]","",str(x))
    if "," in s and "." in s: s=s.replace(",","")
    elif "," in s and s.count(",")==1 and len(s.split(",")[-1]) in (2,3): s=s.replace(",",".")
    else: s=s.replace(",","")
    try: return float(s)
    except: return np.nan

CURRENCY_SYMBOL_MAP={"€":"EUR","$":"USD","£":"GBP","¥":"JPY","₩":"KRW","₹":"INR","₺":"TRY","R$":"BRL","S$":"SGD"}
ISO_3={"EUR","USD","GBP","JPY","CNY","CHF","SEK","NOK","DKK","PLN","HUF","CZK","RON",
       "AUD","NZD","CAD","MXN","BRL","ZAR","AED","SAR","HKD","SGD","INR","TRY","KRW",
       "TWD","THB","PHP","ILS","NGN","VND","RUB"}
def detect_iso_from_text(text):
    if text is None or (isinstance(text,float) and np.isnan(text)): return None
    s=str(text).upper().strip()
    alias={"RMB":"CNY","YUAN":"CNY","CN¥":"CNY","ZŁ":"PLN","ZL":"PLN","KČ":"CZK","LEI":"RON"}
    m=re.search(r"\b([A-Z]{3})\b",s)
    if m:
        c=m.group(1)
        if c in ISO_3: return c
        if c in alias: return alias[c]
    for k,v in alias.items():
        if k in s: return v
    for sym in sorted(CURRENCY_SYMBOL_MAP,key=len,reverse=True):
        if sym in s: return CURRENCY_SYMBOL_MAP[sym]
    return None

# ------------- Load latest ECB FX (CSV, robust) -------------
@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_latest_ecb():
    url="https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"
    fx=pd.read_csv(url)
    fx.rename(columns={"Date":"date"},inplace=True)
    fx["date"]=pd.to_datetime(fx["date"],errors="coerce")
    latest=fx["date"].max()
    row=fx.loc[fx["date"]==latest].melt(id_vars=["date"],var_name="currency",value_name="eur_to_cur")
    row["currency"]=row["currency"].str.upper().str.strip()
    row["eur_to_cur"]=pd.to_numeric(row["eur_to_cur"],errors="coerce")
    row=row.dropna(subset=["eur_to_cur"])
    row["rate_to_eur"]=1/row["eur_to_cur"]
    eur=pd.DataFrame([{"currency":"EUR","rate_to_eur":1.0}])
    fx=pd.concat([row[["currency","rate_to_eur"]],eur]).drop_duplicates(subset=["currency"])
    return fx

def apply_fx_latest(df,fx):
    df["currency_iso"]=df["currency"].astype(str).apply(lambda x:detect_iso_from_text(x)or BASE)
    df=df.merge(fx,left_on="currency_iso",right_on="currency",how="left")
    df["rate_to_eur"]=df["rate_to_eur"].fillna(1.0)
    return df

# ------------- Column mapping ----------------
TARGETS={
 "category":["category","commodity","spend category","material group","material group desc","gl category",
             "family","family group","group","sub category","item family","item family group","item group",
             "procurement group","cluster"],
 "supplier":["supplier","supplier name","vendor","vendor name","seller","payee"],
 "currency":["currency","ccy","curr","currency code","iso currency"],
 "unit_price":["unit price","price","unit cost","net price","price per unit","unit gross price"],
 "quantity":["quantity","qty","order qty","qty ordered","units","volume"],
 "amount":["amount","line amount","total","total price","value","net value","extended price"]
}
def suggest_columns(df):
    cols=df.columns.tolist()
    norm=normalize_headers(cols)
    back={n:o for n,o in zip(norm,cols)}
    out={}
    for k,syns in TARGETS.items():
        best,score=None,-1
        for s in syns:
            m=process.extractOne(s,norm,scorer=fuzz.token_sort_ratio)
            if m and m[1]>score: best,score=m[0],m[1]
        if best: out[k]=back[best]
    return out

# -------- Sustainable spend-column detection --------
SPEND_NAME_CUES=["purchase amount","po amount","line total","line value","total value","net value","gross amount",
                 "extended price","spend","base curr","base currency","global curr","local curr","invoice value","base curr"]
def detect_spend_column(df):
    hits=[c for c in df.columns if any(k in c.lower() for k in SPEND_NAME_CUES)]
    if not hits: return None
    if len(hits)==1: return hits[0]
    med={c:pd.to_numeric(df[c].apply(parse_price_to_float),errors="coerce").median(skipna=True) for c in hits}
    return max(med,key=med.get)

# -------- Savings ranges & VAVE --------
SAVINGS={"stampings":(0.10,0.15),"tubes":(0.08,0.14),"plast":(0.08,0.18),"steel":(0.05,0.12),"logist":(0.08,0.15)}
VAVE={"stampings":["Standardize steel grades","Improve sheet nesting","Relax tolerances","Bundle volumes"],
      "tubes":["Standardize diameters","Use HF-welded","Loosen length tolerance"],
      "plast":["Consolidate resins","Family molds","Standard inserts"],
      "steel":["Standardize material specs","Use index-linked pricing"],
      "logist":["Optimize mode mix","Increase load factor"]}
def savings_for(c):
    c=(c or "").lower()
    for k,v in SAVINGS.items():
        if k in c:return v
    return (0.05,0.10)
def vave_for(c):
    c=(c or "").lower()
    for k,v in VAVE.items():
        if k in c:return v
    return ["Standardize specs","Consolidate variants","Bundle volumes"]

def fmt_k(series): return (series/1_000).round(0)

# -------- Outlier baseline --------
def add_baseline(df):
    if "unit_price_eur" not in df:return df
    base=df.groupby("category")["unit_price_eur"].median().rename("baseline_price").reset_index()
    df=df.merge(base,on="category",how="left")
    if "quantity" in df.columns:
        df["potential_eur"]=((df["unit_price_eur"]-df["baseline_price"]).clip(lower=0))*df["quantity"]
    return df

# ======================================================
# ------------------------ MAIN ------------------------
# ======================================================
if uploaded:
    raw=pd.read_excel(uploaded)
    raw.columns=[str(c) for c in raw.columns]

    mapping=suggest_columns(raw)
    df=raw.rename(columns={v:k for k,v in mapping.items() if v}).copy()

    spend_col=detect_spend_column(raw)
    if spend_col: st.success(f"Detected '{spend_col}' as spend column.")
    else: st.warning("No clear spend column found; will use Unit Price × Quantity.")

    if "unit_price" in df.columns: df["unit_price"]=df["unit_price"].apply(parse_price_to_float)
    if "quantity" in df.columns: df["quantity"]=pd.to_numeric(df["quantity"],errors="coerce")

    fx=load_latest_ecb()
    df["currency"]=df.get("currency","")
    df=apply_fx_latest(df,fx)

    # === Spend computation block (prevents double FX) ===
    def _is_global_header(c): return any(k in c.lower() for k in ["base curr","base currency","global curr","reporting curr"])
    def _find_base_ccy_col(df_raw,spend_col):
        cands=[c for c in df_raw.columns if c!=spend_col and any(k in c.lower() for k in
               ["base curr code","base currency code","base curr","base currency","reporting curr","reporting currency","global curr","global currency"])]
        return cands[0] if cands else None

    if "unit_price" in df.columns: df["unit_price_eur"]=df["unit_price"]*df["rate_to_eur"]
    else: df["unit_price_eur"]=np.nan

    if spend_col:
        spend_vals=raw[spend_col].apply(parse_price_to_float)
        if _is_global_header(spend_col):
            base_col=_find_base_ccy_col(raw,spend_col)
            if base_col:
                base_iso=raw[base_col].astype(str).apply(detect_iso_from_text).fillna("EUR")
                fx_map=dict(zip(fx["currency"],fx["rate_to_eur"]))
                base_rate=base_iso.map(lambda c:fx_map.get(c,1.0))
                df["_spend_eur"]=(pd.to_numeric(spend_vals,errors="coerce")*base_rate).fillna(0.0)
            else:
                df["_spend_eur"]=pd.to_numeric(spend_vals,errors="coerce").fillna(0.0)
        else:
            df["_spend_eur"]=(pd.to_numeric(spend_vals,errors="coerce")*df["rate_to_eur"]).fillna(0.0)
    else:
        df["_spend_eur"]=(pd.to_numeric(df.get("unit_price"),errors="coerce")*
                          pd.to_numeric(df.get("quantity"),errors="coerce")*
                          df["rate_to_eur"]).fillna(0.0)

    if {"unit_price","quantity"}.issubset(df.columns):
        df["calc_spend"]=df["unit_price"]*df["quantity"]*df["rate_to_eur"]
        df["consistency_ratio"]=np.where(df["calc_spend"]>0,df["_spend_eur"]/df["calc_spend"],np.nan)
    else:
        df["calc_spend"]=np.nan; df["consistency_ratio"]=np.nan

    df=add_baseline(df)

    # === Category Overview ===
    cat=df.groupby("category",dropna=False).agg(
        spend_eur=("_spend_eur","sum"),
        lines=("unit_price","count"),
        suppliers=("supplier",pd.Series.nunique)
    ).reset_index()
    rngs=[savings_for(c) for c in cat["category"]]
    cat["Savings Range (%)"]=[f"{int(r[0]*100)}–{int(r[1]*100)}" for r in rngs]
    cat["Potential Min (€ k)"]=(cat["spend_eur"]*[r[0] for r in rngs]/1_000).round(0)
    cat["Potential Max (€ k)"]=(cat["spend_eur"]*[r[1] for r in rngs]/1_000).round(0)
    cat["Spend (€ k)"]=fmt_k(cat["spend_eur"])
    cat.rename(columns={"category":"Category","lines":"# PO Lines","suppliers":"# Suppliers"},inplace=True)

    st.subheader("1) Category Overview")
    st.dataframe(cat,use_container_width=True,column_config={
        "Spend (€ k)":st.column_config.NumberColumn(format="€ %d k"),
        "Potential Min (€ k)":st.column_config.NumberColumn(format="€ %d k"),
        "Potential Max (€ k)":st.column_config.NumberColumn(format="€ %d k"),
    })

    # === Supplier Drill-Down ===
    st.subheader("2) Supplier Drill-Down")
    chosen=st.selectbox("Choose Category:",options=cat["Category"])
    sup=(df[df["category"]==chosen].groupby("supplier",dropna=False)
         .agg(spend_eur=("_spend_eur","sum"))
         .reset_index()
         .assign(**{"Spend (€ k)":lambda x:fmt_k(x["spend_eur"])})
         .rename(columns={"supplier":"Supplier"})
         .sort_values("Spend (€ k)",ascending=False))
    st.dataframe(sup[["Supplier","Spend (€ k)"]],use_container_width=True,
                 column_config={"Spend (€ k)":st.column_config.NumberColumn(format="€ %d k")})

    # === VAVE Ideas ===
    st.subheader("3) Example VAVE Ideas")
    vrows=[{"Category":c,"Example VAVE Ideas":" • ".join(vave_for(c))} for c in cat["Category"]]
    st.dataframe(pd.DataFrame(vrows),use_container_width=True)

    # === Consistency Check ===
    st.subheader("4) Consistency Check (Spend / Unit×Qty)")
    dbg=df[["supplier","category","_spend_eur","calc_spend","consistency_ratio"]].copy()
    dbg["Spend (€ k)"]=fmt_k(dbg["_spend_eur"]); dbg["Calc (€ k)"]=fmt_k(dbg["calc_spend"])
    st.dataframe(dbg[["supplier","category","Spend (€ k)","Calc (€ k)","consistency_ratio"]],
                 use_container_width=True,column_config={
                     "Spend (€ k)":st.column_config.NumberColumn(format="€ %d k"),
                     "Calc (€ k)":st.column_config.NumberColumn(format="€ %d k")
                 })

    # === Download ===
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        df.to_excel(w,index=False,sheet_name="Lines")
        cat.to_excel(w,index=False,sheet_name="Categories")
        sup.to_excel(w,index=False,sheet_name="Suppliers")
        pd.DataFrame(vrows).to_excel(w,index=False,sheet_name="VAVE")
    st.download_button("Download results.xlsx",buf.getvalue(),"results.xlsx")
else:
    st.info("Upload an Excel file to begin. The app will auto-detect the spend column, apply latest ECB FX to EUR, "
            "and display interactive category, supplier, and VAVE analyses with spends formatted as € k.")
