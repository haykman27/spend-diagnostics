# app.py
import io
import pandas as pd
import numpy as np
import streamlit as st
from rapidfuzz import process, fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Procurement Diagnostics", layout="wide")

st.title("Procurement Diagnostics MVP")

uploaded = st.file_uploader("Upload spend cube (Excel)", type=["xlsx", "xls"])
base_currency_rate = st.number_input("FX: 1 uploaded currency = ? EUR", value=1.0)
uom_factor = st.number_input("UoM factor to normalize quantities (optional)", value=1.0)

def canonicalize_suppliers(df, col="supplier", threshold=92):
    names = df[col].dropna().unique().tolist()
    canon = {}
    for n in names:
        if n in canon: 
            continue
        match, score, _ = process.extractOne(n, list(canon.keys()), scorer=fuzz.token_sort_ratio) if canon else (None, 0, None)
        if score >= threshold:
            canon[n] = match
        else:
            canon[n] = n
    return df.assign(supplier_canon=df[col].map(canon))

def compute_baselines(df):
    # baseline by category+sku if sku exists else by category+description bucket
    key_cols = ["category"] + (["sku"] if "sku" in df.columns else [])
    grp = df.groupby(key_cols, dropna=False)["unit_price_norm"]
    baseline = grp.median().rename("baseline_price").reset_index()
    return df.merge(baseline, on=key_cols, how="left")

def outlier_flags(df):
    key_cols = ["category"] + (["sku"] if "sku" in df.columns else [])
    def flag(group):
        x = group["unit_price_norm"].astype(float)
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1 if q3 > q1 else 0.0
        upper = q3 + 1.5*iqr if iqr > 0 else q3*1.25
        return group.assign(price_outlier=(group["unit_price_norm"] > upper))
    return df.groupby(key_cols, group_keys=False).apply(flag)

def potential_calc(df):
    df["potential_eur"] = np.maximum(0, df["unit_price_norm"] - df["baseline_price"]) * df["quantity_norm"]
    return df

def vave_clusters(df, n_clusters=12):
    # quick-and-dirty: TF-IDF-like bag of words on top tokens from description length filter
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = df["description"].fillna("").astype(str).str.lower()
    vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    if X.shape[0] < n_clusters:
        n_clusters = max(2, X.shape[0] // 5) if X.shape[0] >= 10 else 2
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    return df.assign(vave_cluster=labels)

if uploaded:
    df = pd.read_excel(uploaded)
    # expected columns
    expected = ["po_id","date","category","description","supplier","unit_price","quantity","currency","uom"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.warning(f"Missing columns: {missing}. The app will try to proceed with what’s available.")

    # normalize
    df["unit_price"] = pd.to_numeric(df.get("unit_price", np.nan), errors="coerce")
    df["quantity"] = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
    df["unit_price_norm"] = df["unit_price"] / base_currency_rate
    df["quantity_norm"] = df["quantity"] * uom_factor
    if "supplier" in df.columns:
        df = canonicalize_suppliers(df, "supplier")

    # baselines & outliers
    df = compute_baselines(df)
    df = outlier_flags(df)
    df = potential_calc(df)

    # VAVE clusters
    if "description" in df.columns:
        df = vave_clusters(df)

    # Category view
    cat = df.groupby("category", dropna=False).agg(
        spend_eur=("unit_price_norm", lambda x: float(np.nansum(x * df.loc[x.index,"quantity_norm"]))),
        potential_eur=("potential_eur", "sum"),
        lines=("po_id", "count"),
        suppliers=("supplier_canon", pd.Series.nunique)
    ).reset_index()
    cat["potential_pct"] = np.where(cat["spend_eur"]>0, cat["potential_eur"]/cat["spend_eur"], 0.0)

    st.subheader("Category savings potential")
    st.dataframe(cat.sort_values("potential_eur", ascending=False), use_container_width=True)

    # Supplier hotlist
    sup = df[df["price_outlier"]].groupby("supplier_canon", dropna=False).agg(
        flagged_lines=("po_id","count"),
        flagged_potential_eur=("potential_eur","sum")
    ).reset_index().sort_values("flagged_potential_eur", ascending=False)
    st.subheader("Supplier negotiation hotlist (high-price flags)")
    st.dataframe(sup, use_container_width=True)

    # VAVE opportunities
    vave = df.groupby(["category","vave_cluster"], dropna=False).agg(
        cluster_lines=("po_id","count"),
        suppliers=("supplier_canon", pd.Series.nunique),
        price_dispersion=("unit_price_norm", lambda x: np.nanpercentile(x,75)-np.nanpercentile(x,25))
    ).reset_index().sort_values(["price_dispersion","cluster_lines"], ascending=False)
    st.subheader("VAVE clusters (standardize/substitute here first)")
    st.dataframe(vave.head(50), use_container_width=True)

    # Downloads
    with pd.ExcelWriter(io.BytesIO(), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="line_level")
        cat.to_excel(writer, index=False, sheet_name="category_view")
        sup.to_excel(writer, index=False, sheet_name="supplier_hotlist")
        vave.to_excel(writer, index=False, sheet_name="vave_clusters")
        out = writer.book  # placeholder to keep context
    st.caption("Use the 'Download' button in the top-right of each table to export as CSV. For XLSX packaging, add a download button with BytesIO if desired.")
else:
    st.info("Upload an Excel spend cube to begin.")
