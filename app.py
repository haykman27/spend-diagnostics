import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ---------------------- App chrome ----------------------
st.set_page_config(page_title="Procurement Diagnostics — 1-Click", layout="wide")
st.title("Procurement Diagnostics — 1-Click (Auto FX → EUR)")
st.caption("Upload your Excel spend cube. The app auto-detects columns & currencies, fetches ECB FX rates, converts to EUR, and runs the diagnostics.")

# ---------------------- Upload ----------------------
uploaded = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])

# ---------------------- Currency helpers ----------------------
BASE = "EUR"  # fixed base for simplicity

CURRENCY_SYMBOL_MAP = {
    "$":"USD", "US$":"USD", "A$":"AUD", "AU$":"AUD", "C$":"CAD", "CA$":"CAD",
    "€":"EUR", "£":"GBP", "¥":"JPY", "₩":"KRW", "₽":"RUB", "₹":"INR", "₺":"TRY",
    "R$":"BRL", "S$":"SGD", "HK$":"HKD", "₫":"VND", "₪":"ILS", "₱":"PHP", "₦":"NGN",
}
ISO_3 = {
    "EUR","USD","GBP","JPY","CNY","CHF","SEK","NOK","DKK","PLN","HUF","CZK","RON",
    "AUD","NZD","CAD","MXN","BRL","ZAR","AED","SAR","HKD","SGD","INR","TRY","KRW","TWD","THB","PHP","ILS","NGN","VND","RUB"
}

def detect_iso_from_text(text: str):
    if text is None or (isinstance(text, float) and np.isnan(text)): return None
    s = str(text).strip()
    if not s: return None
    # explicit 3-letter code
    m = re.search(r"\b([A-Za-z]{3})\b", s.upper())
    if m and m.group(1) in ISO_3:
        return m.group(1)
    # symbols (prefer longer matches)
    for sym in sorted(CURRENCY_SYMBOL_MAP.keys(), key=len, reverse=True):
        if sym in s:
            return CURRENCY_SYMBOL_MAP[sym]
    # leading/trailing codes
    m2 = re.match(r"^([A-Za-z]{3})\b", s.upper())
    if m2 and m2.group(1) in ISO_3: return m2.group(1)
    m3 = re.search(r"\b([A-Za-z]{3})$", s.upper())
    if m3 and m3.group(1) in ISO_3: return m3.group(1)
    return None

def parse_price_to_float(x):
    """Parse strings like '€1,234.56', '1.234,56 €', '$12', 'USD 1,200'."""
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x)
    s2 = re.sub(r"[^\d,\.\-]", "", s)
    if "," in s2 and "." in s2:
        s2 = s2.replace(",", "")
    else:
        if "," in s2 and s2.count(",")==1 and len(s2.split(",")[-1]) in (2,3):
            s2 = s2.replace(",", ".")
        else:
            s2 = s2.replace(",", "")
    try:
        return float(s2)
    except:
        return np.nan

# ---------------------- Auto-mapper ----------------------
def normalize_headers(cols):
    return [re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()) for c in cols]

TARGETS = {
    "po_id": ["po", "po id", "po_num", "po number", "purchase order", "order id", "document", "line id", "invoice line id"],
    "date": ["date", "posting date", "order date", "document date", "invoice date"],
    "category": ["category", "commodity", "spend category", "material group", "gl category", "family"],
    "description": ["description", "item", "item description", "material description", "service description", "short text", "long text"],
    "supplier": ["supplier", "supplier name", "vendor", "vendor name", "seller", "payee"],
    "sku": ["sku", "material", "material no", "material number", "item code", "product code", "part number", "pn"],
    "unit_price": ["unit price", "price", "unit cost", "net price", "price per unit", "unit gross price"],
    "amount": ["amount", "line amount", "total", "total price", "value", "net value", "extended price"],
    "quantity": ["quantity", "qty", "order qty", "qty ordered", "units", "volume"],
    "currency": ["currency", "ccy", "curr", "currency code", "iso currency"],
    "uom": ["uom", "unit of measure", "unit", "measure", "um"]
}
REQUIRED = ["po_id", "category", "description", "supplier", "quantity", "currency", "uom"]

def detect_columns(df):
    cols = df.columns.tolist()
    norm = normalize_headers(cols)
    back = {n: orig for n, orig in zip(norm, cols)}
    suggestions = {k: (None, 0) for k in TARGETS.keys()}
    for field, synonyms in TARGETS.items():
        best = None; best_score = -1
        for syn in synonyms:
            match, score, _ = process.extractOne(syn, norm, scorer=fuzz.token_sort_ratio)
            if score > best_score:
                best, best_score = match, score
        if best is not None:
            suggestions[field] = (back[best], best_score)
    # date heuristic if confidence low
    if suggestions["date"][1] < 70:
        for c in cols:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.6:
                suggestions["date"] = (c, 80); break
    return suggestions

def mapping_from_suggestions(sug):
    # required fields must exist; if any missing we’ll stop with a friendly message
    mapping = {k: (sug[k][0] if k in sug else None) for k in TARGETS.keys()}
    return mapping

def apply_mapping(df, mapping):
    rename = {v: k for k, v in mapping.items() if v}
    df2 = df.rename(columns=rename).copy()

    if "unit_price" in df2.columns: df2["unit_price"] = df2["unit_price"].apply(parse_price_to_float)
    if "quantity" in df2.columns:   df2["quantity"]   = pd.to_numeric(df2["quantity"], errors="coerce")
    if "amount" in df2.columns:     df2["amount"]     = df2["amount"].apply(parse_price_to_float)

    if "unit_price" not in df2.columns and "amount" in df2.columns and "quantity" in df2.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df2["unit_price"] = df2["amount"] / df2["quantity"]

    df2["quantity"] = df2.get("quantity", 1).fillna(1)

    # supplier canonicalization (light dedupe)
    if "supplier" in df2.columns:
        vals = df2["supplier"].fillna("").astype(str).str.strip()
        canon = {}
        for n in vals.unique():
            if not canon:
                canon[n] = n; continue
            match, score, _ = process.extractOne(n, list(canon.keys()), scorer=fuzz.token_sort_ratio)
            canon[n] = match if score >= 92 else n
        df2["supplier_canon"] = vals.map(canon)
    return df2

# ---------------------- FX: fetch ECB rates ----------------------
@st.cache_data(show_spinner=False, ttl=6*60*60)
def load_ecb_rates():
    """
    Loads historical EUR FX rates from ECB.
    - Source CSV: 1 EUR = X currency (e.g., USD column value is how many USD per 1 EUR)
    - We convert to 'rate_to_eur' per currency = 1 / (EUR->currency)
    """
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"
    fx_wide = pd.read_csv(url)
    fx_wide.rename(columns={"Date":"date"}, inplace=True)
    fx_wide["date"] = pd.to_datetime(fx_wide["date"], errors="coerce")
    # Melt to long
    fx_long = fx_wide.melt(id_vars=["date"], var_name="currency", value_name="eur_to_cur")
    fx_long["currency"] = fx_long["currency"].str.upper().str.strip()
    fx_long["eur_to_cur"] = pd.to_numeric(fx_long["eur_to_cur"], errors="coerce")
    # Build rate: cur -> EUR
    fx_long["rate_to_eur"] = 1.0 / fx_long["eur_to_cur"]
    # Add EUR itself at 1.0
    eur_rows = fx_long[["date"]].drop_duplicates().assign(currency="EUR", rate_to_eur=1.0)
    fx = pd.concat([fx_long[["date","currency","rate_to_eur"]], eur_rows], ignore_index=True)
    fx = fx.dropna(subset=["date","currency","rate_to_eur"]).sort_values(["currency","date"])
    return fx

def merge_fx(df, fx):
    # currency detection
    if "currency" not in df.columns:
        st.error("No currency column detected. Please include a currency column or currency symbols in price text.")
        st.stop()

    df["currency_iso"] = df["currency"].astype(str).apply(lambda x: detect_iso_from_text(x) or BASE)

    # If unit_price is text, parse and possibly refine currency from the price string
    if "unit_price" in df.columns:
        is_text = df["unit_price"].apply(lambda x: isinstance(x, str))
        if is_text.any():
            idxs = df[is_text].index
            detected_from_price = df.loc[idxs, "unit_price"].apply(detect_iso_from_text)
            df.loc[idxs, "currency_iso"] = np.where(
                detected_from_price.notna(), detected_from_price, df.loc[idxs, "currency_iso"]
            )
            df.loc[idxs, "unit_price"] = df.loc[idxs, "unit_price"].apply(parse_price_to_float)

    # Merge FX by date (if available) using last-known rate on/before date; else most-recent per currency
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        out = []
        for cur, g in df.groupby("currency_iso", dropna=False):
            f = fx[fx["currency"]==cur]
            if f.empty:
                rt = 1.0 if cur == BASE else np.nan
                gg = g.assign(rate_to_eur=rt)
            else:
                gg = pd.merge_asof(
                    g.sort_values("date"),
                    f.sort_values("date")[["date","rate_to_eur"]],
                    on="date", direction="backward"
                )
            out.append(gg)
        df = pd.concat(out, ignore_index=True)
    else:
        last_rates = fx.sort_values("date").groupby("currency")["rate_to_eur"].last()
        df["rate_to_eur"] = df["currency_iso"].map(lambda c: 1.0 if c==BASE else last_rates.get(c, np.nan))

    # Fill missing as 1.0 (with warning)
    miss = df["rate_to_eur"].isna().sum()
    if miss:
        st.warning(f"{miss} row(s) have currencies not in ECB table. Treating them as EUR (rate=1.0).")
        df["rate_to_eur"] = df["rate_to_eur"].fillna(1.0)

    # Normalized price/qty
    df["unit_price_eur"] = df.get("unit_price", np.nan) * df["rate_to_eur"]
    df["quantity_norm"] = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
    return df

# ---------------------- Analytics ----------------------
def add_baseline_and_flags(df, iqr_multiplier=1.5, baseline="P50"):
    has_sku = "sku" in df.columns
    key = ["category"] + (["sku"] if has_sku else [])
    grp = df.groupby(key, dropna=False)["unit_price_eur"]
    p50 = grp.median()
    p25 = grp.quantile(0.25)
    base = (p50 if baseline == "P50" else p25).rename("baseline_price").reset_index()
    df = df.merge(base, on=key, how="left")
    df["potential_eur"] = ((df["unit_price_eur"] - df["baseline_price"]).clip(lower=0)) * df["quantity_norm"]

    def iqr_flag(g):
        x = g["unit_price_eur"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 4:
            upper = np.nanpercentile(x, 75) * 1.25 if len(x) else np.nan
        else:
            q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
            upper = q3 + iqr_multiplier * iqr if iqr > 0 else q3 * 1.25
        return g.assign(price_outlier = g["unit_price_eur"] > upper if pd.notna(upper) else False)

    return df.groupby(key, group_keys=False).apply(iqr_flag)

def vave_cluster(df, k=12):
    if "description" not in df.columns:
        return df.assign(vave_cluster=np.nan)
    texts = df["description"].fillna("").astype(str).str.lower()
    if texts.str.len().sum() == 0 or len(texts) < 8:
        return df.assign(vave_cluster=0)
    vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    k = max(2, min(k, max(2, X.shape[0] // 50)))
    labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(X)
    return df.assign(vave_cluster=labels)

# ---------------------- Main flow ----------------------
if uploaded:
    raw = pd.read_excel(uploaded)
    raw.columns = [str(c) for c in raw.columns]

    # 1) auto-detect columns
    suggestions = detect_columns(raw)
    mapping = mapping_from_suggestions(suggestions)

    # minimal validation
    missing_required = [f for f in REQUIRED if not mapping.get(f)]
    if missing_required:
        st.error(f"Could not detect required field(s): {', '.join(missing_required)}. "
                 "Please rename your columns to include these concepts (e.g., 'supplier', 'category', 'quantity', 'currency', 'uom').")
        st.stop()

    # 2) apply mapping and compute unit price if needed
    df = apply_mapping(raw, mapping)

    # sanity: must have unit_price after fallback
    if "unit_price" not in df.columns or df["unit_price"].isna().all():
        st.error("Could not derive unit price. Make sure you have either a unit price column, or amount + quantity.")
        st.stop()

    # 3) fetch FX & convert to EUR
    with st.spinner("Fetching ECB FX rates and converting to EUR..."):
        fx = load_ecb_rates()
        df = merge_fx(df, fx)

    # 4) analytics
    df = add_baseline_and_flags(df, iqr_multiplier=1.5, baseline="P50")
    df = vave_cluster(df, k=12)

    # 5) views
    st.subheader("Results (EUR)")

    df["_spend_eur"] = (df["unit_price_eur"] * df["quantity_norm"]).fillna(0.0)

    cat = df.groupby("category", dropna=False).agg(
        spend_eur=("_spend_eur", "sum"),
        potential_eur=("potential_eur", "sum"),
        lines=("po_id", "count"),
        suppliers=("supplier_canon", pd.Series.nunique)
    ).reset_index()
    cat["potential_pct"] = np.where(cat["spend_eur"]>0, cat["potential_eur"]/cat["spend_eur"], 0.0)
    st.markdown("**Category savings potential**")
    st.dataframe(cat.sort_values("potential_eur", ascending=False), use_container_width=True)

    sup = (
        df[df["price_outlier"]]
        .groupby("supplier_canon", dropna=False)
        .agg(flagged_lines=("po_id","count"),
             flagged_potential_eur=("potential_eur","sum"))
        .reset_index()
        .sort_values("flagged_potential_eur", ascending=False)
    )
    st.markdown("**Supplier negotiation hotlist (overpriced flags)**")
    st.dataframe(sup, use_container_width=True)

    vave = (
        df.groupby(["category","vave_cluster"], dropna=False)
        .agg(
            cluster_lines=("po_id","count"),
            suppliers=("supplier_canon", pd.Series.nunique),
            price_dispersion=("unit_price_eur",
                              lambda x: np.nanpercentile(x,75)-np.nanpercentile(x,25))
        )
        .reset_index()
        .sort_values(["price_dispersion","cluster_lines"], ascending=False)
    )
    st.markdown("**VAVE clusters (standardize/substitute here first)**")
    st.dataframe(vave.head(50), use_container_width=True)

    # 6) download pack
    st.markdown("#### Download pack (XLSX)")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.drop(columns=["_spend_eur"], errors="ignore").to_excel(writer, index=False, sheet_name="line_level")
        cat.to_excel(writer, index=False, sheet_name="category_view")
        sup.to_excel(writer, index=False, sheet_name="supplier_hotlist")
        vave.to_excel(writer, index=False, sheet_name="vave_clusters")
    st.download_button("Download results (EUR).xlsx", data=buffer.getvalue(),
                       file_name="procurement_diagnostics_results_EUR.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload an Excel file to begin. The app will auto-detect currencies and fetch ECB FX rates to convert everything into EUR.")
