import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(page_title="Procurement Diagnostics — Auto-Mapper + FX", layout="wide")
st.title("Procurement Diagnostics — Auto-Mapper (with Currency & FX)")

st.markdown(
    """
Upload an Excel *spend cube* with **any** column names.  
This app will **guess the right columns**, let you confirm, and convert all prices to a **base currency** using:
- **auto-detected currency** per row (from a `currency` column or symbols like `$`, `€`, `£`)
- an **optional FX table** you can upload (by date & currency) to get historical rates.

**Target fields**  
**Required:** `po_id`, `category`, `description`, `supplier`, `quantity`, (`unit_price` or `amount`), `currency`, `uom`  
**Optional:** `date`, `sku`
"""
)

# ---------- Sidebar normalization ----------
with st.sidebar:
    st.header("Normalization")
    base_currency = st.selectbox("Base currency", ["EUR","USD","GBP","CHF","SEK","NOK","DKK","PLN","HUF","CZK","RON","JPY","CNY","AUD","NZD","CAD","MXN","BRL","ZAR","AED","SAR"])
    uom_factor = st.number_input("UoM multiplier", value=1.0, min_value=0.000001, format="%.6f")
    baseline_choice = st.selectbox("Baseline method", ["Median (P50)", "P25 (lower quartile)"])
    iqr_multiplier = st.number_input("Outlier IQR multiplier", value=1.5, step=0.1)
    cluster_count = st.slider("VAVE cluster count", 6, 24, 12)
    st.markdown("#### Optional FX table upload")
    fx_file = st.file_uploader("FX rates (CSV/XLSX): columns = date, currency, rate_to_base", type=["csv","xlsx","xls"])

uploaded = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])

# ---------- Currency helpers ----------
CURRENCY_SYMBOL_MAP = {
    "$":"USD", "US$":"USD", "A$":"AUD", "AU$":"AUD", "C$":"CAD", "CA$":"CAD",
    "€":"EUR", "£":"GBP", "¥":"JPY", "₩":"KRW", "₽":"RUB", "₹":"INR", "₺":"TRY",
    "R$":"BRL", "S$":"SGD", "HK$":"HKD", "₫":"VND", "₪":"ILS", "₱":"PHP", "₦":"NGN",
}

ISO_3 = {
    "EUR","USD","GBP","JPY","CNY","CHF","SEK","NOK","DKK","PLN","HUF","CZK","RON",
    "AUD","NZD","CAD","MXN","BRL","ZAR","AED","SAR","HKD","SGD","INR","TRY","KRW","TWD","THB","PHP","ILS","NGN","VND","RUB"
}

def normalize_headers(cols):
    out = []
    for c in cols:
        out.append(re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()))
    return out

def detect_iso_from_text(text: str):
    if text is None or (isinstance(text, float) and np.isnan(text)): return None
    s = str(text).strip()
    if not s: return None
    # Try explicit 3-letter ISO
    m = re.search(r"\b([A-Za-z]{3})\b", s.upper())
    if m:
        cand = m.group(1)
        if cand in ISO_3:
            return cand
    # Try symbols / prefixes
    # check longest symbol first
    for sym in sorted(CURRENCY_SYMBOL_MAP.keys(), key=len, reverse=True):
        if sym in s:
            return CURRENCY_SYMBOL_MAP[sym]
    # Common textual forms
    s_clean = s.replace("dollars","").replace("euro","").replace("euros","").strip()
    # Heuristic: leading or trailing currency code like "USD 10" or "10 USD"
    m2 = re.match(r"^([A-Za-z]{3})\b", s_clean.upper())
    if m2 and m2.group(1) in ISO_3: return m2.group(1)
    m3 = re.search(r"\b([A-Za-z]{3})$", s_clean.upper())
    if m3 and m3.group(1) in ISO_3: return m3.group(1)
    return None

def parse_price_to_float(x):
    """Handle strings like '€1,234.56' or '1.234,56 €' or '$12'."""
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x)
    # Remove currency symbols/letters except separators and minus
    s2 = re.sub(r"[^\d,\.\-]", "", s)
    # Heuristic: if both , and . present, assume , is thousands and . is decimal
    if "," in s2 and "." in s2:
        s2 = s2.replace(",", "")
    else:
        # If only ',' present and appears as decimal (e.g., '12,34'), replace with '.'
        if "," in s2 and s2.count(",")==1 and len(s2.split(",")[-1]) in (2,3):
            s2 = s2.replace(",", ".")
        else:
            s2 = s2.replace(",", "")
    try:
        return float(s2)
    except:
        return np.nan

# ---------- Auto-mapper ----------
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
    "currency": ["currency", "ccy", "curr", "currency code", "iso currency"]
}
REQUIRED = ["po_id", "category", "description", "supplier", "quantity", "currency", "uom"]
OPTIONAL = ["date", "sku", "unit_price", "amount"]

def detect_columns(df):
    cols = df.columns.tolist()
    norm = normalize_headers(cols)
    back = {n: orig for n, orig in zip(norm, cols)}
    suggestions = {k: (None, 0) for k in TARGETS.keys()}
    suggestions["uom"] = (None, 0)  # uom handled below

    for field, synonyms in TARGETS.items():
        best = None; best_score = -1
        for syn in synonyms:
            match, score, _ = process.extractOne(syn, norm, scorer=fuzz.token_sort_ratio)
            if score > best_score:
                best, best_score = match, score
        if best is not None:
            suggestions[field] = (back[best], best_score)

    # UoM heuristic
    for cand in cols:
        if re.search(r"\b(uom|unit of measure|measure|um|unit)\b", str(cand).lower()):
            suggestions["uom"] = (cand, 80); break

    # Date heuristic if needed
    if suggestions["date"][1] < 70:
        for c in cols:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.6:
                suggestions["date"] = (c, 80)
                break
    return suggestions

def build_mapping_ui(df, suggestions):
    st.subheader("Step 1 — Confirm column mapping")
    cols = ["— none —"] + df.columns.tolist()

    def pick(label, key, required=False):
        guess = suggestions.get(key, (None,0))[0]
        label_required = f"{label} *" if required else label
        return st.selectbox(label_required, cols, index=(cols.index(guess) if guess in cols else 0), key=f"map_{key}")

    mapping = {
        "po_id": pick("PO / Line ID", "po_id", True),
        "date": pick("Date", "date", False),
        "category": pick("Category", "category", True),
        "description": pick("Description", "description", True),
        "supplier": pick("Supplier", "supplier", True),
        "sku": pick("SKU / Material (optional)", "sku", False),
        "unit_price": pick("Unit Price", "unit_price", False),
        "amount": pick("Line Amount / Total", "amount", False),
        "quantity": pick("Quantity", "quantity", True),
        "currency": pick("Currency", "currency", True),
        "uom": pick("UoM", "uom", True),
    }

    missing = [k for k in REQUIRED if mapping[k] == "— none —"]
    if mapping["unit_price"] == "— none —" and mapping["amount"] == "— none —":
        missing.append("unit_price_or_amount")

    if missing:
        st.info("Select required fields (*) and either Unit Price **or** Amount.")
    proceed = st.button("Apply mapping")
    return mapping, proceed, missing

def load_fx_table(fx_file, base_currency):
    if fx_file is None:
        return None
    if fx_file.name.lower().endswith(".csv"):
        fx = pd.read_csv(fx_file)
    else:
        fx = pd.read_excel(fx_file)
    fx = fx.rename(columns={c: c.strip().lower() for c in fx.columns})
    required_cols = {"date","currency","rate_to_base"}
    if not required_cols.issubset(set(fx.columns)):
        st.error("FX file must have columns: date, currency, rate_to_base")
        return None
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
    fx["currency"] = fx["currency"].str.upper().str.strip()
    fx = fx.sort_values(["currency","date"])
    # Ensure base currency rate == 1.0 if provided
    fx.loc[fx["currency"]==base_currency, "rate_to_base"] = 1.0
    return fx

def merge_fx(df, fx, base_currency):
    # Detect currency per row -> currency_iso
    cur_col = "currency"
    df[cur_col] = df[cur_col].astype(str)

    # If currency column is mixed (e.g., contains symbols), resolve to ISO
    df["currency_iso"] = df[cur_col].apply(lambda x: detect_iso_from_text(x) or base_currency)

    # If unit_price is text with currency, try to refine currency_iso and parse number
    if "unit_price" in df.columns:
        unit_txt_mask = df["unit_price"].apply(lambda x: isinstance(x, str))
        if unit_txt_mask.any():
            # try detect iso from price text where currency_iso is base (uncertain)
            idxs = df[unit_txt_mask].index
            detected_from_price = df.loc[idxs, "unit_price"].apply(detect_iso_from_text)
            df.loc[idxs, "currency_iso"] = np.where(
                detected_from_price.notna(), detected_from_price, df.loc[idxs, "currency_iso"]
            )
            # parse numeric
            df.loc[idxs, "unit_price"] = df.loc[idxs, "unit_price"].apply(parse_price_to_float)

    # Build per-row rate_to_base
    if fx is not None:
        # If we have dates, do asof-merge (last known rate on/before date) per currency
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            fx = fx.dropna(subset=["date"]).sort_values(["currency","date"])
            # Use merge_asof per currency
            out = []
            for cur, g in df.groupby("currency_iso", dropna=False):
                fxg = fx[fx["currency"]==cur]
                if fxg.empty:
                    # fallback: 1.0 for base, NaN for others
                    rt = 1.0 if cur == base_currency else np.nan
                    gg = g.assign(rate_to_base=rt)
                else:
                    gg = pd.merge_asof(
                        g.sort_values("date"),
                        fxg[["date","rate_to_base"]].sort_values("date"),
                        on="date", direction="backward"
                    )
                out.append(gg)
            df = pd.concat(out, ignore_index=True)
        else:
            # No dates -> use most recent rate per currency
            last_rates = fx.sort_values("date").groupby("currency")["rate_to_base"].last()
            df["rate_to_base"] = df["currency_iso"].map(lambda c: 1.0 if c==base_currency else last_rates.get(c, np.nan))
    else:
        # No FX uploaded: base->1.0, others->1.0 (warn later)
        df["rate_to_base"] = np.where(df["currency_iso"]==base_currency, 1.0, 1.0)

    missing_rates = df["rate_to_base"].isna().sum()
    if missing_rates > 0:
        st.warning(f"{missing_rates} row(s) have no FX rate. They will be treated as base currency for now (rate=1.0).")
        df["rate_to_base"] = df["rate_to_base"].fillna(1.0)

    # Compute normalized price and quantity
    df["unit_price_norm"] = df.get("unit_price", np.nan) * df["rate_to_base"]
    return df

def apply_mapping(df, mapping):
    rename = {v: k for k, v in mapping.items() if v and v != "— none —"}
    df2 = df.rename(columns=rename).copy()

    # Numeric coercions & fallbacks
    if "unit_price" in df2.columns: df2["unit_price"] = df2["unit_price"].apply(parse_price_to_float)
    if "quantity" in df2.columns:   df2["quantity"]   = pd.to_numeric(df2["quantity"], errors="coerce")
    if "amount" in df2.columns:     df2["amount"]     = df2["amount"].apply(parse_price_to_float)

    # Compute unit_price if missing but amount+quantity present
    if "unit_price" not in df2.columns and "amount" in df2.columns and "quantity" in df2.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df2["unit_price"] = df2["amount"] / df2["quantity"]

    df2["quantity"] = df2.get("quantity", 1).fillna(1)
    # supplier canonicalization
    if "supplier" in df2.columns:
        vals = df2["supplier"].fillna("").astype(str).str.strip()
        canon_map = {}
        for n in vals.unique():
            if not canon_map:
                canon_map[n] = n
                continue
            match, score, _ = process.extractOne(n, list(canon_map.keys()), scorer=fuzz.token_sort_ratio)
            canon_map[n] = match if score >= 92 else n
        df2["supplier_canon"] = vals.map(canon_map)

    # UoM normalization
    df2["quantity_norm"] = df2["quantity"] * uom_factor
    return df2

def add_baseline_and_flags(df):
    has_sku = "sku" in df.columns
    key = ["category"] + (["sku"] if has_sku else [])
    grp = df.groupby(key, dropna=False)["unit_price_norm"]
    p50 = grp.median()
    p25 = grp.quantile(0.25)
    base = (p50 if baseline_choice.startswith("Median") else p25).rename("baseline_price").reset_index()
    df = df.merge(base, on=key, how="left")
    df["potential_eur"] = ((df["unit_price_norm"] - df["baseline_price"]).clip(lower=0)) * df["quantity_norm"]

    def iqr_flag(g):
        x = g["unit_price_norm"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 4:
            upper = np.nanpercentile(x, 75) * 1.25 if len(x) else np.nan
        else:
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            upper = q3 + iqr_multiplier * iqr if iqr > 0 else q3 * 1.25
        return g.assign(price_outlier = g["unit_price_norm"] > upper if pd.notna(upper) else False)

    df = df.groupby(key, group_keys=False).apply(iqr_flag)
    return df

def vave_cluster(df, k):
    if "description" not in df.columns:
        return df.assign(vave_cluster=np.nan)
    texts = df["description"].fillna("").astype(str).str.lower()
    if texts.str.len().sum() == 0 or len(texts) < 8:
        return df.assign(vave_cluster=0)
    vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    k = max(2, min(k, max(2, X.shape[0] // 50)))
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(X)
    return df.assign(vave_cluster=labels)

# ---------- Main flow ----------
if uploaded:
    raw = pd.read_excel(uploaded)
    raw.columns = [str(c) for c in raw.columns]

    suggestions = detect_columns(raw)
    mapping, proceed, missing = build_mapping_ui(raw, suggestions)

    fx = load_fx_table(fx_file, base_currency)

    if proceed and not missing:
        df = apply_mapping(raw, mapping)

        # FX merge & currency detection
        if "currency" not in df.columns:
            st.error("Currency column is required for FX conversion. Please map it above.")
            st.stop()
        df = merge_fx(df, fx, base_currency)

        # Sanity checks
        req_present = all(k in df.columns for k in ["po_id","category","description","supplier","quantity","unit_price_norm","uom"])
        if not req_present:
            st.error("Required fields missing after mapping. Please adjust selections above.")
            st.stop()

        if fx is None:
            st.warning("No FX table uploaded. All currencies are treated as base (=1.0). Upload FX to convert non-base currencies correctly.")

        # Analytics
        df = add_baseline_and_flags(df)
        df = vave_cluster(df, cluster_count)

        # Views
        st.subheader(f"Step 2 — Results (in {base_currency})")

        spend = (df["unit_price_norm"] * df["quantity_norm"]).fillna(0.0)
        df["_spend_base"] = spend

        cat = df.groupby("category", dropna=False).agg(
            spend_base=("_spend_base", "sum"),
            potential_base=("potential_eur", "sum"),
            lines=("po_id", "count"),
            suppliers=("supplier_canon", pd.Series.nunique)
        ).reset_index()
        cat["potential_pct"] = np.where(cat["spend_base"]>0, cat["potential_base"]/cat["spend_base"], 0.0)
        st.markdown("**Category savings potential**")
        st.dataframe(cat.sort_values("potential_base", ascending=False), use_container_width=True)

        sup = (
            df[df["price_outlier"]]
            .groupby("supplier_canon", dropna=False)
            .agg(flagged_lines=("po_id","count"),
                 flagged_potential_base=("potential_eur","sum"))
            .reset_index()
            .sort_values("flagged_potential_base", ascending=False)
        )
        st.markdown("**Supplier negotiation hotlist (overpriced flags)**")
        st.dataframe(sup, use_container_width=True)

        vave = (
            df.groupby(["category","vave_cluster"], dropna=False)
            .agg(
                cluster_lines=("po_id","count"),
                suppliers=("supplier_canon", pd.Series.nunique),
                price_dispersion=("unit_price_norm",
                                  lambda x: np.nanpercentile(x,75)-np.nanpercentile(x,25))
            )
            .reset_index()
            .sort_values(["price_dispersion","cluster_lines"], ascending=False)
        )
        st.markdown("**VAVE clusters (standardize/substitute here first)**")
        st.dataframe(vave.head(50), use_container_width=True)

        # Download pack
        st.markdown("#### Download pack (XLSX)")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.drop(columns=["_spend_base"], errors="ignore").to_excel(writer, index=False, sheet_name="line_level")
            cat.to_excel(writer, index=False, sheet_name="category_view")
            sup.to_excel(writer, index=False, sheet_name="supplier_hotlist")
            vave.to_excel(writer, index=False, sheet_name="vave_clusters")
        st.download_button(f"Download results_{base_currency}.xlsx", data=buffer.getvalue(),
                           file_name=f"procurement_diagnostics_results_{base_currency}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    elif uploaded is not None:
        st.info("Select the correct columns above, then click **Apply mapping**.")
else:
    st.info("Upload an Excel file to begin. Any column names are fine — the app will guess and let you confirm.")
