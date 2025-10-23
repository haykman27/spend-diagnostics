import io, re
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# ================= App chrome =================
st.set_page_config(page_title="Procurement Diagnostics — Full", layout="wide")
st.title("Procurement Diagnostics — Full (Auto FX → EUR)")
st.caption("1 upload • auto-map columns • ECB FX to EUR • robust spend • category savings • supplier drill-down • VAVE ideas • sanity checks")

uploaded = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])
BASE = "EUR"  # fixed base

# ================= Currency helpers =================
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
    """Detect ISO-3 currency code from free text/symbols/aliases."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    s = str(text).strip()
    if not s:
        return None
    s_up = s.upper()

    # Common aliases not in ECB 3-letter text
    alias = {
        "RMB": "CNY", "YUAN": "CNY", "CN¥": "CNY", "元": "CNY",
        "ZŁ": "PLN", "ZL": "PLN",
        "KČ": "CZK",
        "LEI": "RON",
        "RUR": "RUB", "РУБ": "RUB",
        # WARNING: 'KR' is ambiguous (SEK/NOK/DKK); map only if your data uses it consistently.
    }

    # Explicit 3-letter code in text
    m = re.search(r"\b([A-Z]{3})\b", s_up)
    if m:
        code = m.group(1)
        if code in ISO_3:
            return code
        if code in alias:
            return alias[code]

    # Alias keywords anywhere
    for k, v in alias.items():
        if k in s_up:
            return v

    # Symbols
    for sym in sorted(CURRENCY_SYMBOL_MAP.keys(), key=len, reverse=True):
        if sym in s:
            return CURRENCY_SYMBOL_MAP[sym]

    # Leading/trailing codes
    m2 = re.match(r"^([A-Z]{3})\b", s_up)
    if m2 and (m2.group(1) in ISO_3 or m2.group(1) in alias):
        return m2.group(1) if m2.group(1) in ISO_3 else alias[m2.group(1)]
    m3 = re.search(r"\b([A-Z]{3})$", s_up)
    if m3 and (m3.group(1) in ISO_3 or m3.group(1) in alias):
        return m3.group(1) if m3.group(1) in ISO_3 else alias[m3.group(1)]
    return None

def parse_price_to_float(x):
    """Parse prices like '€1,234.56', '1.234,56 €', 'USD 1,200' into float."""
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = re.sub(r"[^\d,\.\-]", "", str(x))
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        if "," in s and s.count(",") == 1 and len(s.split(",")[-1]) in (2, 3):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

# ================= Auto-mapper =================
def normalize_headers(cols):
    return [re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()) for c in cols]

TARGETS = {
    "po_id":       ["po","po id","po number","purchase order","order id","document","line id","invoice line id"],
    "date":        ["date","posting date","order date","document date","invoice date"],
    "category":    ["category","commodity","spend category","material group","gl category","family","family group","group"],
    "description": ["description","item","item description","material description","service description","short text","long text"],
    "supplier":    ["supplier","supplier name","vendor","vendor name","seller","payee"],
    "sku":         ["sku","material","material no","material number","item code","product code","part number","pn"],
    "unit_price":  ["unit price","price","unit cost","net price","price per unit","unit gross price"],
    "amount":      ["amount","line amount","total","total price","value","net value","extended price"],
    "quantity":    ["quantity","qty","order qty","qty ordered","units","volume"],
    "currency":    ["currency","ccy","curr","currency code","iso currency"],
    "uom":         ["uom","unit of measure","unit","measure","um"],
}

REQUIRED = ["po_id","category","description","supplier","quantity","currency","uom"]

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

    # Prefer "Item family" / "Item family group" as Category
    for pr in ["item family group", "item family"]:
        for n, orig in zip(norm, cols):
            if n == pr:
                suggestions["category"] = (orig, 100)

    # Date heuristic if low confidence
    if suggestions["date"][1] < 70:
        for c in cols:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.6:
                suggestions["date"] = (c, 80); break
    return suggestions

def apply_mapping(df, suggestions):
    mapping = {k: (suggestions[k][0] if k in suggestions else None) for k in TARGETS.keys()}
    missing = [k for k in REQUIRED if not mapping.get(k)]
    if missing:
        st.error("Missing required field(s): " + ", ".join(missing) + ". Please include columns like supplier, quantity, currency, uom and category/item family.")
        st.stop()

    df2 = df.rename(columns={v:k for k,v in mapping.items() if v}).copy()

    # numeric coercion
    if "unit_price" in df2.columns: df2["unit_price"] = df2["unit_price"].apply(parse_price_to_float)
    if "amount"     in df2.columns: df2["amount"]     = df2["amount"].apply(parse_price_to_float)
    if "quantity"   in df2.columns: df2["quantity"]   = pd.to_numeric(df2["quantity"], errors="coerce")

    if "unit_price" not in df2.columns and {"amount","quantity"}.issubset(df2.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            df2["unit_price"] = df2["amount"] / df2["quantity"]

    df2["quantity"] = df2.get("quantity", 1).fillna(1)

    # Supplier canonicalization (light)
    if "supplier" in df2.columns:
        vals = df2["supplier"].fillna("").astype(str).str.strip()
        canon = {}
        for n in vals.unique():
            if not canon: canon[n] = n; continue
            match, score, _ = process.extractOne(n, list(canon.keys()), scorer=fuzz.token_sort_ratio)
            canon[n] = match if score >= 92 else n
        df2["supplier_canon"] = vals.map(canon)

    return df2

# ================= ECB FX (historical) =================
@st.cache_data(show_spinner=False, ttl=6*60*60)
def load_ecb_rates():
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"
    fx_wide = pd.read_csv(url)
    fx_wide.rename(columns={"Date":"date"}, inplace=True)
    fx_wide["date"] = pd.to_datetime(fx_wide["date"], errors="coerce")
    fx_long = fx_wide.melt(id_vars=["date"], var_name="currency", value_name="eur_to_cur")
    fx_long["currency"] = fx_long["currency"].str.upper().str.strip()
    fx_long["eur_to_cur"] = pd.to_numeric(fx_long["eur_to_cur"], errors="coerce")
    fx_long["rate_to_eur"] = 1.0 / fx_long["eur_to_cur"]
    eur_rows = fx_long[["date"]].drop_duplicates().assign(currency="EUR", rate_to_eur=1.0)
    fx = pd.concat([fx_long[["date","currency","rate_to_eur"]], eur_rows], ignore_index=True)
    fx = fx.dropna(subset=["date","currency","rate_to_eur"]).sort_values(["currency","date"])
    return fx

def merge_fx(df, fx):
    if "currency" not in df.columns:
        st.error("No currency column detected. Include a currency column or currency symbols in price text.")
        st.stop()

    # currency per row
    df["currency_iso"] = df["currency"].astype(str).apply(lambda x: detect_iso_from_text(x) or BASE)

    # refine from unit_price text if needed
    if "unit_price" in df.columns:
        is_text = df["unit_price"].apply(lambda x: isinstance(x, str))
        if is_text.any():
            idxs = df[is_text].index
            detected_from_price = df.loc[idxs, "unit_price"].apply(detect_iso_from_text)
            df.loc[idxs, "currency_iso"] = np.where(
                detected_from_price.notna(), detected_from_price, df.loc[idxs, "currency_iso"]
            )
            df.loc[idxs, "unit_price"] = df.loc[idxs, "unit_price"].apply(parse_price_to_float)

    # merge FX
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        out = []
        for cur, g in df.groupby("currency_iso", dropna=False):
            f = fx[fx["currency"]==cur]
            if f.empty:
                gg = g.assign(rate_to_eur=(1.0 if cur==BASE else np.nan))
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

    miss = df["rate_to_eur"].isna().sum()
    if miss:
        st.warning(f"{miss} row(s) have currencies not in ECB table. Treating them as EUR (rate=1.0).")
        df["rate_to_eur"] = df["rate_to_eur"].fillna(1.0)

    df["unit_price_eur"] = df.get("unit_price", np.nan) * df["rate_to_eur"]
    df["amount_eur"]     = pd.to_numeric(df.get("amount"), errors="coerce") * df["rate_to_eur"]
    df["quantity_norm"]  = pd.to_numeric(df.get("quantity"), errors="coerce").fillna(1)
    return df

# ================= Savings library & VAVE ideas =================
SAVINGS_LIBRARY = {
    "stampings": (0.10, 0.15),
    "tubes": (0.08, 0.14),
    "machin": (0.07, 0.14),
    "cast": (0.06, 0.12),
    "forg": (0.05, 0.10),
    "plast": (0.08, 0.18),
    "fasten": (0.05, 0.10),
    "bear": (0.04, 0.08),
    "pcb": (0.06, 0.12), "electro": (0.06, 0.12), "electronics": (0.06, 0.12),
    "cable": (0.06, 0.12), "wire": (0.06, 0.12),
    "packag": (0.05, 0.15),
    "logist": (0.08, 0.15), "freight": (0.08, 0.15),
    "mro": (0.05, 0.12),
    "chem": (0.04, 0.10),
    "tool": (0.05, 0.12),
    "seal": (0.04, 0.09), "gasket": (0.04, 0.09),
    "rubber": (0.06, 0.12),
    "alum": (0.05, 0.12), "steel": (0.05, 0.12),
}
VAVE_IDEAS = {
    "stampings": [
        "Standardize **steel grades**; consolidate to 2–3 specs.",
        "Improve **sheet nesting/yield**; supplier rules & audits.",
        "Relax non-critical **tolerances/radii**; common die sets.",
        "Bundle volumes across plants; consider **progressive die**."
    ],
    "tubes": [
        "Standardize **diameter/wall** to catalog sizes.",
        "Use **HF-welded** if seamless not required.",
        "Loosen **cut-length tolerance** to stock lengths."
    ],
    "machin": [
        "Shift to **near-net** (casting/forging) for high chip ratios.",
        "Reduce **setups**; combine ops; DFM simplifications.",
        "Use **commercial finishes** unless safety/fit needs tighter."
    ],
    "plast": [
        "Consolidate **resins/colors**; masterbatch over pre-colored.",
        "Design **family molds**; increase cavitation.",
        "Use **standard inserts/fasteners**."
    ],
    "fasten": [
        "Rationalize to **ISO/DIN** sizes; eliminate customs.",
        "Adopt **VMI**; convert to kit packaging."
    ],
    "electronics": [
        "Approve **cross-vendor alternates** for passives; multi-source ICs.",
        "Optimize **panelization**; right-size PCB finish/thickness."
    ],
    "packag": [
        "Switch to **standard carton footprints**; improve board yield.",
        "Right-size **flute/grade** to protection needs."
    ],
    "logist": [
        "Optimize **mode mix**; implement **milk-runs** & route planning.",
        "Increase **load factor** with packaging/cube redesign."
    ],
    "mro": [
        "Create **approved catalog** with tiered pricing; consolidate brands.",
        "Set **MOQ** & **rebates**; restrict spot buys."
    ],
    "raw": [
        "Move to **index-linked** pricing; hedge surcharges.",
        "Standardize **material specs** (e.g., 304L vs 316L only where needed)."
    ],
}
def classify_category_for_savings(cat: str):
    c = (cat or "").lower()
    for k, rng in SAVINGS_LIBRARY.items():
        if k in c: return rng, k
    if any(w in c for w in ["steel","alum","aluminum","sheet"]): return SAVINGS_LIBRARY["steel"], "steel"
    if any(w in c for w in ["pcb","electro","electronics"]):     return SAVINGS_LIBRARY["electronics"], "electronics"
    return (0.05, 0.10), "generic"

def vave_ideas_for_category(cat: str):
    c = (cat or "").lower()
    for key, ideas in VAVE_IDEAS.items():
        if key in c: return ideas
    if any(w in c for w in ["steel","alum","aluminum","raw"]): return VAVE_IDEAS["raw"]
    return [
        "Standardize **specs** & tolerances to functional need.",
        "Consolidate variants; prefer **catalog parts**.",
        "Bundle volumes; reduce supplier count; **negotiate**."
    ]

# ================= Outlier baseline (kept) =================
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

# ================= Main flow =================
if uploaded:
    # Read
    raw = pd.read_excel(uploaded)
    raw.columns = [str(c) for c in raw.columns]

    # Auto-map & clean
    suggestions = detect_columns(raw)
    df = apply_mapping(raw, suggestions)

    if "unit_price" not in df.columns or df["unit_price"].isna().all():
        st.error("Could not derive unit price. Ensure you have either a unit price column, or amount + quantity.")
        st.stop()

    # FX → EUR
    with st.spinner("Fetching ECB FX rates and converting to EUR..."):
        fx = load_ecb_rates()
        df = merge_fx(df, fx)

    # Show unknown currency strings (debug)
    unknown = (
        df.loc[df["rate_to_eur"].eq(1.0) & ~df["currency_iso"].eq("EUR"), ["currency","currency_iso"]]
          .assign(sample=lambda x: x["currency"].astype(str).str.slice(0,20))
          .value_counts(subset=["currency","currency_iso"], sort=True)
          .reset_index(name="rows")
    )
    if len(unknown):
        st.warning("Unrecognized/non-ECB currencies (treated as EUR=1.0). Map these in the detector if needed:")
        st.dataframe(unknown.head(50), use_container_width=True)

    # Robust spend (fixed .where/.fillna)
    df["_spend_eur"] = df["amount_eur"].where(
        df["amount_eur"].notna(),
        df["unit_price_eur"] * df["quantity_norm"]
    )
    df["_spend_eur"] = df["_spend_eur"].fillna(0.0)

    # Sanity checks
    suspicious_suppliers = (
        df.groupby("supplier_canon", dropna=False)
          .agg(spend_eur=("_spend_eur","sum"), lines=("po_id","count"))
          .query("(spend_eur < 100) and (lines >= 3)")
    )
    too_small_prices = (df["unit_price_eur"].dropna() < 0.05).mean() > 0.4
    decimal_confusion_hint = df.get("unit_price", pd.Series([], dtype="object")).astype(str).str.contains(r",\d{3}\b").any()

    msgs = []
    if len(suspicious_suppliers):
        msgs.append(f"Suppliers with very low total spend: {', '.join(suspicious_suppliers.index.tolist()[:10])} … check units/decimals.")
    if too_small_prices:
        msgs.append("Many unit prices are under €0.05 — decimal or pricing unit may be off.")
    if decimal_confusion_hint:
        msgs.append("Detected values like '1,234' that may be thousands separators — check decimal conventions.")
    if msgs:
        st.warning("Sanity checks:\n- " + "\n- ".join(msgs))

    # Baseline + flags (kept)
    df = add_baseline_and_flags(df, iqr_multiplier=1.5, baseline="P50")

    # VIEW 1: Categories with typical savings
    cat = df.groupby("category", dropna=False).agg(
        spend_eur=("_spend_eur","sum"),
        lines=("po_id","count"),
        suppliers=("supplier_canon", pd.Series.nunique)
    ).reset_index()

    ranges, buckets = [], []
    for cat_name in cat["category"]:
        rng, bucket = classify_category_for_savings(str(cat_name))
        ranges.append(rng); buckets.append(bucket)
    cat["typical_savings_min_pct"] = [r[0] for r in ranges]
    cat["typical_savings_max_pct"] = [r[1] for r in ranges]
    cat["potential_min_eur"] = cat["spend_eur"] * cat["typical_savings_min_pct"]
    cat["potential_max_eur"] = cat["spend_eur"] * cat["typical_savings_max_pct"]
    cat["spend_eur_k"] = cat["spend_eur"] / 1_000.0
    cat["spend_eur_m"] = cat["spend_eur"] / 1_000_000.0
    cat["bucket"] = buckets

    st.subheader("1) Categories — Spend and Typical Savings (EUR)")
    st.dataframe(
        cat.sort_values("spend_eur", ascending=False)[
            ["category","spend_eur","spend_eur_k","spend_eur_m",
             "typical_savings_min_pct","typical_savings_max_pct",
             "potential_min_eur","potential_max_eur","lines","suppliers","bucket"]
        ],
        use_container_width=True
    )

    # VIEW 2: Supplier drill-down
    st.subheader("2) Suppliers by Category")
    if len(cat):
        chosen_cat = st.selectbox("Choose category to drill down:", cat["category"].tolist())
        sup = (
            df[df["category"]==chosen_cat]
            .groupby("supplier_canon", dropna=False)
            .agg(spend_eur=("_spend_eur","sum"), lines=("po_id","count"))
            .reset_index()
            .assign(spend_eur_k=lambda x: x["spend_eur"]/1_000.0,
                    spend_eur_m=lambda x: x["spend_eur"]/1_000_000.0)
            .sort_values("spend_eur", ascending=False)
        )
        st.dataframe(sup, use_container_width=True)
    else:
        st.info("No categories detected to drill down.")

    # VIEW 3: VAVE ideas
    st.subheader("3) VAVE Ideas by Category (examples)")
    vave_rows = []
    for _, row in cat.iterrows():
        cat_name = str(row["category"])
        ideas = vave_ideas_for_category(cat_name)
        vave_rows.append({"category": cat_name, "example_vave_ideas": " • ".join(ideas[:4])})
    vave_table = pd.DataFrame(vave_rows)
    st.dataframe(vave_table, use_container_width=True)

    # Supplier debug tray (optional)
    st.markdown("#### Debug a supplier (optional)")
    sup_to_debug = st.selectbox(
        "Pick a supplier to inspect line-level calculations:",
        sorted(df["supplier_canon"].dropna().unique().tolist()) if "supplier_canon" in df.columns else []
    )
    if sup_to_debug:
        debug_cols = [
            "po_id","date","category","description","supplier_canon",
            "currency","currency_iso","rate_to_eur",
            "unit_price","unit_price_eur","quantity","quantity_norm",
            "amount","amount_eur","_spend_eur"
        ]
        st.dataframe(
            df[df["supplier_canon"]==sup_to_debug][[c for c in debug_cols if c in df.columns]].head(300),
            use_container_width=True
        )

    # Download pack
    st.markdown("#### Download pack (XLSX)")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="line_level")
        cat.to_excel(writer, index=False, sheet_name="category_overview")
        if len(cat) and 'chosen_cat' in locals():
            sup.to_excel(writer, index=False, sheet_name="suppliers_selected_category")
        vave_table.to_excel(writer, index=False, sheet_name="vave_ideas")
    st.download_button("Download results (EUR).xlsx", data=buffer.getvalue(),
                       file_name="procurement_diagnostics_results_EUR.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload an Excel file to begin. The app will auto-map columns, prefer 'Item family' / 'Item family group' as category, convert to EUR, and produce all outputs.")
