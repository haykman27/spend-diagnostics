import io, re
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans  # kept in case you want to re-enable later

# ================= App chrome =================
st.set_page_config(page_title="Procurement Diagnostics — Simple", layout="wide")
st.title("Procurement Diagnostics — Simple (Auto FX → EUR)")
st.caption("Upload your Excel. The app auto-detects columns & currencies, fetches ECB FX rates, converts to EUR, and shows category savings + supplier drill-downs + VAVE ideas.")

uploaded = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx","xls"])

# ================= Currency helpers =================
BASE = "EUR"

CURRENCY_SYMBOL_MAP = {
    "$":"USD", "US$":"USD", "A$":"AUD", "AU$":"AUD", "C$":"CAD", "CA$":"CAD",
    "€":"EUR", "£":"GBP", "¥":"JPY", "₩":"KRW", "₹":"INR", "₺":"TRY",
    "R$":"BRL", "S$":"SGD", "HK$":"HKD", "₪":"ILS", "₱":"PHP", "₦":"NGN", "₫":"VND"
}
ISO_3 = {
    "EUR","USD","GBP","JPY","CNY","CHF","SEK","NOK","DKK","PLN","HUF","CZK","RON",
    "AUD","NZD","CAD","MXN","BRL","ZAR","AED","SAR","HKD","SGD","INR","TRY","KRW","TWD","THB","PHP","ILS","NGN","VND","RUB"
}

def detect_iso_from_text(text: str):
    if text is None or (isinstance(text, float) and np.isnan(text)): return None
    s = str(text).strip()
    if not s: return None
    m = re.search(r"\b([A-Za-z]{3})\b", s.upper())
    if m and m.group(1) in ISO_3: return m.group(1)
    for sym in sorted(CURRENCY_SYMBOL_MAP.keys(), key=len, reverse=True):
        if sym in s: return CURRENCY_SYMBOL_MAP[sym]
    m2 = re.match(r"^([A-Za-z]{3})\b", s.upper())
    if m2 and m2.group(1) in ISO_3: return m2.group(1)
    m3 = re.search(r"\b([A-Za-z]{3})$", s.upper())
    if m3 and m3.group(1) in ISO_3: return m3.group(1)
    return None

def parse_price_to_float(x):
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
    try: return float(s2)
    except: return np.nan

# ================= Auto-mapper =================
def normalize_headers(cols):
    return [re.sub(r"[\s_\-:/]+", " ", str(c).strip().lower()) for c in cols]

TARGETS = {
    "po_id": ["po", "po id", "po_num", "po number", "purchase order", "order id", "document", "line id", "invoice line id"],
    "date": ["date", "posting date", "order date", "document date", "invoice date"],
    # NOTE: we'll prefer "item family"/"item family group" later regardless of this
    "category": ["category", "commodity", "spend category", "material group", "gl category", "family", "family group", "group"],
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

    # 1) generic fuzzy suggestions
    for field, synonyms in TARGETS.items():
        best = None; best_score = -1
        for syn in synonyms:
            match, score, _ = process.extractOne(syn, norm, scorer=fuzz.token_sort_ratio)
            if score > best_score:
                best, best_score = match, score
        if best is not None:
            suggestions[field] = (back[best], best_score)

    # 2) Explicit preference: item family / item family group as category
    priority_names = ["item family group", "item family"]
    for pr in priority_names:
        for i, n in enumerate(norm):
            if n == pr:
                suggestions["category"] = (back[n], 100)
                break

    # 3) Date heuristic
    if suggestions["date"][1] < 70:
        for c in cols:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.6:
                suggestions["date"] = (c, 80); break
    return suggestions

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

    # supplier canonicalization
    if "supplier" in df2.columns:
        vals = df2["supplier"].fillna("").astype(str).str.strip()
        canon = {}
        for n in vals.unique():
            if not canon: canon[n] = n; continue
            match, score, _ = process.extractOne(n, list(canon.keys()), scorer=fuzz.token_sort_ratio)
            canon[n] = match if score >= 92 else n
        df2["supplier_canon"] = vals.map(canon)
    return df2

# ================= ECB FX =================
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

    df["currency_iso"] = df["currency"].astype(str).apply(lambda x: detect_iso_from_text(x) or BASE)

    if "unit_price" in df.columns:
        is_text = df["unit_price"].apply(lambda x: isinstance(x, str))
        if is_text.any():
            idxs = df[is_text].index
            detected_from_price = df.loc[idxs, "unit_price"].apply(detect_iso_from_text)
            df.loc[idxs, "currency_iso"] = np.where(
                detected_from_price.notna(), detected_from_price, df.loc[idxs, "currency_iso"]
            )
            df.loc[idxs, "unit_price"] = df.loc[idxs, "unit_price"].apply(parse_price_to_float)

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

    miss = df["rate_to_eur"].isna().sum()
    if miss:
        st.warning(f"{miss} row(s) have currencies not in ECB table. Treating them as EUR (rate=1.0).")
        df["rate_to_eur"] = df["rate_to_eur"].fillna(1.0)

    df["unit_price_eur"] = df.get("unit_price", np.nan) * df["rate_to_eur"]
    df["quantity_norm"] = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
    return df

# ================= Savings ranges & VAVE ideas =================
# Heuristic library (customize per client). Ranges are indicative for screening, not guarantees.
SAVINGS_LIBRARY = {
    "stampings": (0.10, 0.15),
    "tubes": (0.08, 0.14),
    "machin": (0.07, 0.14),     # machining/machined parts
    "cast": (0.06, 0.12),       # castings
    "forg": (0.05, 0.10),       # forgings
    "plast": (0.08, 0.18),      # injection molding/plastics
    "fasten": (0.05, 0.10),
    "bear": (0.04, 0.08),       # bearings
    "pcb": (0.06, 0.12), "electro": (0.06, 0.12), "electronics": (0.06,0.12),
    "cable": (0.06, 0.12), "wire": (0.06, 0.12),
    "packag": (0.05, 0.15),
    "logist": (0.08, 0.15), "freight": (0.08, 0.15),
    "mro": (0.05, 0.12),
    "chem": (0.04, 0.10),
    "tool": (0.05, 0.12),       # tooling
    "seal": (0.04, 0.09), "gasket": (0.04, 0.09),
    "rubber": (0.06, 0.12),
    "alum": (0.05, 0.12), "steel": (0.05, 0.12),  # raw material categories
}

VAVE_IDEAS = {
    "stampings": [
        "Standardize **steel grades** across stampings; consolidate to 2–3 specs.",
        "Increase **nesting/yield** via sheet utilization rules with suppliers.",
        "Relax unnecessary **tolerances** / radii; move to common die sets where possible.",
        "Bundle volumes across plants; **progressive die** where feasible."
    ],
    "tubes": [
        "Standardize **diameters/wall thickness** to catalog sizes; avoid odd specs.",
        "Switch to **HF-welded** where seamless is not functionally required.",
        "Increase **cut length tolerances** to use stock lengths and reduce scrap."
    ],
    "machin": [
        "Move to **near-net** (casting/forging) for high chip-to-part ratios.",
        "Reduce **setups** by combining operations; redesign for manufacturability.",
        "Specify **commercial surface finishes** unless safety/fit needs tighter."
    ],
    "plast": [
        "Consolidate **resins & colors**; use masterbatch instead of pre-colored.",
        "Design for **family molds** and shared inserts; increase cavitation.",
        "Use **standard hardware** for fasteners/inserts instead of custom."
    ],
    "fasten": [
        "Rationalize to **ISO/DIN** sizes; eliminate custom lengths/heads where possible.",
        "Vendor-managed inventory (VMI); convert to **kit packaging** for assembly cells."
    ],
    "electronics": [
        "Approve **cross-vendor alternates** for passives; multi-source key ICs.",
        "Increase **panelization** and optimize PCB thickness/finish to need."
    ],
    "packag": [
        "Switch to **standard carton footprints**; increase board utilization.",
        "Right-size **flute/board grade** to product protection needs."
    ],
    "logist": [
        "Optimize **mode mix** (road/rail/sea vs air); implement **milk-runs**.",
        "Increase **load factor** via packaging redesign and cube optimization."
    ],
    "mro": [
        "Create **approved catalog** with tiered pricing; consolidate brands.",
        "Set **min order** and **rebate** structure; restrict spot buys."
    ],
    "raw": [
        "Hedge commodity-linked surcharges; move to **index-linked** pricing.",
        "Standardize **material specs** (e.g., 304L vs 316L only where needed)."
    ]
}

def classify_category_for_savings(cat: str):
    c = (cat or "").lower()
    # try most specific keys first via fuzzy contains
    for k, rng in SAVINGS_LIBRARY.items():
        if k in c: return rng, k
    # map some broad buckets
    if any(w in c for w in ["steel","alum","aluminum","sheet"]): return SAVINGS_LIBRARY["steel"], "steel"
    if any(w in c for w in ["pcb","electro"]): return SAVINGS_LIBRARY["electronics"], "electronics"
    return (0.05, 0.10), "generic"  # safe default

def vave_ideas_for_category(cat: str):
    c = (cat or "").lower()
    for key, ideas in VAVE_IDEAS.items():
        if key in c: return ideas
    if any(w in c for w in ["steel","alum","aluminum","raw"]): return VAVE_IDEAS["raw"]
    return [
        "Standardize **specs** and tolerances to functional need.",
        "Consolidate variants; use **catalog parts** where possible.",
        "Bundle volumes to a smaller supplier set and negotiate."
    ]

# ================= Analytics helpers =================
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
    raw = pd.read_excel(uploaded)
    raw.columns = [str(c) for c in raw.columns]

    # --- detect columns ---
    suggestions = detect_columns(raw)
    mapping = {k: (suggestions[k][0] if k in suggestions else None) for k in TARGETS.keys()}
    # ensure category uses Item family/group if present (already boosted, but enforce again)
    for pref in ["item family group", "item family"]:
        for c in raw.columns:
            if normalize_headers([c])[0] == pref:
                mapping["category"] = c

    # minimal validation
    missing_required = [f for f in REQUIRED if not mapping.get(f)]
    if missing_required:
        st.error("Missing required field(s): " + ", ".join(missing_required) +
                 ". Please include columns like supplier, quantity, currency, uom and category/item family.")
        st.stop()

    # --- apply mapping; compute unit price if needed ---
    df = apply_mapping(raw, mapping)
    if "unit_price" not in df.columns or df["unit_price"].isna().all():
        st.error("Could not derive unit price. Ensure you have either a unit price column, or amount + quantity.")
        st.stop()

    # --- FX conversion to EUR ---
    with st.spinner("Fetching ECB FX rates and converting to EUR..."):
        fx_wide = pd.read_csv("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv")
    fx_wide.rename(columns={"Date":"date"}, inplace=True)
    fx_wide["date"] = pd.to_datetime(fx_wide["date"], errors="coerce")
    fx_long = fx_wide.melt(id_vars=["date"], var_name="currency", value_name="eur_to_cur")
    fx_long["currency"] = fx_long["currency"].str.upper().str.strip()
    fx_long["eur_to_cur"] = pd.to_numeric(fx_long["eur_to_cur"], errors="coerce")
    fx_long["rate_to_eur"] = 1.0 / fx_long["eur_to_cur"]
    eur_rows = fx_long[["date"]].drop_duplicates().assign(currency="EUR", rate_to_eur=1.0)
    fx = pd.concat([fx_long[["date","currency","rate_to_eur"]], eur_rows], ignore_index=True)
    fx = fx.dropna(subset=["date","currency","rate_to_eur"]).sort_values(["currency","date"])

    # merge FX
    if "currency" not in df.columns:
        st.error("No currency column detected. Include a currency column or symbols in price text.")
        st.stop()
    df["currency_iso"] = df["currency"].astype(str).apply(lambda x: detect_iso_from_text(x) or BASE)
    if "unit_price" in df.columns:
        is_text = df["unit_price"].apply(lambda x: isinstance(x, str))
        if is_text.any():
            idxs = df[is_text].index
            detected_from_price = df.loc[idxs, "unit_price"].apply(detect_iso_from_text)
            df.loc[idxs, "currency_iso"] = np.where(
                detected_from_price.notna(), detected_from_price, df.loc[idxs, "currency_iso"]
            )
            df.loc[idxs, "unit_price"] = df.loc[idxs, "unit_price"].apply(parse_price_to_float)
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
    miss = df["rate_to_eur"].isna().sum()
    if miss:
        st.warning(f"{miss} row(s) have currencies not in ECB table. Treating them as EUR (rate=1.0).")
        df["rate_to_eur"] = df["rate_to_eur"].fillna(1.0)
    df["unit_price_eur"] = df.get("unit_price", np.nan) * df["rate_to_eur"]
    df["quantity_norm"] = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)

    # --- compute analytics baseline + flags (kept for future) ---
    df = add_baseline_and_flags(df, iqr_multiplier=1.5, baseline="P50")

    # ---------- VIEW 1: Category overview with savings ranges ----------
    df["_spend_eur"] = (df["unit_price_eur"] * df["quantity_norm"]).fillna(0.0)
    cat_base = df.groupby("category", dropna=False).agg(
        spend_eur=("_spend_eur", "sum"),
        lines=("po_id","count"),
        suppliers=("supplier_canon", pd.Series.nunique)
    ).reset_index()

    # attach typical savings ranges + € potential min/max
    ranges, buckets = [], []
    for cat in cat_base["category"]:
        rng, bucket = classify_category_for_savings(str(cat))
        ranges.append(rng); buckets.append(bucket)
    cat_base["typical_savings_min_pct"] = [r[0] for r in ranges]
    cat_base["typical_savings_max_pct"] = [r[1] for r in ranges]
    cat_base["potential_min_eur"] = cat_base["spend_eur"] * cat_base["typical_savings_min_pct"]
    cat_base["potential_max_eur"] = cat_base["spend_eur"] * cat_base["typical_savings_max_pct"]
    cat_base["bucket"] = buckets

    st.subheader("1) Categories — Spend and Typical Savings")
    st.dataframe(
        cat_base.sort_values("spend_eur", ascending=False)[
            ["category","spend_eur","typical_savings_min_pct","typical_savings_max_pct","potential_min_eur","potential_max_eur","lines","suppliers","bucket"]
        ],
        use_container_width=True
    )

    # ---------- VIEW 2: Supplier drill-down ----------
    st.subheader("2) Drill down — Suppliers by Category")
    if len(cat_base):
        chosen_cat = st.selectbox("Choose a category to drill down:", cat_base["category"].tolist())
        sup_view = (
            df[df["category"] == chosen_cat]
            .groupby("supplier_canon", dropna=False)
            .agg(spend_eur=("_spend_eur","sum"), lines=("po_id","count"))
            .reset_index()
            .sort_values("spend_eur", ascending=False)
        )
        st.dataframe(sup_view, use_container_width=True)
    else:
        st.info("No categories detected to drill down.")

    # ---------- VIEW 3: Practical VAVE ideas ----------
    st.subheader("3) VAVE Ideas by Category (examples)")
    vave_rows = []
    for _, row in cat_base.iterrows():
        cat = str(row["category"])
        ideas = vave_ideas_for_category(cat)
        vave_rows.append({
            "category": cat,
            "example_vave_ideas": " • ".join(ideas[:4])
        })
    vave_table = pd.DataFrame(vave_rows)
    st.dataframe(vave_table, use_container_width=True)

    # ---------- DOWNLOAD PACK ----------
    st.markdown("#### Download pack (XLSX)")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.drop(columns=["_spend_eur"], errors="ignore").to_excel(writer, index=False, sheet_name="line_level")
        cat_base.to_excel(writer, index=False, sheet_name="category_overview")
        if len(cat_base) and 'chosen_cat' in locals():
            sup_view.to_excel(writer, index=False, sheet_name="suppliers_selected_category")
        vave_table.to_excel(writer, index=False, sheet_name="vave_ideas")
    st.download_button("Download results (EUR).xlsx", data=buffer.getvalue(),
                       file_name="procurement_diagnostics_results_EUR.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload an Excel file to begin. The app will auto-map columns, prefer 'Item family' / 'Item family group' as the category, convert to EUR, and produce the new outputs.")
