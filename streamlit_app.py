import io
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import norm

try:
    import yfinance as yf
except Exception:
    yf = None


# ============================================================
# Configuracion general
# ============================================================

st.set_page_config(
    page_title="Portafolio óptimo - Minimum Variance Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
SAMPLE_PRICES_PATH = os.path.join(DATA_DIR, "precios_bloomberg_61m.csv")
SAMPLE_ANR_PATH = os.path.join(DATA_DIR, "anr_targets_20260515.csv")

DEFAULT_ASSETS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "AMZN", "PANW", "LLY", "JNJ", "UNH",
    "JPM", "BRK/B", "V", "XOM", "NEE", "ETN", "PG", "WMT", "KO", "LMT",
]
MARKET_TICKER = "SPX"
YAHOO_MAP = {
    "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "AVGO": "AVGO", "GOOGL": "GOOGL",
    "AMZN": "AMZN", "PANW": "PANW", "LLY": "LLY", "JNJ": "JNJ", "UNH": "UNH",
    "JPM": "JPM", "BRK/B": "BRK-B", "V": "V", "XOM": "XOM", "NEE": "NEE", "ETN": "ETN",
    "PG": "PG", "WMT": "WMT", "KO": "KO", "LMT": "LMT", "SPX": "^GSPC",
}
REVERSE_YAHOO_MAP = {v: k for k, v in YAHOO_MAP.items()}

ASSET_INFO = {
    "AAPL": ("Apple Inc.", "Technology / Hardware & Services"),
    "MSFT": ("Microsoft Corp.", "Technology / Software & Cloud"),
    "NVDA": ("NVIDIA Corp.", "Technology / Semiconductors & AI"),
    "AVGO": ("Broadcom Inc.", "Technology / Semiconductors & Infrastructure"),
    "GOOGL": ("Alphabet Inc.", "Communication Services / Digital Ads & Cloud"),
    "AMZN": ("Amazon.com Inc.", "Consumer Discretionary / Cloud & E-commerce"),
    "PANW": ("Palo Alto Networks Inc.", "Technology / Cybersecurity"),
    "LLY": ("Eli Lilly & Co.", "Health Care / Pharma"),
    "JNJ": ("Johnson & Johnson", "Health Care / Defensive Pharma"),
    "UNH": ("UnitedHealth Group Inc.", "Health Care / Managed Care"),
    "JPM": ("JPMorgan Chase & Co.", "Financials / Banking"),
    "BRK/B": ("Berkshire Hathaway Inc. Class B", "Financials / Diversified Holding"),
    "V": ("Visa Inc.", "Financials / Payments"),
    "XOM": ("Exxon Mobil Corp.", "Energy"),
    "NEE": ("NextEra Energy Inc.", "Utilities / Renewable Power"),
    "ETN": ("Eaton Corp. PLC", "Industrials / Electrification"),
    "PG": ("Procter & Gamble Co.", "Consumer Staples"),
    "WMT": ("Walmart Inc.", "Consumer Staples / Retail"),
    "KO": ("Coca-Cola Co.", "Consumer Staples / Beverages"),
    "LMT": ("Lockheed Martin Corp.", "Industrials / Defense"),
}

VAR_TARGETS = [0.04, 0.02, 0.01, 0.00, -0.01]


# ============================================================
# Utilidades de formato
# ============================================================

def fmt_pct(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.{decimals}f}%"


def fmt_num(x: float, decimals: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{decimals}f}"


def clean_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    aliases = {
        "BRK-B": "BRK/B",
        "BRKB": "BRK/B",
        "BRK B": "BRK/B",
        "^GSPC": "SPX",
        "GSPC": "SPX",
        "S&P 500": "SPX",
        "SP500": "SPX",
        "S&P500": "SPX",
        "SPX INDEX": "SPX",
        "APPLE": "AAPL",
        "MICROSOFT": "MSFT",
        "MICROSFOT": "MSFT",
        "NVIDIA": "NVDA",
        "BROADCOM": "AVGO",
        "ALPHABET": "GOOGL",
        "AMAZON": "AMZN",
        "PALO ALTO NETWORKS": "PANW",
        "ELI LILLY": "LLY",
        "JOHNSON & JOHNSON": "JNJ",
        "UNITED HEALTH": "UNH",
        "UNITEDHEALTH": "UNH",
        "JPMORGAN CHASE": "JPM",
        "BERKSHIRE HATHAWAY": "BRK/B",
        "VISA": "V",
        "EXXON MOBIL": "XOM",
        "NEXTERA ENERGY": "NEE",
        "EATON": "ETN",
        "PROCTER & GAMBLE": "PG",
        "WALMART": "WMT",
        "COCA-COLA": "KO",
        "COCA COLA": "KO",
        "LOCKHEED MARTIN": "LMT",
    }
    return aliases.get(t, t)


def percent_style(df: pd.DataFrame, columns: Iterable[str], decimals: int = 2):
    fmt = {c: f"{{:.{decimals}%}}" for c in columns if c in df.columns}
    return df.style.format(fmt)


# ============================================================
# Carga de datos
# ============================================================

@st.cache_data(show_spinner=False)
def load_sample_prices() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_PRICES_PATH)
    df = df.rename(columns={"Fecha": "Date", "date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = [clean_ticker(c) for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


@st.cache_data(show_spinner=False)
def load_sample_anr() -> pd.DataFrame:
    if not os.path.exists(SAMPLE_ANR_PATH):
        return pd.DataFrame()
    df = pd.read_csv(SAMPLE_ANR_PATH)
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].map(clean_ticker)
    return df


@st.cache_data(show_spinner=True)
def fetch_yahoo_prices(asset_tickers: Tuple[str, ...], start: str, end: str, market_ticker: str = MARKET_TICKER) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance no está instalado. Use el archivo incluido o suba un Excel/CSV.")

    tickers = list(asset_tickers) + [market_ticker]
    yahoo_symbols = [YAHOO_MAP.get(t, t.replace("/", "-")) for t in tickers]
    end_dt = pd.to_datetime(end) + pd.DateOffset(days=5)

    raw = yf.download(
        yahoo_symbols,
        start=start,
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1mo",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("Yahoo Finance no retornó datos. Use la opción de archivo incluido o suba un Excel/CSV.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("La descarga no contiene campo Close.")
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [yahoo_symbols[0]]

    close = close.rename(columns={sym: REVERSE_YAHOO_MAP.get(sym, sym) for sym in close.columns})
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close = close[[c for c in tickers if c in close.columns]]
    close = close.dropna(axis=1, how="all").dropna(how="all")
    return close


def _convert_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")


def _try_simple_price_format(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
    if df.empty or df.shape[1] < 2:
        return None

    date_col = None
    for col in df.columns:
        name = str(col).strip().lower()
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        if name in ["date", "fecha", "dates"] or parsed.notna().mean() > 0.70:
            date_col = col
            break
    if date_col is None:
        return None

    out = pd.DataFrame(index=pd.to_datetime(df[date_col], errors="coerce", dayfirst=True))
    for col in df.columns:
        if col == date_col:
            continue
        ticker = clean_ticker(col)
        if ticker in ["NAN", "NONE", "UNNAMED: 0"]:
            continue
        values = _convert_numeric_series(df[col])
        if values.notna().sum() >= 3:
            out[ticker] = values.values

    out.index.name = "Date"
    out = out[~out.index.isna()].sort_index()
    out = out.loc[:, ~out.columns.duplicated()].dropna(axis=1, how="all")
    return out if out.shape[1] >= 2 else None


def _try_pair_format(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    # Formato Bloomberg exportado como pares: columna fecha + columna precio para cada activo.
    if raw.shape[1] < 4 or raw.shape[0] < 10:
        return None
    data = {}
    base_dates = None
    for c in range(0, raw.shape[1] - 1, 2):
        raw_name = raw.iloc[0, c]
        if pd.isna(raw_name):
            continue
        ticker = clean_ticker(str(raw_name))
        dates = pd.to_datetime(raw.iloc[1:, c], errors="coerce", dayfirst=True)
        values = _convert_numeric_series(raw.iloc[1:, c + 1])
        valid = dates.notna() & values.notna()
        if valid.sum() < 3:
            continue
        tmp = pd.Series(values[valid].values, index=dates[valid], name=ticker)
        data[ticker] = tmp
        if base_dates is None:
            base_dates = tmp.index
    if not data:
        return None
    out = pd.concat(data.values(), axis=1).sort_index()
    out.index.name = "Date"
    out = out.loc[:, ~out.columns.duplicated()].dropna(axis=1, how="all")
    return out if out.shape[1] >= 2 else None


def parse_uploaded_prices(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep)
                parsed = _try_simple_price_format(df)
                if parsed is not None:
                    return parsed
            except Exception:
                continue
        raise ValueError("No se pudo interpretar el CSV. Use una columna Date/Fecha y columnas de precios por activo.")

    # Excel: intenta hoja Precios, luego primera hoja, con diferentes filas de encabezado.
    xls = pd.ExcelFile(io.BytesIO(content))
    candidate_sheets = []
    if "Precios" in xls.sheet_names:
        candidate_sheets.append("Precios")
    candidate_sheets += [s for s in xls.sheet_names if s not in candidate_sheets]

    for sheet in candidate_sheets:
        raw = pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=None)
        pair = _try_pair_format(raw)
        if pair is not None:
            return pair
        for header in [0, 1, 2, 3]:
            try:
                df = pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=header)
                parsed = _try_simple_price_format(df)
                if parsed is not None:
                    return parsed
            except Exception:
                continue

    raise ValueError("No se pudo interpretar el Excel. Use una hoja con Date/Fecha en la primera columna y precios en las columnas siguientes.")


# ============================================================
# Calculos financieros
# ============================================================

@dataclass
class PortfolioResult:
    name: str
    expected_model: str
    weights: pd.Series
    ret_ann: float
    risk_ann: float
    sharpe: float
    var5: float
    risk_unit: float


def align_price_data(prices: pd.DataFrame, assets: List[str], market: str) -> pd.DataFrame:
    prices = prices.copy()
    prices.columns = [clean_ticker(c) for c in prices.columns]
    keep = [c for c in assets + [market] if c in prices.columns]
    if market not in keep:
        raise ValueError(f"No se encontró el benchmark de mercado {market}. Incluya SPX/^GSPC o use Yahoo Finance.")
    missing_assets = [a for a in assets if a not in prices.columns]
    if missing_assets:
        st.warning("Activos faltantes en la fuente de datos: " + ", ".join(missing_assets))
    prices = prices[keep].dropna(how="any")
    if len(prices) < 13:
        raise ValueError("La serie final tiene muy pocos datos. Se requieren al menos 13 precios mensuales.")
    return prices


def calculate_inputs(prices: pd.DataFrame, assets: List[str], market: str, rf_ea: float, erp_usa: float, anr_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    asset_prices = prices[assets]
    log_returns = np.log(asset_prices / asset_prices.shift(1)).dropna()
    simple_returns = asset_prices.pct_change().dropna()
    market_simple = prices[market].pct_change().dropna()

    common = simple_returns.index.intersection(market_simple.index)
    simple_returns = simple_returns.loc[common]
    market_simple = market_simple.loc[common]

    hist_monthly = log_returns.mean()
    hist_ann = hist_monthly * 12
    sigma_monthly = log_returns.std(ddof=0)
    sigma_ann = sigma_monthly * math.sqrt(12)
    corr = log_returns.corr()
    cov_monthly = log_returns.cov(ddof=0)
    cov_ann = cov_monthly * 12

    raw_betas = {}
    market_var = np.var(market_simple.values, ddof=0)
    for asset in assets:
        x = simple_returns[asset].dropna()
        common_asset = x.index.intersection(market_simple.index)
        if len(common_asset) < 3 or market_var == 0:
            raw_betas[asset] = np.nan
        else:
            cov = np.cov(x.loc[common_asset].values, market_simple.loc[common_asset].values, ddof=0)[0, 1]
            raw_betas[asset] = cov / market_var
    raw_beta = pd.Series(raw_betas)
    adjusted_beta = (2 / 3) * raw_beta + (1 / 3)
    capm_ann = rf_ea + adjusted_beta * erp_usa
    capm_monthly = capm_ann / 12

    anr_ann = pd.Series(index=assets, dtype=float)
    buy_pct = pd.Series(index=assets, dtype=float)
    target_price = pd.Series(index=assets, dtype=float)
    if anr_df is not None and not anr_df.empty and "Ticker" in anr_df.columns:
        temp = anr_df.copy()
        temp["Ticker"] = temp["Ticker"].map(clean_ticker)
        temp = temp.set_index("Ticker")
        for asset in assets:
            if asset in temp.index:
                if "ANR_Log_Return_Annual" in temp.columns:
                    anr_ann.loc[asset] = pd.to_numeric(temp.loc[asset, "ANR_Log_Return_Annual"], errors="coerce")
                elif {"TargetPrice12M", "LastPriceANR"}.issubset(temp.columns):
                    tp = pd.to_numeric(temp.loc[asset, "TargetPrice12M"], errors="coerce")
                    lp = pd.to_numeric(temp.loc[asset, "LastPriceANR"], errors="coerce")
                    if pd.notna(tp) and pd.notna(lp) and lp > 0:
                        anr_ann.loc[asset] = math.log(tp / lp)
                if "BuyPctANR" in temp.columns:
                    buy_pct.loc[asset] = pd.to_numeric(temp.loc[asset, "BuyPctANR"], errors="coerce")
                if "TargetPrice12M" in temp.columns:
                    target_price.loc[asset] = pd.to_numeric(temp.loc[asset, "TargetPrice12M"], errors="coerce")

    stats = pd.DataFrame({
        "Company": [ASSET_INFO.get(a, (a, ""))[0] for a in assets],
        "Sector": [ASSET_INFO.get(a, ("", ""))[1] for a in assets],
        "E[r] Hist Mensual": hist_monthly,
        "E[r] Hist Anual": hist_ann,
        "σ Mensual": sigma_monthly,
        "σ Anual": sigma_ann,
        "Risk Unit Mensual": sigma_monthly / hist_monthly.replace(0, np.nan),
        "Risk Unit Anual": sigma_ann / hist_ann.replace(0, np.nan),
        "Raw Beta": raw_beta,
        "Adjusted Beta": adjusted_beta,
        "E[r] CAPM Anual": capm_ann,
        "E[r] CAPM Mensual": capm_monthly,
        "12M Tgt Px": target_price,
        "E[r] ANR Anual": anr_ann,
        "E[r] ANR Mensual": anr_ann / 12,
        "Buy % ANR": buy_pct,
    })
    stats.index.name = "Ticker"

    return {
        "prices": prices,
        "asset_prices": asset_prices,
        "log_returns": log_returns,
        "simple_returns": simple_returns,
        "hist_ann": hist_ann,
        "capm_ann": capm_ann,
        "anr_ann": anr_ann,
        "corr": corr,
        "cov_ann": cov_ann,
        "stats": stats,
    }


def portfolio_metrics(weights: pd.Series, exp_returns_ann: pd.Series, cov_ann: pd.DataFrame, rf_ea: float, z: float, name: str, model: str) -> PortfolioResult:
    weights = weights.astype(float).reindex(exp_returns_ann.index).fillna(0.0)
    ret = float(weights.values @ exp_returns_ann.values)
    variance = float(weights.values @ cov_ann.loc[weights.index, weights.index].values @ weights.values)
    risk = math.sqrt(max(variance, 0))
    sharpe = (ret - rf_ea) / risk if risk > 0 else np.nan
    var5 = ret - z * risk
    risk_unit = risk / ret if abs(ret) > 1e-12 else np.nan
    return PortfolioResult(name, model, weights, ret, risk, sharpe, var5, risk_unit)


def optimize_min_variance(cov_ann: pd.DataFrame) -> pd.Series:
    assets = list(cov_ann.columns)
    n = len(assets)
    x0 = np.repeat(1 / n, n)
    cov = cov_ann.values
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = [(0, 1)] * n
    res = minimize(lambda w: float(w @ cov @ w), x0=x0, bounds=bounds, constraints=cons, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        st.warning(f"Solver mínima varianza: {res.message}")
    w = np.clip(res.x if res.success else x0, 0, 1)
    w = w / w.sum()
    return pd.Series(w, index=assets)


def optimize_max_sharpe(exp_returns_ann: pd.Series, cov_ann: pd.DataFrame, rf_ea: float) -> pd.Series:
    assets = list(exp_returns_ann.index)
    n = len(assets)
    x0 = np.repeat(1 / n, n)
    cov = cov_ann.loc[assets, assets].values
    mu = exp_returns_ann.values
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = [(0, 1)] * n

    def objective(w):
        risk = math.sqrt(max(float(w @ cov @ w), 0))
        if risk <= 1e-12:
            return 1e6
        return -float((w @ mu - rf_ea) / risk)

    res = minimize(objective, x0=x0, bounds=bounds, constraints=cons, method="SLSQP", options={"maxiter": 1500, "ftol": 1e-12})
    if not res.success:
        st.warning(f"Solver máxima Sharpe: {res.message}")
    w = np.clip(res.x if res.success else x0, 0, 1)
    w = w / w.sum()
    return pd.Series(w, index=assets)


def efficient_frontier(exp_returns_ann: pd.Series, cov_ann: pd.DataFrame, n_points: int = 60) -> pd.DataFrame:
    assets = list(exp_returns_ann.index)
    n = len(assets)
    x0 = np.repeat(1 / n, n)
    cov = cov_ann.loc[assets, assets].values
    mu = exp_returns_ann.values
    bounds = [(0, 1)] * n
    min_mu, max_mu = float(np.nanmin(mu)), float(np.nanmax(mu))
    targets = np.linspace(min_mu, max_mu, n_points)
    rows = []

    for target in targets:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: float(w @ mu) - target},
        )
        res = minimize(lambda w: float(w @ cov @ w), x0=x0, bounds=bounds, constraints=cons, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-10})
        if res.success:
            w = np.clip(res.x, 0, 1)
            if w.sum() > 0:
                w = w / w.sum()
                risk = math.sqrt(max(float(w @ cov @ w), 0))
                rows.append({"Return": float(w @ mu), "Risk": risk})
                x0 = w
    return pd.DataFrame(rows).drop_duplicates()


def calculate_all_portfolios(calc: Dict[str, object], rf_ea: float, z: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series], pd.DataFrame, pd.DataFrame]:
    cov_ann: pd.DataFrame = calc["cov_ann"]
    assets = list(cov_ann.columns)
    hist_ann: pd.Series = calc["hist_ann"].reindex(assets)
    capm_ann: pd.Series = calc["capm_ann"].reindex(assets)
    anr_ann: pd.Series = calc["anr_ann"].reindex(assets)

    equal_w = pd.Series(np.repeat(1 / len(assets), len(assets)), index=assets)
    min_var_w = optimize_min_variance(cov_ann)
    max_sharpe_hist_w = optimize_max_sharpe(hist_ann, cov_ann, rf_ea)
    max_sharpe_capm_w = optimize_max_sharpe(capm_ann, cov_ann, rf_ea)
    weights = {
        "Equal Weight": equal_w,
        "Mínimo Riesgo": min_var_w,
        "Máx Sharpe Histórico": max_sharpe_hist_w,
        "Máx Sharpe CAPM": max_sharpe_capm_w,
    }

    if anr_ann.notna().sum() == len(assets):
        max_sharpe_anr_w = optimize_max_sharpe(anr_ann, cov_ann, rf_ea)
        weights["Máx Sharpe ANR"] = max_sharpe_anr_w
    else:
        max_sharpe_anr_w = None

    results = []
    for w_name, w in [("Equal Weight", equal_w), ("Mínimo Riesgo", min_var_w)]:
        results.append(portfolio_metrics(w, hist_ann, cov_ann, rf_ea, z, w_name, "Histórico"))
        results.append(portfolio_metrics(w, capm_ann, cov_ann, rf_ea, z, w_name, "CAPM"))
        if anr_ann.notna().sum() == len(assets):
            results.append(portfolio_metrics(w, anr_ann, cov_ann, rf_ea, z, w_name, "ANR"))
    results.append(portfolio_metrics(max_sharpe_hist_w, hist_ann, cov_ann, rf_ea, z, "Máx Sharpe Histórico", "Histórico"))
    results.append(portfolio_metrics(max_sharpe_capm_w, capm_ann, cov_ann, rf_ea, z, "Máx Sharpe CAPM", "CAPM"))
    if max_sharpe_anr_w is not None:
        results.append(portfolio_metrics(max_sharpe_anr_w, anr_ann, cov_ann, rf_ea, z, "Máx Sharpe ANR", "ANR"))

    summary = pd.DataFrame([
        {
            "Portafolio": r.name,
            "E[r] usado": r.expected_model,
            "E[r] Anual": r.ret_ann,
            "σ Anual": r.risk_ann,
            "Sharpe": r.sharpe,
            "VaR5%": r.var5,
            "Risk Unit Anual": r.risk_unit,
        }
        for r in results
    ])

    weights_df = pd.DataFrame(weights).fillna(0)

    frontier = efficient_frontier(capm_ann, cov_ann, n_points=70)
    capm_tangent = portfolio_metrics(max_sharpe_capm_w, capm_ann, cov_ann, rf_ea, z, "Máx Sharpe CAPM", "CAPM")
    var_constraints = calculate_var_combinations(max_sharpe_capm_w, capm_tangent.ret_ann, capm_tangent.risk_ann, rf_ea, z)

    return summary, weights_df, weights, frontier, var_constraints


def calculate_var_combinations(tangent_weights: pd.Series, mu_risky: float, sigma_risky: float, rf_ea: float, z: float) -> pd.DataFrame:
    denom = z * sigma_risky - (mu_risky - rf_ea)
    rows = []
    for target in VAR_TARGETS:
        if denom <= 0:
            y = 1.0 if (mu_risky - z * sigma_risky) >= target else 0.0
            feasible = (rf_ea if y == 0 else mu_risky - z * sigma_risky) >= target
        else:
            y = max(0.0, min(1.0, (rf_ea - target) / denom))
            feasible = True
        ret_total = rf_ea + y * (mu_risky - rf_ea)
        risk_total = y * sigma_risky
        var5_total = ret_total - z * risk_total
        row = {
            "VaR5% mínimo requerido": target,
            "Peso portafolio riesgoso": y,
            "Peso activo libre de riesgo": 1 - y,
            "E[r] total": ret_total,
            "σ total": risk_total,
            "VaR5% total": var5_total,
            "Sharpe": (ret_total - rf_ea) / risk_total if risk_total > 0 else np.nan,
            "Factible": feasible,
        }
        for asset, w in tangent_weights.items():
            row[asset] = y * w
        rows.append(row)
    return pd.DataFrame(rows)


def build_download_excel(calc: Dict[str, object], summary: pd.DataFrame, weights_df: pd.DataFrame, frontier: pd.DataFrame, var_constraints: pd.DataFrame, rf_ea: float, erp_usa: float, z: float) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        inputs = pd.DataFrame({
            "Parámetro": ["Rf EA", "ERP USA", "Z 95%", "Metodología"],
            "Valor": [rf_ea, erp_usa, z, "Long-only, 20 activos, rendimientos logarítmicos mensuales"],
        })
        inputs.to_excel(writer, sheet_name="Inputs", index=False)
        calc["prices"].to_excel(writer, sheet_name="Precios")
        calc["log_returns"].to_excel(writer, sheet_name="Rendimientos")
        calc["stats"].to_excel(writer, sheet_name="Estadisticas")
        calc["corr"].to_excel(writer, sheet_name="Correlacion")
        calc["cov_ann"].to_excel(writer, sheet_name="Covarianza Anual")
        weights_df.to_excel(writer, sheet_name="Pesos Optimos")
        summary.to_excel(writer, sheet_name="Portafolios", index=False)
        frontier.to_excel(writer, sheet_name="Frontera CAPM", index=False)
        var_constraints.to_excel(writer, sheet_name="VaR Restricciones", index=False)

        workbook = writer.book
        pct_fmt = workbook.add_format({"num_format": "0.00%"})
        num_fmt = workbook.add_format({"num_format": "0.0000"})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#1F4E79", "font_color": "white"})
        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes(1, 1)
            worksheet.set_row(0, None, header_fmt)
            worksheet.set_column(0, 0, 18)
            worksheet.set_column(1, 30, 14, num_fmt)

        # Formatos puntuales para que el Excel exportado no convierta precios o textos en porcentajes.
        if "Inputs" in writer.sheets:
            writer.sheets["Inputs"].set_column(1, 1, 14, pct_fmt)
        if "Estadisticas" in writer.sheets:
            ws = writer.sheets["Estadisticas"]
            ws.set_column(1, 2, 32)   # Company / Sector
            ws.set_column(3, 6, 14, pct_fmt)   # retornos y riesgos
            ws.set_column(7, 10, 14, num_fmt)  # risk unit y betas
            ws.set_column(11, 12, 14, pct_fmt) # CAPM anual/mensual
            ws.set_column(13, 13, 14, num_fmt) # target price
            ws.set_column(14, 16, 14, pct_fmt) # ANR retornos
        if "Portafolios" in writer.sheets:
            ws = writer.sheets["Portafolios"]
            ws.set_column(2, 3, 14, pct_fmt)
            ws.set_column(4, 6, 14, pct_fmt)
            ws.set_column(7, 7, 14, num_fmt)
        if "Frontera CAPM" in writer.sheets:
            writer.sheets["Frontera CAPM"].set_column(0, 1, 14, pct_fmt)
        if "VaR Restricciones" in writer.sheets:
            writer.sheets["VaR Restricciones"].set_column(0, 6, 14, pct_fmt)
    return output.getvalue()


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .metric-card {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 16px; background: rgba(128,128,128,.06);}
    .small-muted {font-size: 0.86rem; color: #777;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Portafolio óptimo: Minimum-Variance Portfolio")
st.caption("Aplicación interactiva para replicar el trabajo final: 20 activos long-only, CAPM, matriz de correlación, mínima varianza, máxima Sharpe, frontera eficiente y VaR5%.")

with st.sidebar:
    st.header("Configuración")
    data_source = st.radio(
        "Fuente de datos",
        ["Datos incluidos 61 meses", "Yahoo Finance API", "Subir Excel/CSV"],
        index=0,
        help="Use los datos incluidos para una demo estable. Yahoo permite refrescar datos, pero depende de la disponibilidad de la API.",
    )

    selected_assets = st.multiselect(
        "Activos del portafolio",
        DEFAULT_ASSETS,
        default=DEFAULT_ASSETS,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Fecha inicial", value=pd.to_datetime("2021-04-30"))
    with col_b:
        end_date = st.date_input("Fecha final", value=pd.to_datetime("2026-04-30"))

    st.divider()
    st.subheader("Supuestos")
    rf_nominal = st.number_input("Treasury 10Y nominal", min_value=0.0, max_value=0.25, value=0.0459, step=0.001, format="%.4f")
    rf_ea = (1 + rf_nominal / 2) ** 2 - 1
    erp_usa = st.number_input("ERP USA Damodaran", min_value=0.0, max_value=0.20, value=0.046, step=0.001, format="%.4f")
    z_value = norm.ppf(0.95)
    st.write(f"Rf efectiva anual: **{fmt_pct(rf_ea)}**")
    st.write(f"Z 95%: **{z_value:.4f}**")

    uploaded_prices = None
    if data_source == "Subir Excel/CSV":
        uploaded_prices = st.file_uploader("Archivo de precios", type=["xlsx", "xls", "csv"])
        st.caption("Formato recomendado: primera columna Date/Fecha y una columna por activo. También acepta el formato de pares Fecha-Precio exportado de Bloomberg.")

    use_anr = st.toggle("Usar ANR incluido como tercera estimación", value=True)

if len(selected_assets) < 2:
    st.error("Seleccione al menos dos activos.")
    st.stop()

# Carga de precios
try:
    if data_source == "Datos incluidos 61 meses":
        prices_raw = load_sample_prices()
        data_label = "datos incluidos / Bloomberg PX_LAST mensual 30/04/2021-30/04/2026"
    elif data_source == "Yahoo Finance API":
        prices_raw = fetch_yahoo_prices(tuple(selected_assets), str(start_date), str(end_date), MARKET_TICKER)
        data_label = "Yahoo Finance API / Close mensual"
    else:
        if uploaded_prices is None:
            st.info("Suba un archivo Excel/CSV para continuar.")
            st.stop()
        prices_raw = parse_uploaded_prices(uploaded_prices)
        data_label = f"archivo cargado: {uploaded_prices.name}"

    prices = align_price_data(prices_raw, selected_assets, MARKET_TICKER)
    assets = [a for a in selected_assets if a in prices.columns]
    anr_df = load_sample_anr() if use_anr else pd.DataFrame()
    calc = calculate_inputs(prices, assets, MARKET_TICKER, rf_ea, erp_usa, anr_df)
    summary, weights_df, weights_dict, frontier, var_constraints = calculate_all_portfolios(calc, rf_ea, z_value)
except Exception as exc:
    st.error(f"No se pudo ejecutar el modelo: {exc}")
    st.stop()

# ============================================================
# Vista ejecutiva
# ============================================================

st.info(f"Fuente activa: {data_label}. Observaciones de precios: {len(prices)}. Rendimientos mensuales: {len(calc['log_returns'])}.")

summary_capm = summary[(summary["Portafolio"] == "Máx Sharpe CAPM") & (summary["E[r] usado"] == "CAPM")].iloc[0]
summary_min = summary[(summary["Portafolio"] == "Mínimo Riesgo") & (summary["E[r] usado"] == "CAPM")].iloc[0]
summary_eq = summary[(summary["Portafolio"] == "Equal Weight") & (summary["E[r] usado"] == "CAPM")].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Máx Sharpe CAPM", fmt_num(summary_capm["Sharpe"], 4), f"VaR5% {fmt_pct(summary_capm['VaR5%'])}")
m2.metric("E[r] anual óptimo", fmt_pct(summary_capm["E[r] Anual"]), f"σ {fmt_pct(summary_capm['σ Anual'])}")
m3.metric("Mínimo riesgo", fmt_pct(summary_min["σ Anual"]), f"E[r] {fmt_pct(summary_min['E[r] Anual'])}")
m4.metric("Equal Weight", fmt_pct(summary_eq["E[r] Anual"]), f"Sharpe {summary_eq['Sharpe']:.4f}")

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Datos y estadística", "Portafolios", "Frontera eficiente", "Pesos", "VaR restricciones", "Descargas"
])

with tab1:
    st.subheader("Precios acumulados base 100")
    base100 = calc["asset_prices"].divide(calc["asset_prices"].iloc[0]).multiply(100)
    fig = px.line(base100, labels={"value": "Índice base 100", "Date": "Fecha", "variable": "Activo"})
    fig.update_layout(height=520, legend_title_text="Activo")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Estadísticas por activo")
    stats_show = calc["stats"].copy().reset_index()
    pct_cols = ["E[r] Hist Mensual", "E[r] Hist Anual", "σ Mensual", "σ Anual", "E[r] CAPM Anual", "E[r] CAPM Mensual", "E[r] ANR Anual", "E[r] ANR Mensual", "Buy % ANR"]
    st.dataframe(stats_show.style.format({c: "{:.2%}" for c in pct_cols if c in stats_show.columns}).format({"Raw Beta": "{:.4f}", "Adjusted Beta": "{:.4f}", "Risk Unit Anual": "{:.4f}", "Risk Unit Mensual": "{:.4f}"}), use_container_width=True, height=520)

    st.subheader("Matriz de correlaciones")
    fig_corr = px.imshow(calc["corr"], text_auto=".2f", aspect="auto", zmin=-1, zmax=1)
    fig_corr.update_layout(height=720)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.subheader("Resumen de portafolios")
    st.dataframe(summary.style.format({
        "E[r] Anual": "{:.2%}", "σ Anual": "{:.2%}", "Sharpe": "{:.4f}", "VaR5%": "{:.2%}", "Risk Unit Anual": "{:.4f}"
    }), use_container_width=True)

    st.markdown("El portafolio de mínima varianza depende únicamente de la matriz varianza-covarianza. El portafolio de máxima Sharpe cambia al usar rendimiento histórico, CAPM o ANR porque cambia el vector de retornos esperados.")

with tab3:
    st.subheader("Frontera eficiente con retorno esperado CAPM")
    fig_frontier = go.Figure()
    if not frontier.empty:
        fig_frontier.add_trace(go.Scatter(x=frontier["Risk"], y=frontier["Return"], mode="lines", name="Frontera eficiente CAPM"))

    point_specs = [
        ("Equal Weight", summary_eq),
        ("Mínimo Riesgo", summary_min),
        ("Máx Sharpe CAPM", summary_capm),
    ]
    for label, row in point_specs:
        fig_frontier.add_trace(go.Scatter(x=[row["σ Anual"]], y=[row["E[r] Anual"]], mode="markers+text", text=[label], textposition="top center", name=label, marker=dict(size=11)))

    # Línea tangente / Capital Allocation Line desde Rf hasta un punto por encima del tangente
    x_line = np.linspace(0, max(float(summary_capm["σ Anual"]) * 1.25, frontier["Risk"].max() if not frontier.empty else 0.1), 50)
    y_line = rf_ea + float(summary_capm["Sharpe"]) * x_line
    fig_frontier.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Línea tangente / CAL", line=dict(dash="dash")))
    fig_frontier.add_trace(go.Scatter(x=[0], y=[rf_ea], mode="markers+text", text=["Rf"], textposition="bottom right", name="Tasa libre de riesgo"))

    fig_frontier.update_layout(height=620, xaxis_title="Riesgo anual σ", yaxis_title="Rendimiento esperado anual", yaxis_tickformat=".1%", xaxis_tickformat=".1%")
    st.plotly_chart(fig_frontier, use_container_width=True)

with tab4:
    st.subheader("Composición de portafolios")
    selected_weight_col = st.selectbox("Portafolio para visualizar", list(weights_df.columns), index=list(weights_df.columns).index("Máx Sharpe CAPM") if "Máx Sharpe CAPM" in weights_df.columns else 0)
    w_plot = weights_df[selected_weight_col].sort_values(ascending=False)
    fig_w = px.bar(w_plot, labels={"value": "Peso", "index": "Activo"}, title=f"Pesos - {selected_weight_col}")
    fig_w.update_layout(height=520, yaxis_tickformat=".1%", showlegend=False)
    st.plotly_chart(fig_w, use_container_width=True)
    st.dataframe(weights_df.style.format("{:.2%}"), use_container_width=True, height=520)

with tab5:
    st.subheader("Combinación óptima con activo libre de riesgo según VaR5%")
    st.markdown("Se usa como portafolio riesgoso de referencia el portafolio de máxima Sharpe con CAPM. El VaR5% se calcula como E[r] - Z·σ, con Z = INV.NORM.ESTAND(95%).")
    show_cols = ["VaR5% mínimo requerido", "Peso portafolio riesgoso", "Peso activo libre de riesgo", "E[r] total", "σ total", "VaR5% total", "Sharpe", "Factible"]
    st.dataframe(var_constraints[show_cols].style.format({
        "VaR5% mínimo requerido": "{:.2%}",
        "Peso portafolio riesgoso": "{:.2%}",
        "Peso activo libre de riesgo": "{:.2%}",
        "E[r] total": "{:.2%}",
        "σ total": "{:.2%}",
        "VaR5% total": "{:.2%}",
        "Sharpe": "{:.4f}",
    }), use_container_width=True)

    fig_var = px.bar(var_constraints, x="VaR5% mínimo requerido", y=["Peso portafolio riesgoso", "Peso activo libre de riesgo"], barmode="stack", labels={"value": "Peso", "variable": "Componente"})
    fig_var.update_layout(height=500, xaxis_tickformat=".1%", yaxis_tickformat=".1%")
    st.plotly_chart(fig_var, use_container_width=True)

with tab6:
    st.subheader("Descargar resultados")
    excel_bytes = build_download_excel(calc, summary, weights_df, frontier, var_constraints, rf_ea, erp_usa, z_value)
    st.download_button(
        label="Descargar Excel generado por la app",
        data=excel_bytes,
        file_name="portfolio_optimizer_resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.write("Este archivo contiene inputs, precios, rendimientos, estadísticas, matrices, pesos óptimos, frontera eficiente y restricciones VaR.")

st.divider()
st.caption("Modelo educativo. Rendimientos logarítmicos mensuales, desviación estándar poblacional, matriz anualizada multiplicando por 12, long-only y CAPM con ERP USA.")
