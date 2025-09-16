import time, hmac, hashlib
from urllib.parse import urlencode
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

import requests
import numpy as np
import pandas as pd

BASE_URL = "https://api.binance.com"    # Spot
PAIR_DEFAULT = "SOLUSDC"                # simbolo API
HUMAN_PAIR = "SOL/USDC"
EPS = 1e-12

# ---------- HTTP helpers ----------
def http_get(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None, timeout=30):
    return requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)

def binance_public_get(path: str, params: Dict[str, Any] = None):
    r = http_get(f"{BASE_URL}{path}", params=params)
    if r.status_code != 200:
        msg = r.text
        try:
            j = r.json()
            if "code" in j and "msg" in j:
                msg = f"{j['code']} {j['msg']}"
        except Exception:
            pass
        raise RuntimeError(f"Binance public error {r.status_code}: {msg}")
    return r.json()

def get_server_time_ms() -> int:
    data = binance_public_get("/api/v3/time")
    return int(data["serverTime"])

def binance_signed_get(path: str, params: Dict[str, Any], api_key: str, api_secret: str, ts_offset_ms: int = 0):
    params = {**params, "recvWindow": 60000, "timestamp": int(time.time()*1000) + ts_offset_ms}
    q = urlencode(params, doseq=True)
    sig = hmac.new(api_secret.encode(), q.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    url = f"{BASE_URL}{path}?{q}&signature={sig}"
    r = http_get(url, headers=headers)
    if r.status_code != 200:
        msg = r.text
        try:
            j = r.json()
            if "code" in j and "msg" in j:
                msg = f"{j['code']} {j['msg']}"
        except Exception:
            pass
        raise RuntimeError(f"Binance error {r.status_code}: {msg}")
    return r.json()

def validate_symbol_exists(symbol: str):
    info = binance_public_get("/api/v3/exchangeInfo", {"symbol": symbol})
    symbols = [s["symbol"] for s in info.get("symbols", [])]
    if symbol not in symbols:
        raise RuntimeError(f"Simbolo '{symbol}' non trovato su Binance Spot.")

# ---------- Fetchers ----------
def fetch_trades(api_key: str, api_secret: str, symbol: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    """
    Scarica i fill via /api/v3/myTrades rispettando il limite Binance:
    - Dividi in chunk da 3 giorni (massimo)
    - Se il chunk restituisce esattamente 1000 risultati, continua con fromId
    - Attende 1 secondo tra i chunk per evitare rate limit
    """
    CHUNK_MS = 3 * 24 * 60 * 60 * 1000 - 1  # 3 giorni
    server_ms = get_server_time_ms()
    local_ms = int(time.time() * 1000)
    ts_offset = server_ms - local_ms

    all_trades = []
    cursor_ms = start_ms

    last_id_seen = None
    while cursor_ms <= end_ms:
        window_end_ms = min(cursor_ms + CHUNK_MS, end_ms)
        base_params = {
            "symbol": symbol,
            "limit": 1000,
            "startTime": int(cursor_ms),
            "endTime": int(window_end_ms),
        }

        try:
            chunk = binance_signed_get("/api/v3/myTrades", base_params, api_key, api_secret, ts_offset)
        except Exception as e:
            print(f"❌ Errore su chunk {cursor_ms} → {window_end_ms}: {e}")
            raise

        if not isinstance(chunk, list):
            chunk = []

        all_trades.extend(chunk)

        # continua da fromId se 1000 risultati
        while len(chunk) == 1000:
            last_id = chunk[-1]["id"]
            if last_id == last_id_seen:
                break
            last_id_seen = last_id

            try:
                chunk = binance_signed_get(
                    "/api/v3/myTrades",
                    {"symbol": symbol, "limit": 1000, "fromId": last_id + 1},
                    api_key, api_secret, ts_offset
                )
            except Exception as e:
                print(f"❌ Errore da fromId {last_id+1}: {e}")
                break

            if not isinstance(chunk, list) or not chunk:
                break
            all_trades.extend(chunk)

        cursor_ms = window_end_ms + 1
        time.sleep(1.0)  # prevenzione rate limit

    all_trades.sort(key=lambda x: x["time"])
    all_trades = [t for t in all_trades if start_ms <= int(t["time"]) <= end_ms]
    return all_trades


def fetch_price_binance(symbol: str) -> Optional[float]:
    try:
        data = binance_public_get("/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])
    except Exception:
        return None

def fetch_price_coingecko() -> Optional[float]:
    r = http_get("https://api.coingecko.com/api/v3/simple/price",
                 params={"ids": "solana", "vs_currencies": "usd"}, timeout=15)
    if r.status_code == 200:
        try:
            return float(r.json()["solana"]["usd"])
        except Exception:
            return None
    return None

# ---------- Transforms & KPI ----------
def df_from_trades(trades: List[Dict[str, Any]], human_pair: str) -> pd.DataFrame:
    rows = []
    for t in trades:
        qty = float(t["qty"])
        price = float(t["price"])
        is_buyer = bool(t["isBuyer"])
        commission = float(t["commission"])
        commission_asset = t["commissionAsset"]

        total = qty * price
        fee_usdc = 0.0
        if commission > 0:
            if commission_asset == "USDC":
                fee_usdc = commission
            elif commission_asset == "SOL":
                fee_usdc = commission * price

        rows.append({
            "date": pd.to_datetime(t["time"], unit="ms"),
            "pair": human_pair,
            "type": "BUY" if is_buyer else "SELL",
            "qty": qty,
            "price": price,
            "total": total,
            "fee_usdc": fee_usdc,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def fifo_trades_per_match(orders: pd.DataFrame, pair_label: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    df = orders[orders["pair"] == pair_label].copy()
    if df.empty:
        return [], {"start": None, "end": None, "totalMoved": 0.0}, []

    buy_q: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    total_moved = float(df["total"].abs().sum())
    start = df["date"].min()
    end   = df["date"].max()

    for _, o in df.iterrows():
        if o["type"] == "BUY":
            buy_q.append(o.to_dict())
        elif o["type"] == "SELL":
            sell_qty = float(o["qty"])
            while sell_qty > EPS and buy_q:
                b = buy_q[0]
                take = float(min(b["qty"], sell_qty))
                pnl = (float(o["price"]) - float(b["price"])) * take

                fee_buy  = float(b.get("fee_usdc", 0.0)) * (take / float(b["qty"])) if float(b["qty"]) > 0 else 0.0
                fee_sell = float(o.get("fee_usdc", 0.0)) * (take / float(o["qty"])) if float(o["qty"]) > 0 else 0.0
                pnl -= (fee_buy + fee_sell)

                size_usdc = take * float(b["price"])
                dur_min = (o["date"] - b["date"]).total_seconds() / 60.0
                trades.append({
                    "buy_date": b["date"], "sell_date": o["date"],
                    "qty": take,
                    "buy_price": float(b["price"]), "sell_price": float(o["price"]),
                    "size_usdc": size_usdc, "pnl": pnl, "dur_min": dur_min
                })
                b["qty"] -= take
                sell_qty -= take
                if b["qty"] <= EPS:
                    buy_q.pop(0)

    inventory = []
    for b in buy_q:
        if float(b["qty"]) > EPS:
            inventory.append({
                "date": b["date"], "qty": float(b["qty"]),
                "avg_cost": float(b["price"]), "fee_usdc": float(b.get("fee_usdc", 0.0)),
            })
    return trades, {"start": start, "end": end, "totalMoved": total_moved}, inventory

def compute_kpis(trades: List[Dict[str, Any]], meta: Dict[str, Any], residual_cfg: Dict[str, float] | None) -> Dict[str, Any]:
    total_pnl = float(sum(t["pnl"] for t in trades))
    trades_n  = len(trades)
    best  = max(trades, key=lambda t: t["pnl"]) if trades else None
    worst = min(trades, key=lambda t: t["pnl"]) if trades else None
    avg_size = float(np.mean([t["size_usdc"] for t in trades])) if trades else 0.0
    avg_pnl  = float(np.mean([t["pnl"] for t in trades])) if trades else 0.0
    success_rate = float(100.0 * sum(1 for t in trades if t["pnl"] > 0) / trades_n) if trades_n else 0.0
    avg_dur  = float(np.mean([t["dur_min"] for t in trades])) if trades else 0.0

    residual_block = {"qty": 0.0, "avgCost": 0.0, "targetPrice": 0.0, "cost": 0.0, "value": 0.0, "pnl": 0.0}
    if residual_cfg and residual_cfg.get("qty"):
        residual_block["qty"]        = float(residual_cfg["qty"])
        residual_block["avgCost"]    = float(residual_cfg.get("avgCost", 0))
        residual_block["targetPrice"]= float(residual_cfg.get("targetPrice", 0))
        residual_block["cost"]  = residual_block["qty"] * residual_block["avgCost"]
        residual_block["value"] = residual_block["qty"] * residual_block["targetPrice"]
        residual_block["pnl"]   = residual_block["value"] - residual_block["cost"]

    total_with_residual = total_pnl + residual_block["pnl"]

    start = meta.get("start"); end = meta.get("end")
    months = 1
    if start and end:
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    monthly_avg = total_with_residual / max(1, months)

    last10 = []; last10_pnl = 0.0
    if end:
        cutoff = end - timedelta(days=10)
        last10 = [t for t in trades if t["sell_date"] >= cutoff]
        last10_pnl = float(sum(t["pnl"] for t in last10))

    return {
        "period": {"start": start, "end": end},
        "pnl": {"total": round(total_pnl, 2),
                "totalWithResidual": round(total_with_residual, 2),
                "residual": residual_block},
        "bestTrade": best, "worstTrade": worst,
        "counts": {"trades": trades_n, "successRatePct": round(success_rate, 2)},
        "sizes": {"avgTradeUSDC": round(avg_size, 2), "totalUSDCMoved": round(meta.get("totalMoved", 0.0), 2)},
        "perTrade": {"avgPnl": round(avg_pnl, 2), "avgDurationMin": round(avg_dur, 2)},
        "last10d": {"pnl": round(last10_pnl, 2), "trades": len(last10)},
        "monthlyAvgGain": round(monthly_avg, 2),
        "trades": trades,
    }
