import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from kpi_engine import (
    PAIR_DEFAULT, HUMAN_PAIR, validate_symbol_exists,
    fetch_trades, df_from_trades, fifo_trades_per_match,
    compute_kpis, fetch_price_binance, fetch_price_coingecko
)

app = FastAPI(title="Crypto KPI Dashboard – SOL/USDC")

# Monta la cartella static solo se esiste
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

def fmt(x: float) -> str:
    return f"{x:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")

def fmt_duration(mins: float | int) -> str:
    if mins is None:
        return "-"
    mins = float(mins)
    if mins < 60: return f"~{fmt(mins)} min"
    hours = mins/60
    if hours < 24: return f"~{fmt(hours)} h"
    days = hours/24
    return f"~{fmt(days)} giorni"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    end = datetime.utcnow()
    start = end - timedelta(days=180)  # Default: ultimi 6 mesi
    return templates.TemplateResponse("index.html", {
        "request": request,
        "pair": HUMAN_PAIR,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "result": None,
        "inventory": None,
        "live_price": None,
        "error": None,
        "fmt": fmt,
        "fmt_duration": fmt_duration
    })

@app.post("/calc", response_class=HTMLResponse)
async def calc(request: Request,
               start_date: str = Form(...),
               end_date: str = Form(...),
               res_qty: float = Form(15.0),
               res_avg: float = Form(201.0),
               res_target: float = Form(220.0)):
    error: Optional[str] = None
    result = None; inventory = None; live_price = None

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        error = "Chiavi Binance mancanti. Imposta BINANCE_API_KEY e BINANCE_API_SECRET su Heroku."
        return templates.TemplateResponse("index.html", {
            "request": request, "pair": HUMAN_PAIR, "start": start_date, "end": end_date,
            "result": None, "inventory": None, "live_price": None, "error": error,
            "fmt": fmt, "fmt_duration": fmt_duration
        })

    try:
        validate_symbol_exists(PAIR_DEFAULT)
    except Exception as e:
        error = str(e)

    try:
        start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ms   = int((datetime.strptime(end_date, "%Y-%m-%d") + timedelta(hours=23, minutes=59, seconds=59)).timestamp() * 1000)
    except Exception:
        error = "Formato data non valido. Usa YYYY-MM-DD."

    if not error:
        try:
            trades_raw = fetch_trades(api_key, api_secret, PAIR_DEFAULT, start_ms, end_ms)
        except Exception as e:
            msg = str(e)
            if "451" in msg or "restricted location" in msg.lower():
                error = ("Binance ha bloccato le API dal datacenter (451: restricted location). "
                         "Scegli regione **EU** per l'app Heroku e riprova. In alternativa usa un altro host consentito.")
            else:
                error = f"Errore fetch trade: {msg}"

    if not error:
        if not trades_raw:
            error = "Nessun trade trovato nell'intervallo."
        else:
            orders = df_from_trades(trades_raw, HUMAN_PAIR)
            trades, meta, inventory = fifo_trades_per_match(orders, HUMAN_PAIR)
            residual_cfg = {"qty": res_qty, "avgCost": res_avg, "targetPrice": res_target}
            result = compute_kpis(trades, meta, residual_cfg)
            live_price = fetch_price_binance(PAIR_DEFAULT) or fetch_price_coingecko()

    return templates.TemplateResponse("index.html", {
        "request": request, "pair": HUMAN_PAIR,
        "start": start_date, "end": end_date,
        "result": result, "inventory": inventory, "live_price": live_price,
        "error": error, "fmt": fmt, "fmt_duration": fmt_duration
    })
