from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from datetime import datetime, timedelta
import uvicorn
import os
import json

from kpi_engine import fetch_trades_in_chunks, compute_kpis

load_dotenv()

app = FastAPI()

# Mount solo se cartella esiste
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
if os.path.isdir("templates"):
    templates = Jinja2Templates(directory="templates")
else:
    templates = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate")
async def calculate(request: Request):
    data = await request.form()
    pair = data.get("pair", "SOLUSDC").upper()
    start_str = data.get("start_date")
    end_str = data.get("end_date")

    api_key = data.get("api_key")
    api_secret = data.get("api_secret")

    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%dT%H:%M")
        end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H:%M")
    except Exception as e:
        return {"error": f"Formato data non valido: {e}"}

    # Step 1: scarica i trade da Binance (in chunk)
    try:
        trades = fetch_trades_in_chunks(
            api_key=api_key,
            api_secret=api_secret,
            symbol=pair,
            start_time=start_dt,
            end_time=end_dt,
            chunk_hours=48  # ogni chiamata copre 2 giorni
        )
    except Exception as e:
        return {"error": f"Errore fetch trades: {e}"}

    # Step 2: calcolo KPI
    try:
        kpis = compute_kpis(trades)
        return {"success": True, "data": kpis}
    except Exception as e:
        return {"error": f"Errore calcolo KPI: {e}"}
@app.get("/")
async def serve_index():
    return FileResponse("index.html")


# --------- CLI helper for local testing ---------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
