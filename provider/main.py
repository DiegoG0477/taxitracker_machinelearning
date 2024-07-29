from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, async_sessionmaker
from pydantic import BaseModel
import joblib
import io
import numpy as np
from dotenv import load_dotenv
import os
import gzip
from io import BytesIO
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
from functools import lru_cache
import databases

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv('DB_CONNECTION_STRING')
database = databases.Database(DATABASE_URL)

app = FastAPI()

class HeatmapData(BaseModel):
    hour: int

@lru_cache(maxsize=100)
def load_model_from_bytes(model_bytes):
    model_binary = io.BytesIO(model_bytes)
    return joblib.load(model_binary)

async def load_model_from_db(hour: int, quadrant: str):
    query = text("""
    SELECT model_data FROM ml_models 
    WHERE hour = :hour AND quadrant = :quadrant
    """)
    result = await database.fetch_one(query, {'hour': hour, 'quadrant': quadrant})
    if result:
        return load_model_from_bytes(result[0])
    return None

class GZipMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if response.headers.get('Content-Type') == 'application/json':
            response_body = b"".join([chunk async for chunk in response.body_iterator])
            buffer = BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
                f.write(response_body)
            buffer.seek(0)
            response = StreamingResponse(buffer, media_type="application/json", headers={
                "Content-Encoding": "gzip",
                "Content-Length": str(buffer.getbuffer().nbytes)
            })
        return response

app.add_middleware(GZipMiddleware)

@app.post("/api/heatmap-data")
async def get_heatmap_data(data: HeatmapData):
    hour = data.hour
    lat_start, lat_end = 16.725405942451214, 16.805797678422447
    lon_start, lon_end = -93.13070664816753, -93.12799198419825
    lat_step, lon_step = 0.003, 0.003
    
    coordinates = [[lat, lon] for lat in np.arange(lat_start, lat_end, lat_step) for lon in np.arange(lon_start, lon_end, lon_step)]
    
    quadrants = ['norte_oriente', 'norte_poniente', 'sur_oriente', 'sur_poniente']
    results = {}

    for quadrant in quadrants:
        model = await load_model_from_db(hour, quadrant)
        if model is None:
            continue
        
        grid_coords = np.array(coordinates)
        predictions = model.predict(grid_coords)
        
        results[quadrant] = {
            'coordinates': coordinates,
            'predictions': predictions.tolist()
        }

    if not results:
        raise HTTPException(status_code=404, detail="No models found for the given hour")

    return results

async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()

app = FastAPI(lifespan=lifespan)