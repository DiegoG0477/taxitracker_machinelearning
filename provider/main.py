from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
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

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

class HeatmapData(BaseModel):
    hour: int

def get_db_connection():
    db_connection_str = os.getenv('DB_CONNECTION_STRING')
    return create_engine(db_connection_str)

def load_model_from_db(hour: int, quadrant: str):
    engine = get_db_connection()
    query = text("""
    SELECT model_data FROM ml_models 
    WHERE hour = :hour AND quadrant = :quadrant
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {'hour': hour, 'quadrant': quadrant}).fetchone()
        print("Resultado de la consulta:", result)  # Para depuración
        
        if result:
            model_binary = io.BytesIO(result[0])  # Cambiado a result[0]
            model = joblib.load(model_binary)
            print(f"Modelo cargado de la base de datos para hora {hour} y cuadrante {quadrant}")
            return model
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
            # Use StreamingResponse to handle the compressed data
            response = StreamingResponse(buffer, media_type="application/json", headers={
                "Content-Encoding": "gzip",
                "Content-Length": str(buffer.getbuffer().nbytes)
            })
        return response

app.add_middleware(GZipMiddleware)

@app.post("/api/heatmap-data")
async def get_heatmap_data(data: HeatmapData):
    hour = data.hour
    quadrants = ['norte_oriente', 'norte_poniente', 'sur_oriente', 'sur_poniente']
    results = {}

    for quadrant in quadrants:
        model = load_model_from_db(hour, quadrant)
        if model is None:
            print(f"Modelo no encontrado para hora {hour} y cuadrante {quadrant}")
            continue
        
        # Create a grid of coordinates for predictions
        x_range = np.arange(16.7, 16.7 + 100 * 0.003, 0.003)  # Example
        y_range = np.arange(-93.8, -93.8 + 100 * 0.002, 0.002)  # Example
        grid_coords = np.array([[x, y] for x in x_range for y in y_range])
        
        # Obtain predictions from the model
        predictions = model.predict(grid_coords)
        
        results[quadrant] = {
            'x': x_range.tolist(),
            'y': y_range.tolist(),
            'z': predictions.tolist()
        }

    if not results:
        raise HTTPException(status_code=404, detail="No models found for the given hour")

    return results