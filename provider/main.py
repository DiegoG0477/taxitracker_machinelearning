from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from pydantic import BaseModel
import joblib
import io
import numpy as np
import uvicorn

app = FastAPI()

class HeatmapData(BaseModel):
    hour: int

def get_db_connection():
    db_connection_str = 'mysql+pymysql://root:$ui89<>-!gdRLDA145@jaguartech-database.ch4runvfltbm.us-east-1.rds.amazonaws.com/taxitracker'
    return create_engine(db_connection_str)

def load_model_from_db(hour: int, quadrant: str):
    engine = get_db_connection()
    query = text("""
    SELECT model_data FROM ml_models 
    WHERE hour = :hour AND quadrant = :quadrant
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {'hour': hour, 'quadrant': quadrant}).fetchone()
        if result:
            model_binary = io.BytesIO(result['model_data'])
            model = joblib.load(model_binary)
            return model
        return None

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

# --launcher-skip