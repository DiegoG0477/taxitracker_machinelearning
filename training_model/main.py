import io
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pymysql
from shapely import wkt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    db_connection_str = os.getenv('DB_CONNECTION_STRING')
    return create_engine(db_connection_str)

def save_model_to_db(hour, quadrant, model):
    engine = get_db_connection()
    model_binary = io.BytesIO()
    joblib.dump(model, model_binary)
    model_binary.seek(0)
    
    query = text("""
    INSERT INTO ml_models (hour, quadrant, model_data)
    VALUES (:hour, :quadrant, :model_data)
    ON DUPLICATE KEY UPDATE model_data = :model_data
    """)
    
    try:
        with engine.connect() as conn:
            conn.execute(query, {
                'hour': hour,
                'quadrant': quadrant,
                'model_data': model_binary.getvalue()
            })
            conn.commit()
        print(f"Modelo guardado en la base de datos para hora {hour} y cuadrante {quadrant}")
    except Exception as e:
        print(f"Error al guardar el modelo en la base de datos: {e}")

def parse_coordinates(coord_text):
    if coord_text:
        try:
            point = wkt.loads(coord_text)
            return point.y, point.x  # latitud, longitud
        except Exception as e:
            print(f"Error al parsear coordenadas: {coord_text}")
            print(f"Error: {e}")
    return None, None

def load_data():
    engine = get_db_connection()
    query = '''
    SELECT driver_id, date, start_hour, 
        ST_AsText(start_coordinates) AS start_coordinates, 
        ST_AsText(end_coordinates) AS end_coordinates, 
        duration, distance_mts
    FROM travels
    WHERE start_coordinates IS NOT NULL AND end_coordinates IS NOT NULL
    '''
    df = pd.read_sql(query, con=engine)

    print("Datos cargados:", len(df))
    
    # Extraer latitud y longitud de las coordenadas
    df[['start_longitude', 'start_latitude']] = df['start_coordinates'].apply(lambda x: pd.Series(parse_coordinates(x)))
    df[['end_longitude', 'end_latitude']] = df['end_coordinates'].apply(lambda x: pd.Series(parse_coordinates(x)))
    
    # Convertir start_hour a hora y day_of_week
    df['hour'] = pd.to_datetime(df['start_hour']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['start_hour']).dt.dayofweek
    
    # Determinar cuadrante
    df['quadrant'] = df.apply(determine_quadrant, axis=1)
    print(df['quadrant'].value_counts())  # Imprimir conteo de cada cuadrante
    
    print("Columnas después del procesamiento:", df.columns)
    print("Tipos de datos después del procesamiento:")
    print(df.dtypes)
    print("\nMuestra de datos procesados:")
    print(df[['hour', 'day_of_week', 'start_latitude', 'start_longitude', 'quadrant']].head())
    
    return df

def determine_quadrant(row):
    lat = row['start_latitude']
    lon = row['start_longitude']
    if lat is not None and lon is not None:
        if lat >= 16.75897067747087:
            if lon <= -93.11945116216742:
                return 'sur_poniente'
            else:
                return 'sur_oriente'
        else:
            if lon <= -93.11945116216742:
                return 'norte_poniente'
            else:
                return 'norte_oriente'
    return 'desconocido'

def train_and_save_model(hour, quadrant, data):
    hourly_data = data[(data['hour'] == hour) & (data['quadrant'] == quadrant)]
    
    if hourly_data.empty:
        print(f"No hay datos para la hora {hour} y el cuadrante {quadrant}")
        return
    
    X = hourly_data[['start_latitude', 'start_longitude']]
    y = hourly_data['distance_mts']
    
    if len(hourly_data) < 3:
        print(f"No hay suficientes datos para entrenar el modelo para la hora {hour} y el cuadrante {quadrant}")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    param_grid = {'fit_intercept': [True, False]}
    grid_search = GridSearchCV(model, param_grid, cv=3)
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        print(f"Reporte de regresión para la hora {hour} y cuadrante {quadrant}:")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        print(f"R^2 Score: {r2_score(y_test, y_pred)}")
        
        save_model_to_db(hour, quadrant, best_model)
    except ValueError as e:
        print(f"Error en el ajuste del modelo para la hora {hour} y cuadrante {quadrant}: {e}")

def process_hour(hour, data):
    for quadrant in data['quadrant'].unique():
        print(f"Procesando hora {hour} y cuadrante {quadrant}")
        train_and_save_model(hour, quadrant, data)

def main():
    df = load_data()
    
    hours = df['hour'].unique()
    
    for hour in hours:
        process_hour(hour, df)

if __name__ == "__main__":
    main()