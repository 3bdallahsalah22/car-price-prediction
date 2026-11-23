from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

with open(MODEL_DIR / "one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder = pickle.load(f)

with open(MODEL_DIR / "label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open(MODEL_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(MODEL_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)


class CarInput(BaseModel):
    Levy: int
    Manufacturer: str
    Model: str
    Prod_year: int
    Category: str
    Leather_interior: str
    Fuel_type: str
    Engine_volume: float
    Mileage: int
    Cylinders: float
    Gear_box_type: str
    Drive_wheels: str
    Wheel: str
    Color: str
    Airbags: int


@app.post("/predict/")
def predict(car_data: CarInput):
    try:
        data = pd.DataFrame([car_data.dict()])

        data = data.rename(columns={
           
            "Leather_interior": "Leather interior",
            "Fuel_type": "Fuel type",
            "Engine_volume": "Engine volume",
            "Gear_box_type": "Gear box type",
            "Drive_wheels": "Drive wheels"
        })

        data["Age"] = datetime.now().year - data["Prod_year"]

        data.drop(columns=["Doors", "Prod_year"], errors="ignore", inplace=True)

        one_hot_columns = ["Leather interior", "Gear box type", "Drive wheels", "Wheel"]

        encoded_data = one_hot_encoder.transform(data[one_hot_columns])
        encoded_data_df = pd.DataFrame(
            encoded_data,
            columns=one_hot_encoder.get_feature_names_out(one_hot_columns),
            index=data.index
        )

        data = pd.concat([data, encoded_data_df], axis=1)
        data.drop(columns=one_hot_columns, inplace=True)

        label_encode_columns = ["Manufacturer", "Model", "Category", "Fuel type", "Color"]

        for column in label_encode_columns:
            if column in data.columns:
                le = label_encoders[column]
                data[column] = le.transform(data[column])

        numerical_columns = ["Levy", "Engine volume", "Mileage", "Age"]
        data[numerical_columns] = scaler.transform(data[numerical_columns])

        prediction = model.predict(data)

        return {"prediction": float(prediction[0])}
   
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



    