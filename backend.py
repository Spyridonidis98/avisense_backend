from fastapi import FastAPI
import psycopg2
from pydantic import BaseModel

app = FastAPI()
users_position = {}

@app.get("/")
def root():
    return users_position
  
@app.post("/send_position")
def get_position(data: dict):
    user = data['user']
    x = float(data['x'])
    z = float(data['z'])
    angle = float(data['angle'])
    users_position[user] = {'x':x, 'z':z, 'angle':angle} 
    
    return 'data received'
    
    
# @app.post("/test_post")
# def insert_test(data: dict):
#     a = data['x']
#     b = data['y']
#     return [a,b]