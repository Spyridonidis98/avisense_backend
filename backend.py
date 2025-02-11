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
    print(data)
    user = data['user']
    user_type = data['user_type'] #car, bike, pedestrian 
    x = float(data['x'])#longitude
    z = float(data['z'])#latitude
    angle = float(data['angle'])
    users_position[user] = {'x':x, 'z':z, 'angle':angle, 'user_type':user_type}    


    return 'data received'
    
@app.delete("/delete_user")
def delete_user(data: dict):
    global users_position  
    user = data['user']
    if data['user'] == 'all':
        users_position = {}
        return 'deleted all users'
    elif data['user'] == '':
        return 'No user name'
    else:
        users_position.pop(user)
    return "user deleted"
# @app.post("/test_post")
# def insert_test(data: dict):
#     a = data['x']
#     b = data['y']
#     return [a,b]