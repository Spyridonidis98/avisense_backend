from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import base64
from database.sql.client import create_db_pool, SQLClient
from ml.ml import ML

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    #database connection
    global sql_client
    pool = await create_db_pool()
    sql_client = SQLClient(pool)
    
    #ml connection
    global ml_client
    ml_client = ML(sql_client)
    
    yield  # App runs here
    
    # Shutdown
    if sql_client and sql_client.connection_pool:
        await sql_client.connection_pool.close()
        print("Database connection pool closed")

app = FastAPI(lifespan=lifespan)
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

#avisense platform
# Endpoint to receive camera frames
@app.post("/send-camera-frame")
async def send_camera_frame(data: dict):
    image = data['image']
    camera_name = data['camera_name']
    place = data['place']

    
    try:
        # Save camera frame to database
        await sql_client.save_camera_frame(
            camera_name=camera_name,
            place_name=place,
            image_data=image
        )
        
        return {
            'message': 'camera data received and saved',
            'status': 'success',
            'camera': camera_name,
            'place': place
        }
    except Exception as e:
        return {
            'message': 'failed to save camera data',
            'status': 'error',
            'error': str(e)
        }

@app.post("/get-camera-frame")
async def get_camera_frame(data: dict):
    place = data['place']
    
    try:
        cameras = await sql_client.get_camera_frame(place)
        
        return {
            'success': True,
            'cameras': cameras,
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }

@app.delete("/delete-camera")
async def delete_camera(data: dict):
    camera_name = data['camera_name']
    place_name = data['place_name']
    
    try:
        await sql_client.delete_camera(camera_name, place_name)
        return {
            'success': True,
            'message': f'Camera "{camera_name}" deleted successfully from place "{place_name}"'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Failed to delete camera "{camera_name}" from place "{place_name}"'
        }


@app.get("/get-places")
async def get_places():
    try:
        places = await sql_client.get_places()
        
        return {
            'success': True,
            'places': places
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'places': []
        }

#create a new place or update an existing place
@app.post("/save-place")
async def save_place(data: dict):
    place_name = data['place_name']
    position_longitude = data['position_longitude']
    position_latitude = data['position_latitude']
    patch_meters_x = data['patch_meters_x']
    patch_meters_y = data['patch_meters_y']
    
    try:
        await sql_client.save_place(place_name, position_longitude, position_latitude, patch_meters_x, patch_meters_y)
        return {
            'success': True,
            'message': f'Place "{place_name}" updated successfully'
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'error': str(e),
                'message': f'Failed to create place "{place_name}"'
            }
        )

@app.post("/load-place")
async def load_place(data: dict):
    place_name = data['place_name']
    
    try:
        place_data = await sql_client.load_place(place_name)
        
        return {
            'success': True,
            'place_data': place_data,
            'message': f'Place "{place_name}" loaded successfully'
        }
    except Exception as e:
        raise HTTPException(
            status_code=404 if "not found" in str(e).lower() else 500,
            detail={
                'success': False,
                'error': str(e),
                'message': f'Failed to load place "{place_name}"'
            }
        )

@app.post("/save-camera-position")
async def save_camera_position(data: dict):
    camera_name = data['camera_name']
    place_name = data['place_name']
    position_latitude = data['position_latitude']
    position_longitude = data['position_longitude']
    position_height = data['position_height']
    orientation_z = data['orientation_z']
    orientation_y = data['orientation_y']
    
    try:
        rows_affected = await sql_client.save_camera_position(
            camera_name=camera_name,
            place_name=place_name,
            position_height=position_height,
            position_latitude=position_latitude,
            position_longitude=position_longitude,
            orientation_z=orientation_z,
            orientation_y=orientation_y
        )
        
        return {
            'success': True,
            'message': f'Camera position updated successfully for "{camera_name}"',
            'camera_name': camera_name,
            'rows_affected': rows_affected
        }
    except Exception as e:
        raise HTTPException(
            status_code=404 if "not found" in str(e).lower() else 500,
            detail={
                'success': False,
                'error': str(e),
                'message': f'Failed to update camera position for "{camera_name}"'
            }
        )

@app.post("/get-cameras-projection-on-map")
async def get_cameras_projection_on_map(data: dict):
    place_name = data['place_name']
    try:
        projection_image = await ml_client.get_cameras_projection_on_map(place_name)
        return {
            'success': True,
            'projection_image': projection_image
        }   
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }
# run locally : uvicorn backend:app --host 0.0.0.0 --port 8000 --reload