import asyncpg
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def create_db_pool():
    """Create and return a PostgreSQL connection pool"""
    db_config = {
        'host': os.getenv('DB_HOST', os.getenv('SQL_DATABASE_HOST')),
        'port': os.getenv('DB_PORT',  os.getenv('SQL_DATABASE_PORT')),
        'user': os.getenv('DB_USER', os.getenv('SQL_DATABASE_USER')),
        'password': os.getenv('DB_PASSWORD', os.getenv('SQL_DATABASE_PASSWORD')),
        'database': os.getenv('DB_NAME', os.getenv('SQL_DATABASE_NAME'))
    }
    
    try:
        pool = await asyncpg.create_pool(**db_config)
        print("Database connection pool created successfully")
        return pool
    except Exception as e:
        print(f"Failed to create database connection pool: {e}")
        raise

class SQLClient:
    def __init__(self, connection_pool):
        """Initialize SQLClient with an existing connection pool"""
        self.connection_pool = connection_pool

    async def save_camera_frame(self, camera_name: str, place_name: str, image_data: str):
        """Save camera frame data to the database"""
        if not self.connection_pool:
            raise Exception("Database connection pool not provided")

        query = """
            INSERT INTO camera (camera_name, place_name, image_data)
            VALUES ($1, $2, $3)
            ON CONFLICT (camera_name, place_name) 
            DO UPDATE SET 
                image_data = EXCLUDED.image_data
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    query, camera_name, place_name, image_data)
                print(f"Camera frame saved successfully: {camera_name} at {place_name}")
                return True
        except Exception as e:
            print(f"Error saving camera frame: {e}")
            raise

    async def get_camera_frame(self, place_name: str):
        """Get all camera frames for a specific place"""
        if not self.connection_pool:
            raise Exception("Database connection pool not provided")

        query = """
            SELECT camera_name, place_name, image_data, resolution, bytes_size,
                   position_longitude, position_latitude, position_height, orientation_x, orientation_y, orientation_z
            FROM camera 
            WHERE place_name = $1
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(query, place_name)
                
                # Convert rows to dictionary format
                cameras = {}
                for row in rows:
                    camera_name = row['camera_name']
                    cameras[camera_name] = {
                        'camera_name': row['camera_name'],
                        'place_name': row['place_name'],
                        'image_data': row['image_data'],
                        'size': len(row['image_data']) if row['image_data'] else 0,
                        'position_longitude': row['position_longitude'],
                        'position_latitude': row['position_latitude'],
                        'position_height': row['position_height'],
                        'orientation_y': row['orientation_y'],
                        'orientation_z': row['orientation_z']
                    }
                
                print(f"Retrieved {len(cameras)} camera frames for place: {place_name}")
                return cameras
                
        except Exception as e:
            print(f"Error retrieving camera frames: {e}")
            raise

    async def get_places(self):
        """Get all unique place names from the camera table"""
        if not self.connection_pool:
            raise Exception("Database connection pool not provided")

        query = """
            SELECT DISTINCT place_name 
            FROM place 
            ORDER BY place_name
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(query)
                
                # Extract place names from rows
                places = [row['place_name'] for row in rows]
                
                print(f"Retrieved {len(places)} unique places from database")
                return places
                
        except Exception as e:
            print(f"Error retrieving places: {e}")
            raise

    async def save_place(self, place_name: str, position_longitude: str, position_latitude: str, 
                        patch_meters_x: str, patch_meters_y: str):
        """Save or update place data to the database"""
        if not self.connection_pool:
            raise Exception("Database connection pool not provided")

        query = """
            INSERT INTO place (place_name, position_longitude, position_latitude, patch_meters_x, patch_meters_y)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (place_name) 
            DO UPDATE SET 
                position_longitude = EXCLUDED.position_longitude,
                position_latitude = EXCLUDED.position_latitude,
                patch_meters_x = EXCLUDED.patch_meters_x,
                patch_meters_y = EXCLUDED.patch_meters_y
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    query, place_name, position_longitude, position_latitude, 
                    patch_meters_x, patch_meters_y
                )
                print(f"Place saved successfully: {place_name} at ({position_longitude}, {position_latitude})")
                return True
        except Exception as e:
            print(f"Error saving place: {e}")
            raise

    async def load_place(self, place_name: str):
        """Load a specific place's data from the database"""
        if not self.connection_pool:
            raise Exception("Database connection pool not provided")

        query = """
            SELECT place_name, position_longitude, position_latitude, patch_meters_x, patch_meters_y
            FROM place 
            WHERE place_name = $1
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                row = await connection.fetchrow(query, place_name)
                
                if row:
                    place_data = {
                        'place_name': row['place_name'],
                        'position_longitude': row['position_longitude'],
                        'position_latitude': row['position_latitude'],
                        'patch_meters_x': row['patch_meters_x'],
                        'patch_meters_y': row['patch_meters_y']
                    }
                    print(f"Place loaded successfully: {place_name}")
                    return place_data
                else:
                    raise Exception(f"Place '{place_name}' not found")
                
        except Exception as e:
            print(f"Error loading place: {e}")
            raise

    async def save_camera_position(self, camera_name: str, place_name: str, 
                                 position_latitude: str, position_longitude: str, 
                                 position_height: str, orientation_z: str, 
                                 orientation_y: str):
        """Update camera position and orientation data"""
        if not self.connection_pool:
            raise Exception("Database connection pool not provided")

        query = """
            UPDATE camera 
            SET 
                position_longitude = $3,
                position_latitude = $4,
                position_height = $5,
                orientation_z = $6,
                orientation_y = $7
            WHERE camera_name = $1 AND place_name = $2
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                result = await connection.execute(
                    query, camera_name, place_name, position_longitude, position_latitude, 
                    position_height, orientation_z, orientation_y
                )
                
                # Check if any rows were updated
                rows_affected = int(result.split()[-1]) if result else 0
                
                if rows_affected > 0:
                    print(f"Camera position updated successfully: {camera_name} - {rows_affected} record(s) updated")
                    return rows_affected
                else:
                    raise Exception(f"Camera '{camera_name}' not found")
                
        except Exception as e:
            print(f"Error updating camera position: {e}")
            raise

    async def delete_camera(self, camera_name: str, place_name: str):
        """Delete a camera record from the database"""
        if not self.connection_pool:
            raise Exception("Database connection pool not provided")

        query = """
            DELETE FROM camera 
            WHERE camera_name = $1 AND place_name = $2
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                result = await connection.execute(query, camera_name, place_name)
                
                # Check if any rows were deleted
                rows_affected = int(result.split()[-1]) if result else 0
                
                if rows_affected > 0:
                    print(f"Camera deleted successfully: {camera_name} at {place_name} - {rows_affected} record(s) deleted")
                    return rows_affected
                else:
                    raise Exception(f"Camera '{camera_name}' at place '{place_name}' not found")
                
        except Exception as e:
            print(f"Error deleting camera: {e}")
            raise


