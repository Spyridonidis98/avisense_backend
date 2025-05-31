import numpy as np
import torch 
import pygame
import carla
import os

from PIL import Image
from io import BytesIO
import base64

os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()
pygame.display.set_mode((1,1))

class Sensors:
    class Camera:
        def __init__(self, camera_image, camera_transform, dimensions_meters, size, device):
            self.transform = camera_transform
            
            #convert image from base64 to numpy array  #data:image/jpeg;base64, needs to be stripped before decoding. The PIL Image.open() function expects pure base64 data, not the full data URL.
            camera_image = camera_image.split(',', 1)[1]
            self.image = np.array(Image.open(BytesIO(base64.b64decode(camera_image))))# (H,W,3)
            self.height, self.width, _ = self.image.shape
            
            #projection_matrix
            fov = 90.0 #to get later from camera calibration 
            projection_matrix = np.identity(3)
            projection_matrix[0, 2] = self.width / 2.0 # width in camera image pixels
            projection_matrix[1, 2] = self.height / 2.0 # height in camera image pixels
            projection_matrix[0, 0] = projection_matrix[1, 1] = self.width / (2.0 * np.tan(fov * np.pi / 360.0))
            self.projection_matrix = projection_matrix
        
class Lifting2Dto3D:
    @staticmethod
    def get_xyz_space(dimensions_meters = {"x":[-50, 50], "y":[-50, 50], "z": [0, 5]}, size = {"x":1024, "y":1024, "z":5}, device='cpu'):
        '''
        creates a space of voxels in carla coordinate system in meters 
        x axis is looking front 
        y axis is looking right  
        z axis is looking up  
        
        returns:
        xyz of shape(3,x,y,z) where 3 represents the xyz position  
        xyz_id of shape(3,x,y,z) where 3 is the index x,y,z of that cell 
        '''

        #xyz
        x = (torch.arange(size["x"], dtype=torch.float32) / (size["x"] -1))# range [0,1]
        x = x * (dimensions_meters['x'][1] - dimensions_meters['x'][0]) + dimensions_meters['x'][0]
        x = torch.flip(x, dims = [0])
        x = x.unsqueeze(1)
        x = x.repeat(1, size["x"])

        y = (torch.arange(size["y"], dtype=torch.float32) / (size["y"] -1))# range [0,1]
        y = y * (dimensions_meters['y'][1] - dimensions_meters['y'][0]) + dimensions_meters['y'][0]
        y = y.unsqueeze(0)
        y = y.repeat(size["y"], 1)

        xyz = torch.stack((x,y, torch.zeros_like(y)), dim=2).unsqueeze(2)
        xyz = xyz.repeat(1, 1, size["z"], 1)

        z = (torch.arange(size["z"], dtype=torch.float32) / (size["z"] -1))# range [0,1]
        z = z * (dimensions_meters['z'][1] - dimensions_meters['z'][0]) + dimensions_meters['z'][0]
        z = torch.flip(z, dims = [0])
        xyz[:,:,:,2] = z[:] 


        #xyz indices
        x_id = torch.arange(size["x"], dtype=torch.int64).unsqueeze(1)
        x_id = x_id.repeat(1, size["x"]) 

        y_id = torch.arange(size["y"], dtype=torch.int64).unsqueeze(0)
        y_id = y_id.repeat(size["y"], 1)

        xy_id = torch.stack((x_id, y_id, torch.zeros_like(y_id)), dim =2).unsqueeze(2)
        xyz_id = xy_id.repeat(1,1,size["z"],1)
        xyz_id[:,:,:, 2] = torch.arange(size["z"], dtype=torch.int64)

        xyz = xyz.permute([3,0,1,2]); xyz_id = xyz_id.permute([3,0,1,2]) # (x,y,z,3) -> (3,x,y,z)
        return xyz.to(device).contiguous(), xyz_id.to(device).contiguous()

    @staticmethod
    def render_xyz(camera, xyz, xyz_ids, device='cpu'):
        '''        
        xyz of shape(3,x,y,z) where 3 represents the xyz position  
        xyz_id of shape(3,x,y,z) where 3 is the index x,y,z of that cell 
        uvs shape [2, n] to calculate gather_ids used for image of shape [C,H,W]
        scatter_ids shape [3, n] used for volume of shape [C,H,W,Z]  
        '''
        world_2_camera = torch.tensor(camera.transform.get_inverse_matrix(), dtype=torch.float32)
        R = world_2_camera[:3,:3].to(device) #shape (3,3) 
        T = world_2_camera[:3, 3].to(device) #shape (3)
        K = torch.tensor(camera.projection_matrix, dtype=torch.float32).to(device) #shape (3,3)

        # R, T is from camera to body 
        # xyz relative to the car body, xyz_cam is relative to camera
        xyz_cam = (R @ xyz.view(3,-1) + T.unsqueeze(1))
        xyz_cam_new = xyz_cam.clone()
        #from UE4's coordinate system to an standard, (x, y ,z) -> (y, -z, x)
        xyz_cam_new[0, ...] = xyz_cam[1, ...]
        xyz_cam_new[1, ...] = -xyz_cam[2, ...] 
        xyz_cam_new[2, ...] = xyz_cam[0, ...] 
                
        #project points
        uvs = K @ xyz_cam_new
        uvs[0] = uvs[0]/ uvs[2]
        uvs[1] = uvs[1]/ uvs[2]

        bf = (uvs[0, :] >= 0) & (uvs[1, :] >= 0) & (uvs[0, :] < camera.width-1) & (uvs[1, :]<camera.height-1) & (uvs[2, :] > 0.1)
        uvs = uvs[:, bf]

        xyz_ids_filtered = xyz_ids.view(3, -1)[:, bf]
        scatter_ids = xyz_ids_filtered

        return uvs[:2], scatter_ids


    def lift_features(img, volume, uvs, scatter_ids):
            '''
            Lifting features using bilinear interpolation 
            
            img shape (3, H, W)
            volume shape (3, X, Y, Z)
            uvs shape (2,n)
            scatter_ids shape (3,n)
            '''
            gather_ids = torch.zeros_like(uvs)
            gather_ids[0, :] = uvs[1, :]
            gather_ids[1, :] = uvs[0, :]

            gather_ids_dl = gather_ids.clone() #down left 
            gather_ids_dl[0, :] = gather_ids_dl[0, :].floor()
            gather_ids_dl[1, :] = gather_ids_dl[1, :].floor()

            gather_ids_ul = gather_ids.clone() #up left 
            gather_ids_ul[0, :] = gather_ids_ul[0, :].floor() + 1.0
            gather_ids_ul[1, :] = gather_ids_ul[1, :].floor()

            gather_ids_ur = gather_ids.clone() #up right
            gather_ids_ur[0, :] = gather_ids_ur[0, :].floor() + 1.0
            gather_ids_ur[1, :] = gather_ids_ur[1, :].floor() + 1.0

            gather_ids_dr = gather_ids.clone() #down right 
            gather_ids_dr[0, :] = gather_ids_dr[0, :].floor()
            gather_ids_dr[1, :] = gather_ids_dr[1, :].floor() + 1.0

            x_u = gather_ids_ul[0, :] #up 
            x_d = gather_ids_dl[0, :] #down
            y_r = gather_ids_dr[1, :] #right
            y_l = gather_ids_dl[1, :] #left 

            w_dl = (x_u - gather_ids[0]) * (y_r - gather_ids[1]) #/ (x_u - x_d) * (y_r - y_l) #down left weights, no need to divide since the volume is always 1  
            w_ul = (gather_ids[0] - x_d) * (y_r - gather_ids[1]) #/ (x_u - x_d) * (y_r - y_l) #up left weights 
            w_ur = (gather_ids[0] - x_d) * (gather_ids[1] - y_l) #/ (x_u - x_d) * (y_r - y_l) #up right weights 
            w_dr = (x_u - gather_ids[0]) * (gather_ids[1] - y_l) #/ (x_u - x_d) * (y_r - y_l) #down right weights 

            gather_ids_dl = gather_ids_dl.to(torch.int64)
            gather_ids_ul = gather_ids_ul.to(torch.int64)
            gather_ids_ur = gather_ids_ur.to(torch.int64)
            gather_ids_dr = gather_ids_dr.to(torch.int64)

            volume[:, scatter_ids[0], scatter_ids[1], scatter_ids[2]] =  w_dl * img[:, gather_ids_dl[0], gather_ids_dl[1]] \
                                                                        + w_ul * img[:, gather_ids_ul[0], gather_ids_ul[1]] \
                                                                        + w_ur * img[:, gather_ids_ur[0], gather_ids_ur[1]] \
                                                                        + w_dr * img[:, gather_ids_dr[0], gather_ids_dr[1]] \
                                                                        
            return volume

    @staticmethod 
    def lift_camera_to_bev_surface(camera, 
             dimensions_meters = {"x":[-50, 50], "y":[-50, 50], "z": [0, 2.5]}, 
             size = {"x":1024, "y":1024, "z":5}, device = 'cpu',
             transparent_background = True):
        
        image = torch.tensor(camera.image).to(device).permute([2,0,1]) #(H,W,C) -> (C,H,W)
        volume = torch.zeros(size = (3, *tuple(size.values())), dtype=torch.float32).to(device) #shape (3, x, y, z)
        
        xyz, xyz_ids = Lifting2Dto3D.get_xyz_space(dimensions_meters = dimensions_meters, size = size, device=device)
        uvs, scatter_ids = Lifting2Dto3D.render_xyz(camera, xyz, xyz_ids, device=device)
        volume = Lifting2Dto3D.lift_features(image, volume, uvs, scatter_ids)

        #select the last z slice which is the floor with z = 0 
        pygame_image = volume[:,:,:, -1].permute([2,1,0]).to(torch.uint8).to('cpu').numpy() #(C,X,Y,Z)->(C,X,Y), (C,H,W)->(W,H,C), pygame takes width as first argument 
        surface = pygame.surfarray.make_surface(pygame_image).convert(); 
        if transparent_background: surface.set_colorkey((0,0,0))
    
        return surface

class Transforms:
    @staticmethod
    def gps2meters(place_latitude, place_longitude, camera_latitude, camera_longitude):
        """
        Convert GPS coordinates to meters relative to a reference point.
        Based on unreal engine coordinate system. x is forward, y is right 
        """
        earth_radius = 6371000.0
        place_latitude_rad = np.radians(place_latitude)
        place_longitude_rad = np.radians(place_longitude)
        camera_latitude_rad = np.radians(camera_latitude)
        camera_longitude_rad = np.radians(camera_longitude)

        delta_longitude = camera_longitude_rad - place_longitude_rad
        delta_latitude = camera_latitude_rad - place_latitude_rad


        x = earth_radius * delta_latitude
        y = earth_radius * delta_longitude * np.cos(place_latitude_rad)

        # Convert degrees to radians
        return (x,y)


class ML:
    def __init__(self, sql_client):
        self.sql_client = sql_client

    async def get_cameras_projection_on_map(self, place_name):
        #get place data 
        place_data = await self.sql_client.load_place(place_name)
        patch_meters_x = float(place_data["patch_meters_x"])
        patch_meters_y = float(place_data["patch_meters_y"])
        
        dimensions_meters = {
            "x":[-patch_meters_x/2, patch_meters_x/2],
            "y":[-patch_meters_y/2, patch_meters_y/2],
            "z": [0, 2.5]}
        
        size = {"x":1024, "y":1024, "z":5}
        device = 'cpu'
        
        #get cameras data in that place 
        cameras= await self.sql_client.get_camera_frame(place_name)
        
        #create an empty pygame surface 
        lifted_cameras_surface = pygame.Surface((1024, 1024))
        
        for camera in cameras.values():
            #get position of the camera relative to the place, the position is in unreal engine coordinate system 
            camera_pos_x, camera_pos_y = Transforms.gps2meters(float(place_data["position_latitude"]), float(place_data["position_longitude"]), float(camera['position_latitude']), float(camera['position_longitude']))
            camera_pos_z = float(camera['position_height']) # in meters
            camera_orientation_z = float(camera['orientation_z']) #yaw in degrees
            camera_orientation_y = float(camera['orientation_y']) #pitch in degrees
            camera_orientation_x = 0.0 #roll in degrees
                
            #basically the spawn point of the camera 
            camera_transform = carla.Transform(location=carla.Location(x=camera_pos_x, y=camera_pos_y, z=camera_pos_z), rotation=carla.Rotation(yaw=camera_orientation_z, pitch=camera_orientation_y, roll=camera_orientation_x))
            
            #convert to camera class
            camera = Sensors.Camera(camera['image_data'], camera_transform, dimensions_meters, size, device)
            camera_lifted_surface = Lifting2Dto3D.lift_camera_to_bev_surface(camera, dimensions_meters, size, device)
            lifted_cameras_surface.blit(camera_lifted_surface, (0, 0))
        bev_lifted_features = pygame.surfarray.array3d(lifted_cameras_surface).transpose(1,0,2)
        alpha  = (~np.all(bev_lifted_features == np.zeros(3), axis = -1)[...,None]).astype(np.uint8) * 255 #add alpha 
        bev_lifted_features = np.concatenate((bev_lifted_features, alpha), axis = -1)
        #Image.fromarray(bev_lifted_features, mode='RGBA').save('carla_bev_lifted_features.png')
        
        # After creating the PIL image
        pil_image = Image.fromarray(bev_lifted_features, mode='RGBA')
        # Convert to base64 data URL
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        data_url = f"data:image/png;base64,{img_str}"

        return data_url