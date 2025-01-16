
# %% using trimesh (cannot do depth?) - good for 3D visualization (only glb apparently but blender can export to glb)
# MARK: TRIMESH
import trimesh as tm
import os 
import numpy as np

# # obj = "/home/thanos/OneDrive/mybackupOneDrive/PhD/3Dif/data/low-poly-isometric-room/source/obj/obj.obj" # missing some textures (not all...)
# # obj = "/home/thanos/OneDrive/mybackupOneDrive/PhD/3dif_objaverse-xl-subset/testing/noised.obj" # works fine
# obj = "/home/thanos/OneDrive/mybackupOneDrive/PhD/3dif_objaverse-xl-subset/testing/main.obj"
# mesh = tm.load(obj, process=False)

obj = "/home/thanos/OneDrive/mybackupOneDrive/PhD/3dif_objaverse-xl-subset/earth/untitled.obj"
mesh = tm.load(obj, process=False)

# why is this not working anymore???
# mesh.set_camera(angles=(0, 0, 0), distance=40, center=(0,0,0))

# for pyglet: pip install --upgrade pyglet==v1.5.28, only pyglet<2 is supported
mesh.show(viewer='gl')

# %% noise
# MARK: NOISE
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

os.chdir("/home/thanos/OneDrive/mybackupOneDrive/PhD/3dif_objaverse-xl-subset/testing/")

# # Create a sample input tensor X
# X = torch.randn(1, 3, 128, 128)

# # Set the number of upsample iterations
# num_upsamples = 4
# # Set the maximum resolution
# max_resolution = 2048

# # Get initial dimensions
# b, c, h, w = X.shape
# initial_resolution = h

# # Calculate the scale factor
# # scale_factor = (max_resolution / initial_resolution) ** (1 / num_upsamples)
# scale_factor = 2

# # List to hold tensors at each scale
# noised_tensors = [X]
# current_tensor = X.clone()

# for i in range(num_upsamples):
#     b, c, h, w = current_tensor.shape
#     # Calculate new size
#     # new_h, new_w = int(scale_factor * h), int(scale_factor * w)
#     new_h, new_w = int(round(scale_factor * h, 0)), int(round(scale_factor * w, 0))
    
#     # Generate random noise
#     Z = torch.randn(b, c, new_h, new_w, device=X.device)
#     # Compute the mean of noise patches
#     Z_mean = Z.unfold(2, 2, 2).unfold(3, 2, 2).mean((4, 5))
#     Z_mean = F.interpolate(Z_mean, size=(new_h, new_w), mode='nearest')
#     current_tensor = F.interpolate(current_tensor, size=(new_h, new_w), mode='nearest')
#     # current_tensor = current_tensor + Z - Z_mean
#     current_tensor = current_tensor / scale_factor + Z - Z_mean
#     # Append the noised tensor to the list
#     noised_tensors.append(current_tensor)

# # Print shapes of the resulting tensors
# for tensor in noised_tensors:
#     print(tensor.shape)

# # plot
# for i, tensor in enumerate(noised_tensors):
#     # # Move the channel dimension to the last dimension for visualization
#     # img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
#     # # Normalize the image for better visualization
#     # img = (img - img.min()) / (img.max() - img.min())
#     # Get dimensions
#     b, c, h, w = tensor.shape
#     dimensions = f'{h}x{w}'
#     # Plot the image
#     plt.figure(figsize=(6, 6))
#     plt.imshow(np.swapaxes(tensor[0],0,-1))
#     plt.title(f'Scale {i} - Dimensions: {dimensions}')
#     plt.axis('off')
#     plt.show()

# %% (delete separation, unite with above)

# original implementation
def upsample_noise(X,N):
    b, c, h, w = X.shape
    Z = torch.randn(b, c, N*h, N*w)
    Z_mean = Z.unfold(2, N, N).unfold(3, N, N).mean((4, 5))
    Z_mean = F.interpolate(Z_mean, scale_factor=N, mode='nearest')
    X = F.interpolate(X, scale_factor=N, mode='nearest')
    return X / N + Z - Z_mean

# # Initialize the base tensor X
# X = ¨^
# # Dictionary to store results
# noise = {}
# # Loop to upsample up to 16384x16384
# size = 128
# while size <= 16384:
#     N = size // 128  # Scale factor
#     X_up = upsample_noise(X, N)
#     noise[size] = X_up
#     # Visualization of the upsampled result
#     plt.figure(figsize=(6, 6))
#     plt.imshow(np.swapaxes(X_up[0], 0, -1))
#     plt.title(f'{X_up.shape[2]}x{X_up.shape[3]}')
#     plt.axis('off')
#     plt.show()
#     size *= 2  # Increase size for the next loop
#
X = torch.randn(1, 4, 1, 1)
N = 2
X_upscaled = upsample_noise(X,N)
#
data = X_upscaled.squeeze(0).reshape(4, -1).cpu().numpy()
# Plot 4 lines in the same plot
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(data[i], label=f'Channel {i+1}', alpha=0.8)
plt.show()


# %% rendering
# MARK: RENDERING
# %matplotlib inline

import torch
from pytorch3d.io import IO
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,          #?
    OpenGLOrthographicCameras,      #?
    FoVOrthographicCameras,         #?
    SfMPerspectiveCameras,          #?
    PerspectiveCameras,             #?
    SfMOrthographicCameras,         #?
    OrthographicCameras,            #?
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    BlendParams,
    HardPhongShader,
    TexturesUV,
    Textures
)
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import os
import numpy as np
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from types import MethodType
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes

def get_pixel_uvs(self, fragments) -> torch.Tensor:
    # Get the UV coordinates per face and interpolate them using the barycentric coords
    faces_verts_uvs = torch.cat([uv[j] for uv, j in zip(self.verts_uvs_list(), self.faces_uvs_list())], dim=0)
    
    # Interpolate to get pixel UVs
    pixel_uvs = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs)
    
    N, H_out, W_out, K = fragments.pix_to_face.shape
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N*K, H_out, W_out, 2)
    pixel_uvs = torch.lerp(
        pixel_uvs.new_tensor([-1.0, 1.0]),
        pixel_uvs.new_tensor([1.0, -1.0]),
        pixel_uvs,
    )    
    
    return pixel_uvs

# Attach the method to the TexturesUV class
TexturesUV.get_pixel_uvs = get_pixel_uvs

# Custom shader class that deactivates the light
class NoLightShader(torch.nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the working directory
os.chdir("/home/thanos/OneDrive/mybackupOneDrive/PhD/3dif_objaverse-xl-subset/testing/")

mesh = IO().load_mesh("main.obj", device=device)
# Define the parameters for different views (all lists have the same length)
d_list = [10, 10]
e_list = [60, 60] # [60, 60]
a_list = [180, 170]
desired_origin = torch.tensor([0.0, 0.0, 0.0]) 
#
# d_list = [3.7, .6]
# e_list = [40, 40]
# a_list = [180, 180]
# desired_origin = torch.tensor([-1.2, -.5, 0.0]) 

# Define the resolutions
# 1024 for rgb (through the encoder) and depth
# 128 for the pixel_uvs_np in the latent space
resolutions = [1024, 128] # [1024, 128]

# Initialize an empty dictionary to store the results
views = {res: {'rgb_view': [], 'fragments': [], 'depth_map_255': [], 'pixel_uvs_np': [], 'd': [], 'e': [], 'a': []} for res in resolutions}

for d, e, a in zip(d_list, e_list, a_list):
    for res in resolutions:
        # Initialize the camera (R:rotation, T:translation)
        R, T = look_at_view_transform(dist=d, elev=e, azim=a)
        T = T + desired_origin.view(1, 3) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(image_size=res, blur_radius=0.0, faces_per_pixel=1) # 1024
        
        # RGB Renderer
        rgb_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=NoLightShader(device=device))
        
        # Depth Renderer
        depth_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=PointLights(device=device)))
    
        # Render the RGB view
        rendered_view = rgb_renderer(mesh)
        
        # Get fragments for depth calculation
        fragments = depth_renderer.rasterizer(mesh)
        depth_map = fragments.zbuf[0, ..., 0].cpu().numpy()
        
        # Normalize the depth map and convert to 8-bit (0-255) scale
        depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map_255 = (depth_map_normalized * 255).astype(np.uint8)
    
        # Get pixel UVs using the provided function
        texture = TexturesUV(
            maps=mesh.textures.maps_padded(),
            faces_uvs=mesh.textures.faces_uvs_padded(),
            verts_uvs=mesh.textures.verts_uvs_padded()
        )
        pixel_uvs = texture.get_pixel_uvs(fragments)
        
        # Convert pixel_uvs to texture space coordinates
        H, W = mesh.textures._maps_padded.shape[1:3]
        pixel_uvs_converted = (pixel_uvs + 1) / 2 * torch.tensor([pixel_uvs.shape[1]-1], device=pixel_uvs.device)
        
        # Convert pixel_uvs_converted to numpy for plotting
        pixel_uvs_np = np.round(pixel_uvs_converted[0].cpu().numpy())
        
        # Append the results to the dictionary
        views[res]['rgb_view'].append(rendered_view)
        views[res]['fragments'].append(fragments)
        views[res]['depth_map_255'].append(depth_map_255)
        views[res]['pixel_uvs_np'].append(pixel_uvs_np)
        views[res]['d'].append(d)
        views[res]['e'].append(e)
        views[res]['a'].append(a)

# Plot each view's RGB view, depth map, and texture UV points
for res in views:
    for i in range(len(views[res]['rgb_view'])):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot RGB view
        ax1.imshow(views[res]['rgb_view'][i][0,:,:,:3].cpu().numpy())
        ax1.set_title(f'RGB View (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        
        # Plot depth map
        ax2.imshow(views[res]['depth_map_255'][i], cmap='gray')
        ax2.set_title(f'Depth Map (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        
        # # Plot texture map with pixel UVs
        # ax3.scatter(views[res]['pixel_uvs_np'][i][:,:,1], views[res]['pixel_uvs_np'][i][:,:,0], c='blue', marker='x', s=5)
        # ax3.set_title(f'Texture Map with UV Points (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        #
        # Plot UV texture map with UV points overlay
        if res == 128:
            uv_texture = mesh.textures.maps_padded()[0]
            uv_texture_resized = F.interpolate(uv_texture[None].permute(0,-1,1,2), size=(128, 128), mode='bilinear', align_corners=False)
            uv_texture_resized_np = uv_texture_resized.squeeze().permute(1, 2, 0).cpu().numpy()  # Shape [128, 128, 3]
            # uv_texture_resized_np = np.ones((uv_texture_resized_np.shape))
            ax3.imshow(uv_texture_resized_np)
        else:
            uv_texture = mesh.textures.maps_padded()[0].cpu().numpy()  # Extract UV texture
            # uv_texture = np.ones((uv_texture.shape))
            ax3.imshow(uv_texture)
        # Overlay pixel UV points on UV texture map
        ax3.scatter(views[res]['pixel_uvs_np'][i][:,:,0], views[res]['pixel_uvs_np'][i][:,:,1], c='blue', marker='x', s=5)
        ax3.set_title(f'UV Texture Map with UV Points (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        
        plt.show()





# %% check overlappings
shared_coords_per_resolution = {}
for res in resolutions:
    flattened_uv = [
        coords.reshape(-1, 2) for coords in views[res]['pixel_uvs_np']
    ]
    uv_sets = [set(map(tuple, coords)) for coords in flattened_uv]
    shared_coords = set.intersection(*uv_sets)
    num_shared_coords = len(shared_coords)
    shared_coords_per_resolution[res] = num_shared_coords
for res, count in shared_coords_per_resolution.items():
    print(f"Number of unique shared coordinates for resolution {res}: {count}")



# Flatten the coordinates
coords_flat = views[128]['pixel_uvs_np'][0].reshape(-1, 2)
# Get unique coordinates and their counts
unique_coords, counts = np.unique(coords_flat, axis=0, return_counts=True)
# Sort by frequency in descending order
sort_idx = np.argsort(-counts)
unique_coords = unique_coords[sort_idx]
counts = counts[sort_idx]
# Print results
print(f"Total number of coordinates: {len(coords_flat)}")
print(f"Number of unique coordinates: {len(unique_coords)}")
print("\nTop coordinates by frequency:")
print("Coordinate (x,y) : Frequency")
print("-" * 30)
for coord, count in zip(unique_coords[:20], counts[:20]):  # Show top 20
    print(f"({coord[0]}, {coord[1]}) : {count}")
if len(unique_coords) > 20:
    print("\n... showing only top 20 coordinates ...")
# Print frequency distribution summary
print("\nFrequency distribution:")
print(f"Max frequency: {counts[0]}")
print(f"Min frequency: {counts[-1]}")
print(f"Mean frequency: {counts.mean():.2f}")
print(f"Median frequency: {np.median(counts):.2f}")


# %% testing polygons


import torch
from pytorch3d.io import IO
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,          #?
    OpenGLOrthographicCameras,      #?
    FoVOrthographicCameras,         #?
    SfMPerspectiveCameras,          #?
    PerspectiveCameras,             #?
    SfMOrthographicCameras,         #?
    OrthographicCameras,            #?
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    BlendParams,
    HardPhongShader,
    TexturesUV,
    Textures
)
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import os
import numpy as np
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from types import MethodType
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes

def get_pixel_uvs(self, fragments) -> torch.Tensor:
    # Get the UV coordinates per face and interpolate them using the barycentric coords
    faces_verts_uvs = torch.cat([uv[j] for uv, j in zip(self.verts_uvs_list(), self.faces_uvs_list())], dim=0)
    
    # Interpolate to get pixel UVs
    pixel_uvs = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs)
    
    N, H_out, W_out, K = fragments.pix_to_face.shape
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N*K, H_out, W_out, 2)
    pixel_uvs = torch.lerp(
        pixel_uvs.new_tensor([-1.0, 1.0]),
        pixel_uvs.new_tensor([1.0, -1.0]),
        pixel_uvs,
    )    
    
    return pixel_uvs






# Attach the method to the TexturesUV class
TexturesUV.get_pixel_uvs = get_pixel_uvs

# Custom shader class that deactivates the light
class NoLightShader(torch.nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        # texels = meshes.sample_textures(fragments)
        texels = texture.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the working directory
os.chdir("/home/thanos/OneDrive/mybackupOneDrive/PhD/3dif_objaverse-xl-subset/testing/")

mesh = IO().load_mesh("main.obj", device=device)

# Define the parameters for different views (all lists have the same length)
d_list = [3.7, .6]
e_list = [40, 40]
a_list = [180, 180]
desired_origin = torch.tensor([-1.2, -.5, 0.0]) 

# Define the resolutions
# 1024 for rgb (through the encoder) and depth
# 128 for the pixel_uvs_np in the latent space
resolutions = [128] # [1024, 128]

views = {res: {'rgb_view': [], 'fragments': [], 'depth_map_255': [], 'pixel_uvs_np': [], 'd': [], 'e': [], 'a': []} for res in resolutions}


for d, e, a in zip(d_list, e_list, a_list):
    for res in resolutions:
        # Initialize the camera (R:rotation, T:translation)
        R, T = look_at_view_transform(dist=d, elev=e, azim=a)
        T = T + desired_origin.view(1, 3) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(image_size=res, blur_radius=0.0, faces_per_pixel=1) # 1024
        
        # RGB Renderer
        rgb_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=NoLightShader(device=device))
        
        # Depth Renderer
        depth_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=PointLights(device=device)))
    
        # Render the RGB view
        rendered_view = rgb_renderer(mesh)
        
        # Get fragments for depth calculation
        fragments = depth_renderer.rasterizer(mesh)
        depth_map = fragments.zbuf[0, ..., 0].cpu().numpy()
        
        # Normalize the depth map and convert to 8-bit (0-255) scale
        depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map_255 = (depth_map_normalized * 255).astype(np.uint8)
    
        # Get pixel UVs using the provided function
        texture = TexturesUV(
            maps=mesh.textures.maps_padded(),
            faces_uvs=mesh.textures.faces_uvs_padded(),
            verts_uvs=mesh.textures.verts_uvs_padded()
        )
        pixel_uvs = texture.get_pixel_uvs(fragments)
        
        # Convert pixel_uvs to texture space coordinates
        H, W = mesh.textures._maps_padded.shape[1:3]
        pixel_uvs_converted = (pixel_uvs + 1) / 2 * torch.tensor([pixel_uvs.shape[1]-1], device=pixel_uvs.device)
        
        # Convert pixel_uvs_converted to numpy for plotting
        pixel_uvs_np = np.round(pixel_uvs_converted[0].cpu().numpy())
        
        # Append the results to the dictionary
        views[res]['rgb_view'].append(rendered_view)
        views[res]['fragments'].append(fragments)
        views[res]['depth_map_255'].append(depth_map_255)
        views[res]['pixel_uvs_np'].append(pixel_uvs_np)
        views[res]['d'].append(d)
        views[res]['e'].append(e)
        views[res]['a'].append(a)

# Plot each view's RGB view, depth map, and texture UV points
for res in views:
    for i in range(len(views[res]['rgb_view'])):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot RGB view
        ax1.imshow(views[res]['rgb_view'][i][0,:,:,:3].cpu().numpy())
        ax1.set_title(f'RGB View (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        
        # Plot depth map
        ax2.imshow(views[res]['depth_map_255'][i], cmap='gray')
        ax2.set_title(f'Depth Map (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        
        # # Plot texture map with pixel UVs
        # ax3.scatter(views[res]['pixel_uvs_np'][i][:,:,1], views[res]['pixel_uvs_np'][i][:,:,0], c='blue', marker='x', s=5)
        # ax3.set_title(f'Texture Map with UV Points (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        #
        # Plot UV texture map with UV points overlay
        if res == 128:
            uv_texture = mesh.textures.maps_padded()[0]
            uv_texture_resized = F.interpolate(uv_texture[None].permute(0,-1,1,2), size=(128, 128), mode='bilinear', align_corners=False)
            uv_texture_resized_np = uv_texture_resized.squeeze().permute(1, 2, 0).cpu().numpy()  # Shape [128, 128, 3]
            # uv_texture_resized_np = np.ones((uv_texture_resized_np.shape))
            ax3.imshow(uv_texture_resized_np)
        else:
            uv_texture = mesh.textures.maps_padded()[0].cpu().numpy()  # Extract UV texture
            # uv_texture = np.ones((uv_texture.shape))
            ax3.imshow(uv_texture)
        # Overlay pixel UV points on UV texture map
        ax3.scatter(views[res]['pixel_uvs_np'][i][:,:,0], views[res]['pixel_uvs_np'][i][:,:,1], c='blue', marker='x', s=5)
        ax3.set_title(f'UV Texture Map with UV Points (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        
        plt.show()


# %%

pixel_uvs_np[0,0]





# %%
from pytorch3d.io import load_obj
verts, faces, aux = load_obj("main.obj")

verts.shape
faces.verts_idx.shape
faces.textures_idx.shape
aux.verts_uvs.shape


faces.textures_idx.shape








# %% plot 3d points
# # (all of them, not just the rendered view) based on the rendered view, not the original world coordinates
# #
# %matplotlib widget
# #
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# #
# # Check if the figure with number 1 exists
# if plt.fignum_exists(1):
#     fig = plt.figure(num=1)
#     ax = fig.gca()
#     ax.cla()  # Clear the existing plot
# else:
#     fig = plt.figure(num=1)
#     ax = fig.add_subplot(111, projection='3d')
# #
# verts_world = mesh.verts_padded()
# # verts_screen = cameras.transform_points_screen(verts_world, image_size=(1024,1024))
# verts_screen = cameras.transform_points(verts_world)
# points = verts_screen.squeeze().cpu().numpy()  # Shape: (3722, 3)
# #
# print(len(points))
# #
# # Plot the points
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color="red")
# #
# # Set labels
# ax.set_xlabel('Y')
# ax.set_ylabel('X')
# ax.set_zlabel('Z')
# #
# plt.show()


# %% tracking (keeping to double check the pixel_uvs tracking)

# # background mask to know if we are on the object or the irrelevant (infinite?) background
# bg = rendered_view[0,:,:,3] == 0

# gradient_map = gradient_mesh.textures._maps_padded[0, ..., 0:2].cpu().numpy()
# gradient_window = rendered_gradient[0, ..., 0:2].cpu().numpy()

# # plt.imshow(gradient_window[...,0])
# # plt.show()
# # plt.imshow(bg.cpu().numpy())

# y = 100
# x = 100
# if bg[y,x]==1:
#     print("Outside of the object.")
# else:
#     target_point = gradient_window[y,x] #1023,1023
#     print(target_point)
#     tolerance = 0.0005 # .0005
    
#     # Assume gradient_map is your array of shape (1024, 1024, 3)
#     mask_u = np.isclose(gradient_map[:, :, 0], target_point[1], atol=tolerance)
#     mask_v = np.isclose(gradient_map[:, :, 1], target_point[0], atol=tolerance)
    
#     # Combine masks to find exact matches for all three channels
#     mask = mask_u & mask_v
    
#     # Find the indices where the condition is True
#     indices = np.where(mask)
    
#     print(indices[0], indices[1])
#     print(gradient_map[indices[0], indices[1]])

# # missing unique coordinates in the view
# # reshaped_map = gradient_window.reshape(-1, 2)
# # unique_pairs = np.unique(reshaped_map, axis=0)
# # print(1024*1024-len(unique_pairs))
# #
# print("Missing unique coordinates in the view:", bg.sum().item())

# plt.figure(figsize=(10, 10))
# plt.imshow(mesh.textures._maps_padded[0].cpu().numpy())
# plt.scatter(indices[0], indices[1], c="black", marker="x")

# ## %% discretized depth

# # # Compute discrete values based on num_upsamples

# # print(np.unique(depth_map_255))

# # if len(np.unique(bg.cpu().numpy())) == 1:
# #     dmin = depth_map_255.min()
# #     dmax = depth_map_255.max()
# #     zones = len(rendered_views)
# # elif len(np.unique(bg.cpu().numpy())) == 2:
# #     dmin = np.unique(depth_map_255)[1]
# #     dmax = depth_map_255.max()
# #     zones = len(rendered_views)+1
# # else:
# #     raise ValueError(f"Unexpected number of unique values: {num_unique_values}. Expected 1 or 2 unique values.")
 
# # bins = len(rendered_views) 
 
# # bin_edges = np.linspace(dmin, dmax, bins)
# # quantized_depth = np.digitize(depth_map_255, bin_edges)
# # discrete_values = np.unique(quantized_depth)

# # print(discrete_values)

# # plt.figure(figsize=(10, 10))
# # plt.imshow(quantized_depth, cmap="gray")
# # # plt.axis('off')
# # plt.show()

# # # Create an empty canvas for the final image
# # final_render = np.zeros((quantized_depth.shape[0], quantized_depth.shape[1], 3), dtype=np.float32)

# # # Replace each zone in the quantized depth map with the corresponding rendered view
# # for i in range(zones-1, -1, -1):
# #     mask = (quantized_depth==discrete_values[i])
# #     c = zones-i-1
# #     if len(np.unique(bg.cpu().numpy())) == 2 and i==0:
# #         c -= 1
# #     print(i, c)
# #     final_render[mask] = rendered_views[c].cpu().numpy()[0,...,:3][mask]


# # # Plot the final rendered image
# # plt.figure(figsize=(10, 10))
# # plt.imshow(final_render)
# # # plt.title('Final Rendered Image with Noised Textures')
# # # plt.axis('off')
# # plt.show()

# # # Plot the final rendered image
# # plt.figure(figsize=(10, 10))
# # plt.imshow(final_render[0:200, 700:900])
# # plt.show()

# # plt.figure(figsize=(10,10));plt.imshow(rendered_views[0][0,:,:,:3].cpu().numpy())

# %% tracking (pixel_uvs) - ONE point

# from pytorch3d.ops import interpolate_face_attributes
# from pytorch3d.renderer import TexturesUV

# def get_pixel_uvs(self, fragments) -> torch.Tensor:
#     # Get the UV coordinates per face and interpolate them using the barycentric coords
#     faces_verts_uvs = torch.cat([uv[j] for uv, j in zip(self.verts_uvs_list(), self.faces_uvs_list())], dim=0)
    
#     # Interpolate to get pixel UVs
#     pixel_uvs = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs)
    
#     N, H_out, W_out, K = fragments.pix_to_face.shape
#     pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N*K, H_out, W_out, 2)
#     pixel_uvs = torch.lerp(
#         pixel_uvs.new_tensor([-1.0, 1.0]),
#         pixel_uvs.new_tensor([1.0, -1.0]),
#         pixel_uvs,
#     )    
    
#     return pixel_uvs

# # Attach the method to the TexturesUV class
# TexturesUV.get_pixel_uvs = get_pixel_uvs

# # Example usage
# texture = TexturesUV(
#     maps=mesh.textures.maps_padded(),
#     faces_uvs=mesh.textures.faces_uvs_padded(),
#     verts_uvs=mesh.textures.verts_uvs_padded()
# )
# pixel_uvs = texture.get_pixel_uvs(fragments)
# # print(pixel_uvs.shape)

# y = 100
# x = 100
# print(pixel_uvs[0,x,y])

# pixel_uvs_converted = (pixel_uvs + 1) / 2 * H
# print(pixel_uvs_converted[0,x,y])


# texels = F.grid_sample(
#     mesh.textures._maps_padded.permute(0,3,1,2), #texture_maps,
#     pixel_uvs,
#     mode="bilinear",
#     align_corners=True,
#     padding_mode="border",
# )
# # texels now has shape (NK, C, H_out, W_out)
# N, H_out, W_out, K = fragments.pix_to_face.shape
# texels = texels.reshape(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)

# # plt.imshow(texels.cpu().numpy()[0,:,:,0,:])
# plt.figure(figsize=(10, 10))
# plt.imshow(mesh.textures._maps_padded[0].cpu().numpy())
# plt.scatter(round(pixel_uvs_converted[0,x,y][0].item()), 
#             round(pixel_uvs_converted[0,x,y][1].item()), 
#             c="black", marker="x")



# %% tracking (pixel_uvs) - ALL points

# def get_pixel_uvs(self, fragments) -> torch.Tensor:
#     # Get the UV coordinates per face and interpolate them using the barycentric coords
#     faces_verts_uvs = torch.cat([uv[j] for uv, j in zip(self.verts_uvs_list(), self.faces_uvs_list())], dim=0)
    
#     # Interpolate to get pixel UVs
#     pixel_uvs = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs)
    
#     N, H_out, W_out, K = fragments.pix_to_face.shape
#     pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N*K, H_out, W_out, 2)
#     pixel_uvs = torch.lerp(
#         pixel_uvs.new_tensor([-1.0, 1.0]),
#         pixel_uvs.new_tensor([1.0, -1.0]),
#         pixel_uvs,
#     )    
    
#     return pixel_uvs

# # Attach the method to the TexturesUV class
# TexturesUV.get_pixel_uvs = get_pixel_uvs

# # Example usage
# texture = TexturesUV(
#     maps=mesh.textures.maps_padded(),
#     faces_uvs=mesh.textures.faces_uvs_padded(),
#     verts_uvs=mesh.textures.verts_uvs_padded()
# )
# pixel_uvs = texture.get_pixel_uvs(fragments)

# # Convert pixel_uvs to texture space coordinates
# H, W = mesh.textures._maps_padded.shape[1:3]
# pixel_uvs_converted = (pixel_uvs + 1) / 2 * torch.tensor([W, H], device=pixel_uvs.device)

# # Sample texels using grid_sample
# texels = F.grid_sample(
#     mesh.textures._maps_padded.permute(0,3,1,2),
#     pixel_uvs,
#     mode="bilinear",
#     align_corners=True,
#     padding_mode="border",
# )

# # Plotting
# plt.figure(figsize=(15, 15))
# plt.imshow(mesh.textures._maps_padded[0].cpu().numpy())

# # Convert pixel_uvs_converted to numpy for plotting
# # pixel_uvs_np = pixel_uvs_converted[0].cpu().numpy()
# pixel_uvs_np = np.round(pixel_uvs_converted[0].cpu().numpy())

# plt.scatter(pixel_uvs_np[:,:,0], pixel_uvs_np[:,:,1], c='blue', marker='x', s=1, alpha=0.5)
# plt.title("Texture Map with All Mapped Points")
# plt.show()


# temp = pixel_uvs_np.reshape(-1,2)
# temp = np.unique(temp, axis=0)
# print(temp.shape[0])
# print(np.sqrt(temp.shape[0]))
# print("# of squares: ", f"{np.sqrt(temp.shape[0])}²",
#       "\nSquare dim: ", f"{1024 / np.sqrt(temp.shape[0])}²")
# # #
# # # plot
# # # plt.imshow(pixel_uvs_np[:,:,1], cmap="gray")
# # # plt.show()
# # # plt.imshow(pixel_uvs_np[:,:,0], cmap="gray")
# # # plt.show()
# # plt.imshow(pixel_uvs_np[:,:,0]+pixel_uvs_np[:,:,1], cmap="gray")
# # plt.show()
# # plt.imshow((pixel_uvs_np[:,:,0]+pixel_uvs_np[:,:,1])[:200,:200], cmap="gray")
# # plt.show()
# # plt.imshow((pixel_uvs_np[:,:,0]+pixel_uvs_np[:,:,1])[-200:,:200], cmap="gray")
# # plt.show()

# # Reshape to (1024*1024, 2) to get a list of coordinate pairs
# reshaped_uvs = pixel_uvs_np.reshape(-1, 2)
# # Find unique (u, v) coordinates and count their occurrences
# _, indices, counts = np.unique(reshaped_uvs, axis=0, return_inverse=True, return_counts=True)
# # Use the indices to map the counts back to the original 1024x1024 structure
# occurrences = counts[indices].reshape(1024, 1024)
# print(occurrences.min(), occurrences.max())















































































          



# %% SDXL-controlnet: Depth
## %%time
# MARK: CtrlNet
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.utils.import_utils import is_invisible_watermark_available

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput


# if is_invisible_watermark_available():
#     from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def jj(ray):
    """Temp to be deleted"""
    # Plot the last two dimensions of the tensor
    if len(ray.shape) == 4:  # Check if the tensor shape is [1, 4, 64, 64]
        plt.imshow(ray[0, 0].cpu())
    elif len(ray.shape) == 5:  # Check if the tensor shape is [1, 2, 4, 64, 64]
        plt.imshow(ray[0, 0, 0].cpu())
    elif len(ray.shape) == 3:  # Check if the tensor shape is [1, 2, 4, 64, 64]
        plt.imshow(ray[0].cpu())
    plt.show()
#
# Monkey patching: Adding the jj method to the Tensor class
torch.Tensor.jj = jj
#
# Now do some_tensor.jj()  

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
        >>> negative_prompt = "low quality, bad quality, sketches"

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
        ... )
up_coords0
        >>> # initialize the models and pipeline
        >>> controlnet_conditioning_scale = 0.5  # recommended for good generalization
        >>> controlnet = ControlNetModel.from_pretrained(
        ...     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
        ... )
        >>> vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> # get canny image
        >>> image = np.array(image)
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)

        >>> # generate image
        >>> image = pipe(
        ...     prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image
        ... ).images[0]
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionXLControlNetPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionXLLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        text_encoder_2 ([`~transformers.CLIPTextModelWithProjection`]):
            Second frozen text-encoder
            ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        tokenizer_2 ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings should always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark](https://github.com/ShieldMnt/invisible-watermark/) library to
            watermark output images. If not defined, it defaults to `True` if the package is installed; otherwise no
            watermarker is used.
    """

    # leave controlnet out on purpose because it iterates with unet
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "feature_extractor",
        "image_encoder",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        feature_extractor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
    ):
        super().__init__()

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        image,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        negative_pooled_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(image, prompt, prompt_embeds)
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(image, list):
                raise TypeError("For multiple controlnets: `image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in image):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif len(image) != len(self.controlnet.nets):
                raise ValueError(
                    f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(image)} images and {len(self.controlnet.nets)} ControlNets."
                )

            for image_ in image:
                self.check_image(image_, prompt, prompt_embeds)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        if isinstance(self.controlnet, MultiControlNetModel):
            if len(control_guidance_start) != len(self.controlnet.nets):
                raise ValueError(
                    f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.check_image
    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def denoising_end(self):
        return self._denoising_end

    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    def scale_coordinates(self, array, old_min=0, old_max=127, new_min=0, new_max=16383):
        scaled_array = ((array - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min
        return np.round(scaled_array).astype(np.int64)

    def upsample_noise(self, X, N):
        b, c, h, w = X.shape
        Z = torch.randn(b, c, N*h, N*w).to(X.device)
        Z_mean = Z.unfold(2, N, N).unfold(3, N, N).mean((4, 5))
        Z_mean = F.interpolate(Z_mean, scale_factor=N, mode='nearest')
        X = F.interpolate(X, scale_factor=N, mode='nearest')
        return X / N + Z - Z_mean

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        inp: dict = None,
        # noise: dict = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted
                as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or
                width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`,
                images must be passed as a list such that each element of the list can be correctly batched for input
                to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. This is sent to `tokenizer_2`
                and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, pooled text embeddings are generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
                weighting). If not provided, pooled `negative_prompt_embeds` are generated from `negative_prompt` input
                argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned containing the output images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            negative_pooled_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3.1 Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        # 3.2 Encode ip_adapter_image
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
# MARK: HERE
        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            # image = self.prepare_image(
            #     image=image,
            #     width=width,
            #     height=height,
            #     batch_size=batch_size * num_images_per_prompt,
            #     num_images_per_prompt=num_images_per_prompt,
            #     device=device,
            #     dtype=controlnet.dtype,
            #     do_classifier_free_guidance=self.do_classifier_free_guidance,
            #     guess_mode=guess_mode,
            # )
            # height, width = image.shape[-2:]
            #
            image = [self.prepare_image(
                image=PIL.Image.fromarray(depth_map),  # Use each depth map from the list
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
                ) 
                for depth_map in inp[1024]["depth_map_255"]  # List comprehension to iterate over depth maps
                ]
            height, width = image[0].shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # latents = [
        #     self.prepare_latents(
        #         batch_size * num_images_per_prompt,
        #         num_channels_latents,
        #         height,
        #         width,
        #         prompt_embeds.dtype,
        #         device,
        #         generator,
        #         v.to(torch.float16),
        #     )
        #     for v in inp[1024]["rgb_view"]
        #     ]
        
        noise = [
            torch.randn([1, 4, 128, 128], dtype=torch.float16).to("cuda")
            for _ in inp[128]["rgb_view"]
            ]
        # noise[1] = noise[0]
        
        # # unique coords
        # temp = inp[128]["pixel_uvs_np"][0]
        # reshaped_temp = temp.reshape(-1, 2)
        # unique_coords = np.unique(reshaped_temp, axis=0)
        # unique_coords = unique_coords.shape[0]
        # print(f"Unique coords: {unique_coords} out of {reshaped_temp.shape[0]}")

        # reshaped_coords_tuples = [tuple(coord) for coord in reshaped_coords]
        # counts = Counter(reshaped_coords_tuples)
        # unique_coords = list(counts.keys())
        # usage_counts = list(counts.values())
        # plt.figure(figsize=(12, 6))
        # plt.bar(range(len(unique_coords)), usage_counts, color='blue')
        # num_ticks_to_show = min(20, len(unique_coords))  # Show at most 15 ticks
        # indices = np.linspace(0, len(unique_coords) - 1, num_ticks_to_show).astype(int)  # Indices to show
        # selected_labels = [f"{unique_coords[i]}" for i in indices]
        # plt.xticks(ticks=indices, labels=selected_labels, rotation=90, fontsize=8)
        # plt.title('Coordinate Usage Count')
        # plt.xlabel('Unique Coordinates (x, y)')
        # plt.ylabel('Usage Count')
        # plt.grid(True, axis='y')
        # plt.tight_layout()
        # plt.show()






        #
        latents = []
        for v, n in zip(inp[1024]["rgb_view"], noise):
            im = v[...,:3].cpu().numpy().astype(np.float16).transpose(0, 3, 1, 2)  # Adjust shape
            im = torch.from_numpy(im).to("cuda")  # Convert to tensor and move to GPU
            im = 2. * im - 1.  # Scale to [-1, 1]
            
            with torch.no_grad():
                latent = vae.encode(im)  # Encode using VAE
                latent = vae.config.scaling_factor * latent.latent_dist.sample()  # Scale the latent representation
        
            # Add noise to the latent
            latents.append(self.scheduler.add_noise(latent, n, timesteps[0:1]))

        # 
        # latents = [
        #     self.scheduler.add_noise(v, n, timesteps[0:1])
        #     for v, n in zip(inp[128]["rgb_view"], noise)
        #     ]
        # latents = [latent.permute(0, 3, 1, 2).to(torch.float16) for latent in latents]
        # #
        # # noise = torch.randn(inp.shape, dtype=torch.float16).to("cuda")
        # # latents = self.scheduler.add_noise(inp, noise, timesteps[0:1])

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        #
        # initialize track_uv 0's for latents and 1's for frequency
        track_uv = torch.zeros((1, 5, 128, 128), dtype=torch.float16).to("cuda")
        track_uv[:, :4, :, :] = -1
        #
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                temp_denoised = []
                temp_uv = []
                for ind, (ltnt, depth, uv) in enumerate(zip(latents, image, inp[128]["pixel_uvs_np"])):
                    # Relevant thread:
                    # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                    if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                        torch._inductor.cudagraph_mark_step_begin()
                    # expand the ltnt if we are doing classifier free guidance
                    latent_model_input = torch.cat([ltnt] * 2) if self.do_classifier_free_guidance else ltnt
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    
                    # controlnet(s) inference
                    if guess_mode and self.do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = ltnt
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                        controlnet_added_cond_kwargs = {
                            "text_embeds": add_text_embeds.chunk(2)[1],
                            "time_ids": add_time_ids.chunk(2)[1],
                        }
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds
                        controlnet_added_cond_kwargs = added_cond_kwargs
    
                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]
    
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=depth, #image
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        added_cond_kwargs=controlnet_added_cond_kwargs,
                        return_dict=False,
                    )
    
                    if guess_mode and self.do_classifier_free_guidance:
                        # Inferred ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
    
                    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                        added_cond_kwargs["image_embeds"] = image_embeds
    
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
    
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
                    # compute the previous noisy sample x_t -> x_t-1
                    ltnt = self.scheduler.step(noise_pred, t, ltnt, **extra_step_kwargs, return_dict=False)[0]
    
                    latents[ind] = ltnt


                    # if callback_on_step_end is not None:
                    #     callback_kwargs = {}
                    #     for k in callback_on_step_end_tensor_inputs:
                    #         callback_kwargs[k] = locals()[k]
                    #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
    
                    #     latents = callback_outputs.pop("latents", latents)
                    #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    #     add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    #     negative_pooled_prompt_embeds = callback_outputs.pop(
                    #         "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    #     )
                    #     add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    #     negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)
    
                    # # call the callback, if provided
                
                

                # # if i == 20:
                # # Initialize track_uv tensor with an extra channel for frequency
                # track_uv = torch.zeros((128, 128, 5), dtype=torch.float32).cuda()  # 4 channels for latents + 1 for frequency

                # # Iterate through the coordinates and update the sum and frequency tensors
                # for latent, uv_coords in zip(latents, inp[128]["pixel_uvs_np"]):
                #     for i in range(128):
                #         for j in range(128):
                #             x, y = torch.from_numpy(uv_coords[i, j])
                #             x, y = int(x), int(y)
                #             track_uv[x, y, :4] += latent[0, :, i, j]  # Sum the latent values
                #             track_uv[x, y, 4] += 1  # Increment the frequency

                # # Compute the average by dividing the sum tensor by the frequency tensor
                # # Avoid division by zero by using torch.where
                # frequency = track_uv[:, :, 4]
                # average_latents = torch.where(frequency.unsqueeze(-1) > 0, track_uv[:, :, :4] / frequency.unsqueeze(-1), track_uv[:, :, :4])

                # # Map the averaged values back to the latents list
                # for idx, (latent, uv_coords) in enumerate(zip(latents, inp[128]["pixel_uvs_np"])):
                #     # uv_coords = torch.from_numpy(uv_coords).to(device)  # Move uv_coords to the same device
                #     for i in range(128):
                #         for j in range(128):
                #             x, y = uv_coords[i, j]
                #             x, y = int(x), int(y)  # Ensure x and y are integers
                #             latents[idx][0, :, i, j] = average_latents[x, y]  # Update the latent values with the averaged values



                
                
                # pixel_uvs = torch.tensor(inp[128]["pixel_uvs_np"][1])  # Convert to a tensor
                # latents_0 = latents[0]  # Shape: (1, 4, 128, 128)

                # # Flatten the pixel UVs
                # H, W = pixel_uvs.shape[:2]
                # pixel_uvs_flat = pixel_uvs.reshape(-1, 2)  # Shape: (128*128, 2)

                # # Create an empty tensor to hold the averaged latents with the same shape as latents_0
                # averaged_latents = torch.zeros_like(latents_0)  # Shape: (1, 4, 128, 128)

                # # Count occurrences of each pixel UV
                # count = torch.zeros(H, W, device=latents_0.device)

                # # Iterate over each coordinate in the pixel UVs
                # for i in range(H):
                #     for j in range(W):
                #         uv = pixel_uvs[i, j].round().long()  # Round and convert to integer for indexing
                        
                #         if 0 <= uv[0] < H and 0 <= uv[1] < W:
                #             # Accumulate the values for averaging
                #             averaged_latents[0, :, i, j] += latents_0[0, :, uv[0], uv[1]]
                #             count[uv[0], uv[1]] += 1  # Count occurrences

                # # Avoid division by zero by using a mask
                # count[count == 0] = 1  # Prevent division by zero
                # averaged_latents /= count.view(1, 1, H, W)  # Broadcast the count to match the shape of averaged_latents

                # latents[0] = averaged_latents.clone()

                up_dim = 4096 # 16384, 4096

                # Initialize track_uv tensor with an extra channel for frequency (higher scale for uv space)
                # track_uv = torch.zeros((up_dim, up_dim, 5), dtype=torch.float32).cuda()  # 4 channels for latents + 1 for frequency

                
                scale = up_dim//128
                #
                up_l0 = self.upsample_noise(latents[0], scale)
                up_l1 = self.upsample_noise(latents[1], scale)
                #
                coords0 = inp[128]["pixel_uvs_np"][0]
                coords1 = inp[128]["pixel_uvs_np"][1]
                up_coords0 = self.scale_coordinates(coords0, 0, 128, 0, up_dim) # hmmm 128 or 127???
                up_coords1 = self.scale_coordinates(coords1, 0, 128, 0, up_dim)
                up_coords0 = np.repeat(np.repeat(up_coords0, scale, axis=0), scale, axis=1)
                up_coords1 = np.repeat(np.repeat(up_coords1, scale, axis=0), scale, axis=1)
                #
                # make coords unique
                
                



# # Flatten the coordinates
# coords_flat = up_coords0.reshape(-1, 2)
# # Get unique coordinates and their counts
# unique_coords, counts = np.unique(coords_flat, axis=0, return_counts=True)
# # Sort by frequency in descending order
# sort_idx = np.argsort(-counts)
# unique_coords = unique_coords[sort_idx]
# counts = counts[sort_idx]
# # Print results
# print(f"Total number of coordinates: {len(coords_flat)}")
# print(f"Number of unique coordinates: {len(unique_coords)}")
# print("\nTop coordinates by frequency:")
# print("Coordinate (x,y) : Frequency")
# print("-" * 30)
# for coord, count in zip(unique_coords[:20], counts[:20]):  # Show top 20
#     print(f"({coord[0]}, {coord[1]}) : {count}")
# if len(unique_coords) > 20:
#     print("\n... showing only top 20 coordinates ...")
# # Print frequency distribution summary
# print("\nFrequency distribution:")
# print(f"Max frequency: {counts[0]}")
# print(f"Min frequency: {counts[-1]}")
# print(f"Mean frequency: {counts.mean():.2f}")
# print(f"Median frequency: {np.median(counts):.2f}")



# self.scale_coordinates(np.array([[46],[111]]), 0,128,0,up_dim)




# import numpy as np

# # Flatten the coordinates
# coords_flat = up_coords0.reshape(-1, 2)

# # Calculate base coordinates (start of each 32x32 grid)
# base_coords = (coords_flat // 32) * 32

# # Create a unique identifier for each coordinate pair
# coord_ids = coords_flat[:, 0] * 4096 + coords_flat[:, 1]  # Assuming max coordinate is 4096

# # Get unique coordinates and their order
# _, inverse_indices, counts = np.unique(coord_ids, 
#                                       return_inverse=True,
#                                       return_counts=True)

# # Create an array that will store the occurrence number for each coordinate
# occurrence_counts = np.zeros_like(coord_ids, dtype=np.int32)

# # Use argsort to get the original order
# sort_idx = np.argsort(inverse_indices, kind='stable')
# group_idx = inverse_indices[sort_idx]

# # Count occurrences within each group
# occurrence_idx = np.arange(len(coords_flat))
# occurrence_counts[sort_idx] = occurrence_idx - np.concatenate(([0], np.cumsum(counts)[:-1]))[group_idx]

# # Calculate offsets within each 32x32 grid
# offset_x = np.minimum(occurrence_counts % 32, 31)
# offset_y = np.minimum(occurrence_counts // 32, 31)

# # Combine the offsets
# offsets = np.column_stack((offset_x, offset_y))

# # Calculate final coordinates
# result = base_coords + offsets

# # Reshape back to original shape
# result = result.reshape(up_coords0.shape)














# plt.imshow(result[:,:,0]), plt.show()
# plt.imshow(result[2500:2600,2500:2600,0]), plt.show()

# result[2500:2600,2500:2600,0]
# up_coords0[2500:2600,2500:2600,0]



# # unique coords
# temp = result
# reshaped_temp = temp.reshape(-1, 2)
# unique_coords = np.unique(reshaped_temp, axis=0)
# unique_coords = unique_coords.shape[0]
# print(f"Unique coords: {unique_coords} out of {reshaped_temp.shape[0]}")



                
                































                
                
                # # up-down testing
                # temp = self.upsample_noise(latents[0], 2)
                # latents[0] = torch.nn.functional.avg_pool2d(temp, kernel_size=2, stride=2).to(torch.float16) * 2
                



                


        # if not output_type == "latent":
        #     # make sure the VAE is in float32 mode, as it overflows in float16
        #     needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        #     if needs_upcasting:
        #         self.upcast_vae()
        #         latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        #     # unscale/denormalize the latents
        #     # denormalize with the mean and std if available and not None
        #     has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        #     has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        #     if has_latents_mean and has_latents_std:
        #         latents_mean = (
        #             torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        #         )
        #         latents_std = (
        #             torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        #         )
        #         latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        #     else:
        #         latents = latents / self.vae.config.scaling_factor

        #     image = self.vae.decode(latents, return_dict=False)[0]

        #     # cast back to fp16 if needed
        #     if needs_upcasting:
        #         self.vae.to(dtype=torch.float16)
        # else:
        #     image = latents

        # if not output_type == "latent":
        #     # apply watermark if available
        #     if self.watermark is not None:
        #         image = self.watermark.apply_watermark(image)

        #     image = self.image_processor.postprocess(image, output_type=output_type)

        # # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image,)

        # return StableDiffusionXLPipelineOutput(images=image)
        #
        #
        #
        #
        #
        #
        #
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = [latent.to(next(iter(self.vae.post_quant_conv.parameters())).dtype) for latent in latents]

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None

            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents[0].device, latents[0].dtype)
                latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents[0].device, latents[0].dtype)
                latents = [(latent * latents_std / self.vae.config.scaling_factor + latents_mean) for latent in latents]
            else:
                latents = [latent / self.vae.config.scaling_factor for latent in latents]

            # Decode each latent in the list
            images = [self.vae.decode(latent, return_dict=False)[0] for latent in latents]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            images = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                images = [self.watermark.apply_watermark(image) for image in images]

            images = [self.image_processor.postprocess(image, output_type=output_type) for image in images]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)

        return StableDiffusionXLPipelineOutput(images=images)



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# ---- Main

import torch
import numpy as np
from PIL import Image

from diffusers import ControlNetModel, AutoencoderKL,  DDIMScheduler
from diffusers.utils import load_image

try:
    controlnet
except:
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    # variant="fp16",
    # use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, set_alpha_to_one=True) # ToDo: Fix this???


prompt = "photorealistic bedroom, award winning, cgi, 3d rendering"
# prompt = "photorealistic drone, mq-9_reaper, award winning, cgi, 3d rendering"
# image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
controlnet_conditioning_scale = 0.5  # recommended for good generalization

rnd = torch.randint(0, 10000**2, size=(1,)).item()
rnd = 56582723
print(rnd)
generator = torch.manual_seed(rnd)

# im = rendered_view[0,...,:3].cpu().numpy().astype(np.float16)[None].transpose(0, 3, 1, 2)
# im = torch.from_numpy(im)
# im = 2.*im - 1.
# with torch.no_grad():
#     latent = vae.encode(im.to("cuda"))
#     im = vae.config.scaling_factor * latent.latent_dist.sample()
#
# processed_uv_list = []
# for uv in pixel_uvs.cpu().numpy():
#     uv = np.concatenate([uv, np.zeros((*uv.shape[:2], 1))], axis=-1)
#     im = uv.astype(np.float16)[None].transpose(0, 3, 1, 2)
#     im = torch.from_numpy(im)
#     im = 2. * im - 1.
#     with torch.no_grad():
#         latent = vae.encode(im.to("cuda"))
#         im = vae.config.scaling_factor * latent.latent_dist.sample()
#     processed_uv_list.append(im)



dm = Image.fromarray(depth_map_255)
images = pipe(
    prompt,
    image=dm, # to be removed
    num_inference_steps=30, # 50
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    generator = generator,
    # inp = im, # to be removed
    inp = views,
    # noise = noise,
).images

# Number of images
num_images = len(images)
# Create subplots with the number of images
fig, axs = plt.subplots(1, num_images, figsize=(10 * num_images, 10))
# If only one image, matplotlib returns a single Axes object, handle this case
if num_images == 1:
    axs = [axs]
# Plot each image in a subplot
for i, image in enumerate(images):
    if isinstance(image, list):
        image = image[0]  # Unwrap the list if image is inside a list
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Convert tensor to numpy array
    # If the image is a tensor or numpy array, squeeze if needed
    if isinstance(image, (np.ndarray, torch.Tensor)):
        image = image.squeeze()  # Remove any singleton dimensions if it's a tensor or numpy array
    # Plot the image
    axs[i].imshow(image)
    axs[i].axis("off")  # Hide axes
    axs[i].set_title(f"controlnet {i+1}")  # Add a title
# Show the plot
plt.tight_layout()
plt.show()


# %% changing the original UV map (keep it! important!)

# # get the original texture map
# texture_map = mesh.textures.maps_padded()
# # do whatever with it. Making it all black
# new_map = torch.zeros_like(texture_map)
# # you'll need to instantiate  a new TexturesUV. Import the class
# from pytorch3d.renderer.mesh.textures import TexturesUV
# # and instantiate it with the old faces and verts uv's
# new_texture = TexturesUV(maps=new_map, 
#                          faces_uvs=mesh.textures.faces_uvs_padded(), 
#                          verts_uvs=mesh.textures.verts_uvs_padded())
# # assign it to your meshes object
# mesh.textures = new_texture


# %%


