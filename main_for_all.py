import os
os.chdir("/home/thanos/OneDrive/mybackupOneDrive/PhD/3Dif/")
import util
os.chdir("/home/thanos/OneDrive/mybackupOneDrive/PhD/3dif_objaverse-xl-subset/Minecraft Grass Block/")


import numpy as np
import torch
import imageio
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
from matplotlib.collections import LineCollection


def rotate_z(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, s, 0, 0],
                     [-s, c, 0, 0],
                     [ 0, 0, 1, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)
                     

def compare_views(cosine_angles1, cosine_angles2):
    """Compare two views based on cosine angles"""
    # Create mask for valid pixels (where either view has a value)
    valid_mask = (cosine_angles1 != 0) | (cosine_angles2 != 0)
    # Initialize comparison with zeros (background)
    comparison = torch.zeros_like(cosine_angles1)
    # Only compare where we have valid pixels
    better_view1 = cosine_angles1 > cosine_angles2
    comparison[valid_mask & better_view1] = 1  # View 1 is better
    comparison[valid_mask & ~better_view1] = 2  # View 2 is better
    # Leave as 0 where neither view has content
    return comparison

def load_obj(filename, normalization=False):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices  = []
    texcoords = []
    normals = []

    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.split()[0] == 'vt':
            texcoords.append([float(v) for v in line.split()[1:3]])
        elif line.split()[0] == 'vn':
            normals.append([float(v) for v in line.split()[1:4]])

    vertices  = np.array(vertices)
    texcoords = np.array(texcoords)
    normals = np.array(normals)

    # load faces
    faces = []
    tfaces = []
    nfaces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0])
            t0 = int(vv[1]) if len(vv) > 1 else -1
            n0 = int(vv[2]) if len(vv) > 2 else -1
            for i in range(nv - 2):
                vv = vs[i + 1].split('/')
                v1 = int(vv[0])
                t1 = int(vv[1]) if len(vv) > 1 else -1
                n1 = int(vv[2]) if len(vv) > 2 else -1
                vv = vs[i + 2].split('/')
                v2 = int(vv[0])
                t2 = int(vv[1]) if len(vv) > 1 else -1
                n2 = int(vv[2]) if len(vv) > 2 else -1
                faces.append((v0, v1, v2))
                tfaces.append((t0, t1, t2))
                nfaces.append((n0, n1, n2))
    faces = np.array(faces, dtype=np.uint32)-1  # 1-based index -> 0-based
    tfaces = np.array(tfaces, dtype=np.uint32)-1
    nfaces = np.array(nfaces, dtype=np.int32)-1
    return vertices, normals, texcoords, faces, nfaces, tfaces

def get_pixels_in_quad(glctx, quad_coords, resolution):
    """
    Get pixel coordinates inside a 2D quad

    Args:
        glctx: nvdiffrast context
        quad_coords: tensor of shape (4, 2) containing y,x coordinates in pixel space
        resolution: tuple of (height, width) for output resolution

    Returns:
        pixel_coords: Nx2 tensor of (y,x) pixel coordinates inside the quad
    """
    
    if (quad_coords != 0.).any():
        # To be deleted, Thanos
        pass
    
    # Convert to NDC space
    y = (2.0 * quad_coords[:, 0] / resolution[0]) - 1.0
    x = (2.0 * quad_coords[:, 1] / resolution[1]) - 1.0
    quad_coords_ndc = torch.stack([x, y], dim=1)

    # Setup vertices with homogeneous coordinates
    vertices = torch.zeros((4, 4), dtype=torch.float32, device='cuda')
    vertices[:, :2] = quad_coords_ndc
    vertices[:, 2] = 0.0  # z coordinate
    vertices[:, 3] = 1.0  # w coordinate

    # Define triangles
    triangles = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.int32, device='cuda')

    # Add batch dimension
    vertices = vertices[None, ...]

    # Perform rasterization
    rast_out, _ = dr.rasterize(glctx, vertices, triangles, resolution=resolution)

    # Get mask of pixels inside quad
    mask = rast_out[0, :, :, 3] > 0

    # Get coordinates of pixels where mask is True
    x_coords, y_coords = torch.where(mask)

    # Stack into Nx2 tensor of pixel coordinates
    pixel_coords = torch.stack([y_coords, x_coords], dim=1)

    return pixel_coords, mask


def visualize_pixels(resolution, quad_coords, pixel_coords, mask):
    """Visualize the quad and the pixels inside it"""
    plt.figure(figsize=(10, 5))

    # Plot 1: Show the mask
    plt.subplot(1, 2, 1)
    plt.imshow(mask.cpu().numpy(), cmap='gray')
    plt.grid(visible=True, color='white', linestyle='--', linewidth=0.5, alpha=1)
    # Plot quad outline with flipped x and y coordinates
    plt.plot(quad_coords.cpu()[:, 1], quad_coords.cpu()[:, 0], 'r-')
    plt.plot([quad_coords[-1, 1].cpu(), quad_coords[0, 1].cpu()],
             [quad_coords[-1, 0].cpu(), quad_coords[0, 0].cpu()], 'r-')
    plt.title('Rasterized Quad Mask')
    
    plt.ylim(quad_coords[:,0].max().item()+10, quad_coords[:,0].min().item()-10) # Flip Y axis to match image coordinates  
    plt.xlim(quad_coords[:,1].min().item()-10, quad_coords[:,1].max().item()+10)                    
    

    # Plot 2: Show the pixel coordinates
    plt.subplot(1, 2, 2)
    plt.scatter(pixel_coords.cpu()[:, 0], pixel_coords.cpu()[:, 1], s=1)
    # Plot quad outline with flipped x and y coordinates
    plt.plot(quad_coords.cpu()[:, 1], quad_coords.cpu()[:, 0], 'r-')
    plt.plot([quad_coords[-1, 1].cpu(), quad_coords[0, 1].cpu()],
             [quad_coords[-1, 0].cpu(), quad_coords[0, 0].cpu()], 'r-')
    # plt.xlim(0, resolution[1])
    # plt.ylim(resolution[0], 0)  # Flip Y axis to match image coordinates
    plt.title('Pixel Coordinates')
    
    plt.ylim(quad_coords[:,0].max().item()+10, quad_coords[:,0].min().item()-10) # Flip Y axis to match image coordinates  
    plt.xlim(quad_coords[:,1].min().item()-10, quad_coords[:,1].max().item()+10)                    
    
    plt.tight_layout()
    plt.show()


#----------------------------------------------------------------------------

def img_spot():
    
    resolutions = [128, 1024]
    #
    # resolutions = [16]

    # crop_slice = np.s_[5:5+220, 25:25+220, :3] # to save the images

    pos, nrm, uv, tri_pos, tri_nrm, tri_uv = load_obj("untitled.obj")   
    pos = np.concatenate([pos, np.ones_like(pos[:, :1])], axis=1)
    nrm = np.concatenate([nrm, np.zeros_like(nrm[:, :1])], axis=1)
 
    
    tex = imageio.v2.imread("default.png")[::-1].astype(np.float32) / 255.0
    tex = torch.as_tensor(tex, dtype=torch.float32, device='cuda')

    pos     = torch.as_tensor(pos,     dtype=torch.float32, device='cuda')
    uv      = torch.as_tensor(uv,      dtype=torch.float32, device='cuda')
    tri_pos = torch.as_tensor(tri_pos.astype(np.int32), dtype=torch.int32,   device='cuda')
    tri_uv  = torch.as_tensor(tri_uv.astype(np.int32),  dtype=torch.int32,   device='cuda')
    nrm = torch.as_tensor(nrm, dtype=torch.float32, device='cuda')
    tri_nrm = torch.as_tensor(tri_nrm.astype(np.int32), dtype=torch.int32, device='cuda')

    # initialize a CUDA context for rasterization
    glctx = dr.RasterizeCudaContext()

    
    # cube
    # Modelview and projection matrices
    # here this is for 2 views, uncommend the third or add/remove for more/less
    proj_params = [
        # x, n, f
        (.1, 1.0, 60.0),
        (.1, 1.0, 60.0),
        # (.1, 1.0, 60.0),
        ]
    translations = [
        # x, y, z
        (0.2, 1., -15.0),
        (0.1, 1., -15.0),
        # (0.2, 1., -20.0),
        ]
    rotations = [
        #z, y, x
        (np.pi, np.pi, np.pi*2),
        (np.pi, np.pi, np.pi*2), # (np.pi, np.pi, np.pi*2.1),
        # (np.pi*1.1, np.pi*1.03, np.pi*2),
        ]
    #
    # # # # cube, orthographic (orthographic should be set to True in util.projection)
    # proj_params = [
    #     # x, n, f
    #     (2.1, 1.0, 200.0),
    #     (2.1, 1.0, 200.0),
    #     # (2.1, 1.0, 200.0),
    #     ]
    # translations = [
    #     # x, y, z
    #     (0., -1., 55.0),
    #     (0.1, -1., 55.0),
    #     # (0., -1., 55.0),
    #     ]
    # rotations = [
    #     #z, y, x
    #     (np.pi, np.pi, np.pi),
    #     (np.pi, np.pi, np.pi), # (np.pi, np.pi*1.2, np.pi*1.1),
    #     # (np.pi, np.pi, np.pi), # (np.pi, np.pi*1.5, np.pi*1.1),
    #     ]

    results = {}
    
    for resolution in resolutions:
        results[resolution] = {
            'view': [],
            'depth': [],
            'params': [],
            'footprints': [],
            'map': [], # rename to mapping
            'normal': []
        }
            
    # loop for all rendered views
    for idx, resolution in enumerate(resolutions):
        resolution = [resolution, resolution]
        print(f"\nProcessing resolution: {resolution}")
        
        
        # Loop through parameter sets
        for param_idx in range(len(proj_params)):
        
            # Get parameters for this iteration
            x, n, f = proj_params[param_idx]
            tx, ty, tz = translations[param_idx]
            rz, ry, rx = rotations[param_idx]
            
            print(f"Rendering view: {param_idx + 1}")
            
            # Create projection matrix
            proj = util.projection(x=x, n=n, f=f, orthographic=False)
            r_mv = util.translate(tx, ty, tz)
            r_mv = np.matmul(r_mv, rotate_z(rz))
            r_mv = np.matmul(r_mv, util.rotate_y(ry))
            r_mv = np.matmul(r_mv, util.rotate_x(rx))
            r_mvp = np.matmul(proj, r_mv).astype(np.float32)
            r_mvp = torch.as_tensor(r_mvp, dtype=torch.float32, device='cuda')
            
            # Extract rotation part of model-view matrix for normal transformation
            mv_3x3 = r_mv[:3, :3]
            normal_matrix = torch.from_numpy(mv_3x3).cuda()
            transformed_normals = torch.matmul(normal_matrix, nrm[:, :3].t()).t()
            transformed_normals = torch.nn.functional.normalize(transformed_normals, dim=1)
            transformed_normals = torch.cat([transformed_normals, torch.zeros_like(transformed_normals[:, :1])], dim=1)
    
            cpos = torch.matmul(r_mvp, pos.t()).t()[None, ...]
            post_proj_w = cpos[...,3:4]
    
            rast_out, rast_out_db = dr.rasterize(glctx, cpos.contiguous(), tri_pos, resolution=resolution)
            # plot barycentric coordinates
            rgb_image = np.zeros((resolution[0], resolution[1], 3)) 
            u = rast_out[0, :, :, 0]
            v = rast_out[0, :, :, 1]
            rgb_image[:, :, 0] = u.detach().cpu().numpy() # Red channel
            rgb_image[:, :, 1] = v.detach().cpu().numpy()  # Green channel
            # plt.imshow(rgb_image)
            # plt.title('Barycentric coordinates')
            # plt.show()
            # normalized depth
            # plt.imshow(rast_out[0,:,:,2].detach().cpu().numpy())
            # plt.title('Normalized depth')
            # plt.colorbar()
            # plt.show()
            # # triangle_id
            # plt.imshow(rast_out[0,:,:,3].detach().cpu().numpy())
            # plt.title('Triangle ID')
            # plt.show()
            
            
            # (interpolate post_projection w)
            attr_out_w, attr_out_da_w = dr.interpolate(post_proj_w, rast_out, tri_uv, rast_db=rast_out_db, diff_attrs='all')
            
            # Interpolate transformed normals
            normal_out, _ = dr.interpolate(transformed_normals[None, ...], rast_out, tri_nrm, rast_db=rast_out_db)
            # Apply mask and normalize
            mask = rast_out[..., 3:] > 0
            normal_map = torch.where(mask, normal_out, torch.zeros_like(normal_out))
            normal_map = torch.nn.functional.normalize(normal_map, dim=-1)
            #
            view_dirs = torch.zeros_like(normal_map)
            view_dirs[..., 2] = 1.0
            view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1) #Thanos: remove this???
            cosine_angles = torch.sum(normal_map * view_dirs, dim=-1)
                        
            attr_out, attr_out_da = dr.interpolate(uv[None, ...], rast_out, tri_uv, rast_db=rast_out_db, diff_attrs='all')
            # attr_out: interpolated attributes
            # image-space derivatives of the selected attributes w/ shape ...,2*len(attr_out). So, channels 0 & 1 -> dA/dX, dA/dY for attribute A
            # plot attribute (here texture coordinates)
            rgb_image = np.zeros((resolution[0], resolution[1], 3)) 
            u = attr_out[0, :, :, 0]
            v = attr_out[0, :, :, 1]
            rgb_image[:, :, 0] = u.detach().cpu().numpy()  # Red channel
            rgb_image[:, :, 1] = v.detach().cpu().numpy()  # Green channel
            # plt.imshow(rgb_image)
            # plt.title('Texture coordinates (s, t)')
            # plt.show()
            
            tex_out = dr.texture(tex[None, ...], attr_out, attr_out_da, filter_mode='linear-mipmap-linear')
            # to remove the background:
            texw_out = torch.where(rast_out[..., 3:] > 0, tex_out, torch.tensor(1.0).cuda())
    
            plt.imshow(texw_out[0].detach().cpu().numpy())
            plt.title(f'Texture sampling ({resolution[0]}x{resolution[0]})')
            plt.show()

        
            # Calculate footprints
            print("(Calculating footprints...)")
                    
            u_center = attr_out[0, :, :, 0]
            v_center = attr_out[0, :, :, 1]
            w_center = attr_out_w[0, :, :, 0]
            
            valid_mask = w_center != 0
            
            # Initialize arrays for perspective-correct coordinates
            u_over_w = torch.zeros_like(u_center)
            v_over_w = torch.zeros_like(v_center)
            one_over_w = torch.zeros_like(w_center)
            
            # Calculate perspective-correct coordinates where w != 0
            u_over_w[valid_mask] = u_center[valid_mask] / w_center[valid_mask]
            v_over_w[valid_mask] = v_center[valid_mask] / w_center[valid_mask]
            one_over_w[valid_mask] = 1.0 / w_center[valid_mask]
            
            # Extract screen-space derivatives
            du_dX = attr_out_da[0, :, :, 0]
            du_dY = attr_out_da[0, :, :, 1]
            dv_dX = attr_out_da[0, :, :, 2]
            dv_dY = attr_out_da[0, :, :, 3]
            dw_dX = attr_out_da_w[0, :, :, 0]
            dw_dY = attr_out_da_w[0, :, :, 1]
            
            # Initialize arrays for quotient rule derivatives
            duw_dX = torch.zeros_like(u_center)
            duw_dY = torch.zeros_like(u_center)
            dvw_dX = torch.zeros_like(v_center)
            dvw_dY = torch.zeros_like(v_center)
            d1w_dX = torch.zeros_like(w_center)
            d1w_dY = torch.zeros_like(w_center)
            
            # Calculate quotient rule derivatives where w != 0
            duw_dX[valid_mask] = (du_dX[valid_mask] * w_center[valid_mask] - u_center[valid_mask] * dw_dX[valid_mask]) / (w_center[valid_mask]**2)
            duw_dY[valid_mask] = (du_dY[valid_mask] * w_center[valid_mask] - u_center[valid_mask] * dw_dY[valid_mask]) / (w_center[valid_mask]**2)
            dvw_dX[valid_mask] = (dv_dX[valid_mask] * w_center[valid_mask] - v_center[valid_mask] * dw_dX[valid_mask]) / (w_center[valid_mask]**2)
            dvw_dY[valid_mask] = (dv_dY[valid_mask] * w_center[valid_mask] - v_center[valid_mask] * dw_dY[valid_mask]) / (w_center[valid_mask]**2)
            d1w_dX[valid_mask] = -dw_dX[valid_mask] / (w_center[valid_mask]**2)
            d1w_dY[valid_mask] = -dw_dY[valid_mask] / (w_center[valid_mask]**2)
            
            # Calculate corners using perspective-correct interpolation
            footprints = []
            for dy, dx in [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)]:
                corner_one_over_w = one_over_w + d1w_dX * dx + d1w_dY * dy
                corner_u_over_w = u_over_w + duw_dX * dx + duw_dY * dy
                corner_v_over_w = v_over_w + dvw_dX * dx + dvw_dY * dy
                
                # Handle valid pixels
                corner_u = torch.zeros_like(corner_u_over_w)
                corner_v = torch.zeros_like(corner_v_over_w)
                corner_u[valid_mask] = corner_u_over_w[valid_mask] / corner_one_over_w[valid_mask]
                corner_v[valid_mask] = corner_v_over_w[valid_mask] / corner_one_over_w[valid_mask]
                if param_idx == 0:
                    print(corner_u[18,9].item(), corner_v[18,9].item())
                else:
                    print(corner_u[18,14].item(), corner_v[18,14].item())
                
                # Scale to 1024 and handle invalid pixels
                corner_u = torch.where(valid_mask, corner_u * 1024, torch.full_like(corner_u, 0))
                corner_v = torch.where(valid_mask, corner_v * 1024, torch.full_like(corner_v, 0))
                
                footprints.append((corner_u.cpu().numpy(), corner_v.cpu().numpy()))
                
            footprints = np.array(footprints)
            footprints = np.where(footprints == 0, 0, np.clip(footprints, 0, tex.shape[1]-1))
            
            footprints = np.transpose(footprints, (2, 3, 0, 1))


            # plots
            # Reshape footprints to (h*w, 4, 2)
            reshaped_footprints = footprints.reshape(-1, 4, 2)
            # Close the quads by repeating the first vertex at the end
            closed_footprints = np.concatenate([reshaped_footprints, reshaped_footprints[:, :1, :]], axis=1)
            # Create a LineCollection for efficient plotting
            lines = LineCollection(closed_footprints, colors='black', linewidths=0.2)#, alpha=0.5)
            # 
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(tex.cpu().numpy(), origin="upper")
            ax.add_collection(lines)
            plt.title("Quads Per Pixel (Flattened to 1D)")
            plt.show()


            # Normalize the depth map and convert to 8-bit (0-255) scale
            depth_map_normalized = (rast_out[0,:,:,2] - rast_out[0,:,:,2].min()) / (rast_out[0,:,:,2].max() - rast_out[0,:,:,2].min())
            depth_map_255 = (depth_map_normalized * 255).cpu().numpy().astype(np.uint8)
            
            # plt.imshow(depth_map_255)
            # plt.title('Normalized depth')
            # plt.colorbar()
            # plt.show()

            results[resolution[0]]['view'].append(texw_out[...,:3])
            results[resolution[0]]['depth'].append(depth_map_255)
            results[resolution[0]]['params'].append({
                'projection': {'x': x, 'n': n, 'f': f},
                'translation': {'x': tx, 'y': ty, 'z': tz},
                'rotation': {'z': rz, 'y': ry, 'x': rx}
            })
            results[resolution[0]]["footprints"].append(footprints)
            results[resolution[0]]["normal"].append(cosine_angles)

    return results
          


if __name__ == "__main__":
    
    # start rendering, rendered will be a list with the number of views as length
    # each element of the list is a dictionary
    # see the print below as an example
    rendered = img_spot()
    
    # print rendered, for the 128 resolution, the footprints are the quads, 0 is the first view and 18,9 is the y,x of the 128x128
    # basically return the quad of the first view for pixel 18,9
    print(rendered[128]["footprints"][0][18,9])
        
    # Initialize CUDA context
    glctx = dr.RasterizeCudaContext()

    # Define infinite space (the code breaks for less than 1024 but we ll not need less than that)
    resolution = (4096, 4096)

      
    # Calculate the texels per quad (it need optimization to run faster)
    for view in range(len(rendered[128]["footprints"])):
        footprints = rendered[128]["footprints"][view] * resolution[0] / 1024
        # footprints = np.round(rendered[128]["footprints"][view] * resolution[0] / 1024)
        footprints_reshaped = footprints.reshape(-1, 4, 2)
        footprints_tensor = torch.tensor(footprints_reshaped, dtype=torch.float32, device="cuda")
        
        views_num = len(rendered[128]["footprints"])
        print(f"########## View: {view+1} / {views_num} ##########")
        
        view_pixel_coords = []
        
        # Process each quad
        for quad_idx, qc in enumerate(footprints_tensor):
            if quad_idx % 1000 == 0:
                print(f"Processing quad {quad_idx}/{len(footprints_tensor)}")
                
            pixel_coords, _ = get_pixels_in_quad(glctx, qc, resolution)
            view_pixel_coords.append(pixel_coords)
        
        # Reshape the list into a 128x128 nested list for numpy-like indexing
        view_pixel_coords_2d = []
        for i in range(0, len(view_pixel_coords), 128):
            row = view_pixel_coords[i:i + 128]
            view_pixel_coords_2d.append(row)
        
        # Create a custom class for numpy-like indexing
        class NestedPixelCoords:
            def __init__(self, data):
                self.data = data
            
            def __getitem__(self, idx):
                if isinstance(idx, tuple):
                    return self.data[idx[0]][idx[1]]
                return self.data[idx]
        
        rendered[128]["map"].append(NestedPixelCoords(view_pixel_coords_2d))


# %% SDXL-controlnet: Depth
# everything is copy pasted here cause it is easier to troubleshoot (instead of a different .py script)


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
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
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
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
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


class MYStableDiffusionXLControlNetPipeline(
    DiffusionPipeline,
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
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

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
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
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
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
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

            # textual inversion: procecss multi-vector tokens if necessary
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
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
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
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
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

    def num_timesteps(self):
        return self._num_timesteps
    
    def pixel_to_ndc(self, coords, width, height):
        """ Convert pixel coordinates to NDC space [-1,1]"""
        x = (2.0 * coords[:, 0] / width) - 1.0
        y = 1.0 - (2.0 * coords[:, 1] / height)  # Flip Y to match OpenGL convention
        return torch.stack([x, y], dim=1)
    
    def get_pixels_in_quad(self, glctx, quad_coords, resolution):
        """
        Get pixel coordinates inside a 2D quad
        
        Args:
            glctx: nvdiffrast context
            quad_coords: tensor of shape (4, 2) containing x,y coordinates in pixel space
            resolution: tuple of (height, width) for output resolution
        
        Returns:
            pixel_coords: Nx2 tensor of (x,y) pixel coordinates inside the quad
        """
        # Convert to NDC space
        quad_coords_ndc = self.pixel_to_ndc(quad_coords, resolution[1], resolution[0])
        
        # Setup vertices with homogeneous coordinates
        vertices = torch.zeros((4,4), dtype=torch.float32, device="cuda")
        vertices[:, :2] = quad_coords_ndc
        vertices[:, 2] = 0.0 # z coordinate
        vertices[:, 3] = 1.0 # w coordinate
        
        # Define triangles
        triangles = torch.tensor([
            [0, 1, 2],
            [0, 2, 3]
            ], dtype=torch.int32, device="cuda")
        
        # Add batch dimension
        vertices = vertices[None, ...]
        
        # Perform rasterization
        rast_out, _ = dr.rasterize(glctx, vertices, triangles, resolution=resolution)
        
        # Get mask of pixels inside quad
        mask = rast_out[0, :, :, 3] > 0
        
        # Get coordinates of pixels where mask is True
        y_coords, x_coords = torch.where(mask)
        
        # Stack into Nx2 tensor of pixel coordinates
        pixel_coords = torch.stack([x_coords, y_coords], dim=1)
        
        return pixel_coords, mask
    
    @property
    def visualize_pixels(self, resolution, quad_coords, pixel_coords, mask):
        """ TO BE DELETED, TEMP...
        Visualize the quad and the pixels inside it"""
        plt.figure(figsize=(10, 5))
        
        # Plot 1: Show the mask
        plt.subplot(1, 2, 1)
        plt.imshow(mask.cpu().numpy(), cmap='gray')
        # Plot quad outline with flipped y-coordinates
        plt.plot(quad_coords.cpu()[:, 0], resolution[0] - quad_coords.cpu()[:, 1], 'r-')
        plt.plot([quad_coords[-1, 0].cpu(), quad_coords[0, 0].cpu()], 
                 [resolution[0] - quad_coords[-1, 1].cpu(), resolution[0] - quad_coords[0, 1].cpu()], 'r-')
        plt.title('Rasterized Quad Mask')
        
        # Plot 2: Show the pixel coordinates
        plt.subplot(1, 2, 2)
        plt.scatter(pixel_coords.cpu()[:, 0], pixel_coords.cpu()[:, 1], s=1)
        # Plot quad outline with flipped y-coordinates
        plt.plot(quad_coords.cpu()[:, 0], resolution[0] - quad_coords.cpu()[:, 1], 'r-')
        plt.plot([quad_coords[-1, 0].cpu(), quad_coords[0, 0].cpu()], 
                 [resolution[0] - quad_coords[-1, 1].cpu(), resolution[0] - quad_coords[0, 1].cpu()], 'r-')
        plt.xlim(0, resolution[1])
        plt.ylim(resolution[0], 0)  # Flip Y axis to match image coordinates
        plt.title('Pixel Coordinates')
        
        plt.tight_layout()
        plt.show()
        

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
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
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
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        inp: dict = None,
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
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
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
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, pooled text embeddings are generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
                weighting). If not provided, pooled `negative_prompt_embeds` are generated from `negative_prompt` input
                argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
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
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

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
            negative_pooled_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

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
        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state
            )
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

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
                for depth_map in inp[1024]["depth"]  # List comprehension to iterate over depth maps
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
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        # # latents = self.prepare_latents(
        # #     batch_size * num_images_per_prompt,
        # #     num_channels_latents,
        # #     height,
        # #     width,
        # #     prompt_embeds.dtype,
        # #     device,
        # #     generator,
        # #     latents,
        # # )



        #
        big_latent = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            4096*8,
            4096*8,
            prompt_embeds.dtype,
            device,
            generator,
            None
        )
        
        base_latent = torch.randn([1, 4, 128, 128], dtype=torch.float16).to("cuda")
        # Create a list of copies of that tensor
        latents = [base_latent.clone() for _ in range(len(inp[128]["view"]))]
        # populate the latents from the big_latent (this is basically rendering)
        for idx, lt in enumerate(latents):
            for y in range(lt.shape[2]):
                for x in range(lt.shape[2]):
                    if (inp[128]["footprints"][idx][y,x] == 0).all():
                        # this means that pixel y,x has to quad, ie background
                        continue
                    else:
                        quad_pxl_values = big_latent[0, :4,
                                            [inp[128]["map"][idx][y,x]][0][:,0], # -> y
                                            [inp[128]["map"][idx][y,x]][0][:,1] # -> x
                                            ]
                        latents[idx][...,y,x] =  quad_pxl_values.sum(axis=-1) / np.sqrt(quad_pxl_values.shape[-1])

        
        




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
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        #
        # add frequency channel
        temp_last_channel = -torch.ones(big_latent.size(0), 1, big_latent.size(2), big_latent.size(3), device=big_latent.device)
        big_latent = torch.cat([big_latent, temp_last_channel], dim=1)
        #
        # # add a new channel for pxls_in_quad_num (extra channel for previous view's values)
        # big_latent = torch.cat([big_latent, -torch.ones((1, 1, big_latent.shape[2], big_latent.shape[3]), dtype=torch.float32, device=big_latent.device)], dim=1)

        # information in the latents not big_latent
        big_latent_current = big_latent.repeat(len(inp[128]["view"]), 1, 1, 1, 1)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # # flush big_latent (initializing multiple times is slower)
                # big_latent = big_latent * 0
                big_latent[:,-1,:,:] = -1
                deltas = []
                ltnts = []
                for idx, (ltnt, depth, _) in enumerate(zip(latents, image, inp[128]["params"])):
                    print(f"view: {idx}, step: {i}")
                    # Relevant thread:
                    # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                    if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                        torch._inductor.cudagraph_mark_step_begin()
                    # expand the latents if we are doing classifier free guidance
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
                        controlnet_cond=depth, #image,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        added_cond_kwargs=controlnet_added_cond_kwargs,
                        return_dict=False,
                    )
    
                    if guess_mode and self.do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
    
                    if ip_adapter_image is not None:
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
                    prev_ltnt = ltnt.clone()
                    ltnt = self.scheduler.step(noise_pred, t, prev_ltnt, **extra_step_kwargs, return_dict=False)[0]
                    # change in the latent, (scheduler's update):
                    # delta = prev_ltnt-ltnt#ltnt-prev_ltnt
                    deltas.append(prev_ltnt-ltnt)
                    ltnts.append(ltnt)
                    
                                        
                for idx, delta in enumerate(deltas):
                    print(f"Updating delta for view {idx}")
                    for y in range(ltnt.shape[2]):
                        for x in range(ltnt.shape[2]):
                            if (inp[128]["footprints"][idx][y,x] == 0).all():
                                # if background keep whatever you have
                                latents[idx][...,y,x] = ltnts[idx][...,y,x]
                            else:
                                # in here there are 2 idx==0. There are such that 
                                # only the first view will update the infinite space
                                # the second view and any view after that will only
                                # render it and will not update it
                                pxl_coords = [inp[128]["map"][idx][y,x]][0]
                                N = pxl_coords.shape[0]
                                d = deltas[idx][..., y, x]
                                D = d.unsqueeze(-1).repeat(1, 1, N)
                                if idx == 0:
                                    big_latent[:, :-1,
                                                pxl_coords[:, 0],  # -> y
                                                pxl_coords[:, 1]  # ->inp[128]["footprints"][idx][y,x] x
                                                ] -= D/np.sqrt(N)
                                # rendering:
                                quad_pxl_values = big_latent[:, :-1,
                                                              pxl_coords[:, 0],
                                                              pxl_coords[:, 1]
                                                              ][0]
                                sqrt_Nor = torch.where(
                                    big_latent[0, -1, pxl_coords[:,0], pxl_coords[:,1]] == -1,
                                    torch.sqrt(torch.tensor(N, dtype=torch.float32)),
                                    big_latent[0, -1, pxl_coords[:,0], pxl_coords[:,1]]
                                    )
                                average_per_pixel = quad_pxl_values * sqrt_Nor
                                latents[idx][...,y,x] =  average_per_pixel.sum(dim=-1) / N
                                if idx==0:
                                    big_latent[:, -1, pxl_coords[:, 0], pxl_coords[:, 1]] = np.sqrt(N)
            
             
                        
                                
                   

                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
               
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
    
                    # if callback_on_step_end is not None:
                    #     callback_kwargs = {}
                    #     for k in callback_on_step_end_tensor_inputs:
                    #         callback_kwargs[k] = locals()[k]
                    #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
    
                    #     latents = callback_outputs.pop("latents", latents)
                    #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
    
                    # # call the callback, if provided
                    # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    #     progress_bar.update()
                    #     if callback is not None and i % callback_steps == 0:
                    #         step_idx = i // getattr(self.scheduler, "order", 1)
                    #         callback(step_idx, t, latents)

        # manually for max memory savings
        # if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
        #     self.upcast_vae()
        #     latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        # if not output_type == "latent":
        #     # make sure the VAE is in float32 mode, as it overflows in float16
        #     needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        #     if needs_upcasting:
        #         self.upcast_vae()
        #         latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        #     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

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





# !pip install opencv-python transformers accelerate
from diffusers import ControlNetModel, AutoencoderKL, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image

# try is used to not reload the model everytime
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
pipe = MYStableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    # variant="fp16",
    # use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, set_alpha_to_one=True) # ToDo: Fix this???

prompt = "photorealistic bedroom"
prompt = "Minecraft grass block"
prompt = "a 3D rendering of a cubic block made of wood and grass, high quality, award winning, photorealistic, unreal engine"
prompt = "photorealistic, cubic block made of wood and grass, high quality, award winning"

controlnet_conditioning_scale = .5  # recommended for good generalization

rnd = torch.randint(0, 10000**2, size=(1,)).item()
rnd = 56582723
print(rnd)
generator = torch.manual_seed(rnd)



dm = Image.fromarray(rendered[128]["view"][0].cpu().numpy()[0,:,:,0])
images = pipe(
    prompt,
    image=dm, # place holder, to be removed
    num_inference_steps=15, # 50
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    generator = generator,
    # inp = im, # to be removed
    inp = rendered,
    # noise = noise,
    # eta=1.0
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























# %%