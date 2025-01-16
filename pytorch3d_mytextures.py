import itertools
import warnings
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures.utils import list_to_packed, list_to_padded, padded_to_list
from torch.nn.functional import interpolate

from pytorch3d.renderer.mesh.utils import pack_unique_rectangles, PackedRectangle, Rectangle

def _list_to_padded_wrapper(
    x: List[torch.Tensor],
    pad_size: Union[list, tuple, None] = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    r"""
    This is a wrapper function for
    pytorch3d.structures.utils.list_to_padded function which only accepts
    3-dimensional inputs.

    For this use case, the input x is of shape (F, 3, ...) where only F
    is different for each element in the list

    Transforms a list of N tensors each of shape (Mi, ...) into a single tensor
    of shape (N, pad_size, ...), or (N, max(Mi), ...)
    if pad_size is None.

    Args:
      x: list of Tensors
      pad_size: int specifying the size of the first dimension
        of the padded tensor
      pad_value: float value to be used to fill the padded tensor

    Returns:
      x_padded: tensor consisting of padded input tensors
    """
    N = len(x)
    dims = x[0].ndim
    reshape_dims = x[0].shape[1:]
    D = torch.prod(torch.tensor(reshape_dims)).item()
    x_reshaped = []
    for y in x:
        if y.ndim != dims and y.shape[1:] != reshape_dims:
            msg = (
                "list_to_padded requires tensors to have the same number of dimensions"
            )
            raise ValueError(msg)
        # pyre-fixme[6]: For 2nd param expected `int` but got `Union[bool, float, int]`.
        x_reshaped.append(y.reshape(-1, D))
    x_padded = list_to_padded(x_reshaped, pad_size=pad_size, pad_value=pad_value)
    # pyre-fixme[58]: `+` is not supported for operand types `Tuple[int, int]` and
    #  `Size`.
    return x_padded.reshape((N, -1) + reshape_dims)


def _padded_to_list_wrapper(
    x: torch.Tensor, split_size: Union[list, tuple, None] = None
) -> List[torch.Tensor]:
    r"""
    This is a wrapper function for pytorch3d.structures.utils.padded_to_list
    which only accepts 3-dimensional inputs.

    For this use case, the input x is of shape (N, F, ...) where F
    is the number of faces which is different for each tensor in the batch.

    This function transforms a padded tensor of shape (N, M, ...) into a
    list of N tensors of shape (Mi, ...) where (Mi) is specified in
    split_size(i), or of shape (M,) if split_size is None.

    Args:
      x: padded Tensor
      split_size: list of ints defining the number of items for each tensor
        in the output list.

    Returns:
      x_list: a list of tensors
    """
    N, M = x.shape[:2]
    reshape_dims = x.shape[2:]
    D = torch.prod(torch.tensor(reshape_dims)).item()
    # pyre-fixme[6]: For 3rd param expected `int` but got `Union[bool, float, int]`.
    x_reshaped = x.reshape(N, M, D)
    x_list = padded_to_list(x_reshaped, split_size=split_size)
    # pyre-fixme[58]: `+` is not supported for operand types `Tuple[typing.Any]` and
    #  `Size`.
    x_list = [xl.reshape((xl.shape[0],) + reshape_dims) for xl in x_list]
    return x_list


def _pad_texture_maps(
    images: Union[Tuple[torch.Tensor], List[torch.Tensor]], align_corners: bool
) -> torch.Tensor:
    """
    Pad all texture images so they have the same height and width.

    Args:
        images: list of N tensors of shape (H_i, W_i, C)
        align_corners: used for interpolation

    Returns:
        tex_maps: Tensor of shape (N, max_H, max_W, C)
    """
    tex_maps = []
    max_H = 0
    max_W = 0
    for im in images:
        h, w, _C = im.shape
        if h > max_H:
            max_H = h
        if w > max_W:
            max_W = w
        tex_maps.append(im)
    max_shape = (max_H, max_W)

    for i, image in enumerate(tex_maps):
        if image.shape[:2] != max_shape:
            image_BCHW = image.permute(2, 0, 1)[None]
            new_image_BCHW = interpolate(
                image_BCHW,
                size=max_shape,
                mode="bilinear",
                align_corners=align_corners,
            )
            tex_maps[i] = new_image_BCHW[0].permute(1, 2, 0)
    tex_maps = torch.stack(tex_maps, dim=0)  # (num_tex_maps, max_H, max_W, C)
    return tex_maps


def _pad_texture_multiple_maps(
    multiple_texture_maps: Union[Tuple[torch.Tensor], List[torch.Tensor]],
    align_corners: bool,
) -> torch.Tensor:
    """
    Pad all texture images so they have the same height and width.

    Args:
        images: list of N tensors of shape (M_i, H_i, W_i, C)
        M_i : Number of texture maps:w

        align_corners: used for interpolation

    Returns:
        tex_maps: Tensor of shape (N, max_M, max_H, max_W, C)
    """
    tex_maps = []
    max_M = 0
    max_H = 0
    max_W = 0
    C = 0
    for im in multiple_texture_maps:
        m, h, w, C = im.shape
        if m > max_M:
            max_M = m
        if h > max_H:
            max_H = h
        if w > max_W:
            max_W = w
        tex_maps.append(im)
    max_shape = (max_M, max_H, max_W, C)
    max_im_shape = (max_H, max_W)
    for i, tms in enumerate(tex_maps):
        new_tex_maps = torch.zeros(max_shape)
        for j in range(tms.shape[0]):
            im = tms[j]
            if im.shape[:2] != max_im_shape:
                image_BCHW = im.permute(2, 0, 1)[None]
                new_image_BCHW = interpolate(
                    image_BCHW,
                    size=max_im_shape,
                    mode="bilinear",
                    align_corners=align_corners,
                )
                new_tex_maps[j] = new_image_BCHW[0].permute(1, 2, 0)
            else:
                new_tex_maps[j] = im
        tex_maps[i] = new_tex_maps
    tex_maps = torch.stack(tex_maps, dim=0)  # (num_tex_maps, max_H, max_W, C)
    return tex_maps


class TexturesBase:
    def isempty(self):
        if self._N is not None and self.valid is not None:
            return self._N == 0 or self.valid.eq(False).all()
        return False

    def to(self, device):
        for k in dir(self):
            v = getattr(self, k)
            if isinstance(v, (list, tuple)) and all(
                torch.is_tensor(elem) for elem in v
            ):
                v = [elem.to(device) for elem in v]
                setattr(self, k, v)
            if torch.is_tensor(v) and v.device != device:
                setattr(self, k, v.to(device))
        self.device = device
        return self

    def _extend(self, N: int, props: List[str]) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Create a dict with the specified properties
        repeated N times per batch element.

        Args:
            N: number of new copies of each texture
                in the batch.
            props: a List of strings which refer to either
                class attributes or class methods which
                return tensors or lists.

        Returns:
            Dict with the same keys as props. The values are the
            extended properties.
        """
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")
        if N <= 0:
            raise ValueError("N must be > 0.")

        new_props = {}
        for p in props:
            t = getattr(self, p)
            if callable(t):
                t = t()  # class method
            if t is None:
                new_props[p] = None
            elif isinstance(t, list):
                if not all(isinstance(elem, (int, float)) for elem in t):
                    raise ValueError("Extend only supports lists of scalars")
                t = [[ti] * N for ti in t]
                new_props[p] = list(itertools.chain(*t))
            elif torch.is_tensor(t):
                new_props[p] = t.repeat_interleave(N, dim=0)
            else:
                raise ValueError(
                    f"Property {p} has unsupported type {type(t)}."
                    "Only tensors and lists are supported."
                )
        return new_props

    def _getitem(self, index: Union[int, slice], props: List[str]):
        """
        Helper function for __getitem__
        """
        new_props = {}
        if isinstance(index, (int, slice)):
            for p in props:
                t = getattr(self, p)
                if callable(t):
                    t = t()  # class method
                new_props[p] = t[index] if t is not None else None
        elif isinstance(index, list):
            index = torch.tensor(index)
        if isinstance(index, torch.Tensor):
            if index.dtype == torch.bool:
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            for p in props:
                t = getattr(self, p)
                if callable(t):
                    t = t()  # class method
                new_props[p] = [t[i] for i in index] if t is not None else None
        return new_props

    def sample_textures(self) -> torch.Tensor:
        """
        Different texture classes sample textures in different ways
        e.g. for vertex textures, the values at each vertex
        are interpolated across the face using the barycentric
        coordinates.
        Each texture class should implement a sample_textures
        method to take the `fragments` from rasterization.
        Using `fragments.pix_to_face` and `fragments.bary_coords`
        this function should return the sampled texture values for
        each pixel in the output image.
        """
        raise NotImplementedError()

    def submeshes(
        self,
        vertex_ids_list: List[List[torch.LongTensor]],
        faces_ids_list: List[List[torch.LongTensor]],
    ) -> "TexturesBase":
        """
        Extract sub-textures used for submeshing.
        """
        raise NotImplementedError(f"{self.__class__} does not support submeshes")

    def faces_verts_textures_packed(self) -> torch.Tensor:
        """
        Returns the texture for each vertex for each face in the mesh.
        For N meshes, this function returns sum(Fi)x3xC where Fi is the
        number of faces in the i-th mesh and C is the dimensional of
        the feature (C = 3 for RGB textures).
        You can use the utils function in structures.utils to convert the
        packed representation to a list or padded.
        """
        raise NotImplementedError()

    def clone(self) -> "TexturesBase":
        """
        Each texture class should implement a method
        to clone all necessary internal tensors.
        """
        raise NotImplementedError()

    def detach(self) -> "TexturesBase":
        """
        Each texture class should implement a method
        to detach all necessary internal tensors.
        """
        raise NotImplementedError()

    def __getitem__(self, index) -> "TexturesBase":
        """
        Each texture class should implement a method
        to get the texture properties for the
        specified elements in the batch.
        The TexturesBase._getitem(i) method
        can be used as a helper function to retrieve the
        class attributes for item i. Then, a new
        instance of the child class can be created with
        the attributes.
        """
        raise NotImplementedError()

# def Textures(
#     maps: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
#     faces_uvs: Optional[torch.Tensor] = None,
#     verts_uvs: Optional[torch.Tensor] = None,
#     verts_rgb: Optional[torch.Tensor] = None,
# ) -> TexturesBase:
#     """
#     Textures class has been DEPRECATED.
#     Preserving Textures as a function for backwards compatibility.

#     Args:
#         maps: texture map per mesh. This can either be a list of maps
#           [(H, W, C)] or a padded tensor of shape (N, H, W, C).
#         faces_uvs: (N, F, 3) tensor giving the index into verts_uvs for each
#             vertex in the face. Padding value is assumed to be -1.
#         verts_uvs: (N, V, 2) tensor giving the uv coordinate per vertex.
#         verts_rgb: (N, V, C) tensor giving the color per vertex. Padding
#             value is assumed to be -1. (C=3 for RGB.)


#     Returns:
#         a Textures class which is an instance of TexturesBase e.g. TexturesUV,
#         TexturesAtlas, TexturesVertex

#     """

#     warnings.warn(
#         """Textures class is deprecated,
#         use TexturesUV, TexturesAtlas, TexturesVertex instead.
#         Textures class will be removed in future releases.""",
#         PendingDeprecationWarning,
#     )

#     if faces_uvs is not None and verts_uvs is not None and maps is not None:
#         return TexturesUV(maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)

#     if verts_rgb is not None:
#         return TexturesVertex(verts_features=verts_rgb)

#     raise ValueError(
#         "Textures either requires all three of (faces uvs, verts uvs, maps) or verts rgb"
#     )


class TexturesUV(TexturesBase):
    def __init__(
        self,
        maps: Union[torch.Tensor, List[torch.Tensor]],
        faces_uvs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        verts_uvs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        *,
        maps_ids: Optional[
            Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
        ] = None,
        padding_mode: str = "border",
        align_corners: bool = True,
        sampling_mode: str = "bilinear",
    ) -> None:
        """
        Textures are represented as a per mesh texture map and uv coordinates for each
        vertex in each face. NOTE: this class only supports one texture map per mesh.

        Args:
            maps: Either (1) a texture map per mesh. This can either be a list of maps
                    [(H, W, C)] or a padded tensor of shape (N, H, W, C).
                    For RGB, C = 3. In this case maps_ids must be None.
                Or (2) a set of M texture maps per mesh. This can either be a list of sets
                    [(M, H, W, C)] or a padded tensor of shape (N, M, H, W, C).
                    For RGB, C = 3. In this case maps_ids must be provided to
                    identify which is relevant to each face.
            faces_uvs: (N, F, 3) LongTensor giving the index into verts_uvs
                    for each face
            verts_uvs: (N, V, 2) tensor giving the uv coordinates per vertex
                    (a FloatTensor with values between 0 and 1).
            maps_ids: Used if there are to be multiple maps per face. This can be either a list of map_ids [(F,)]
                    or a long tensor of shape (N, F) giving the id of the texture map
                    for each face. If maps_ids is present, the maps has an extra dimension M
                    (so maps_padded is (N, M, H, W, C) and maps_list has elements of
                    shape (M, H, W, C)).
                    Specifically, the color
                    of a vertex V is given by an average of maps_padded[i, maps_ids[i, f], u, v, :]
                    over u and v integers adjacent to
                    _verts_uvs_padded[i, _faces_uvs_padded[i, f, 0], :] .
            align_corners: If true, the extreme values 0 and 1 for verts_uvs
                    indicate the centers of the edge pixels in the maps.
            padding_mode: padding mode for outside grid values
                    ("zeros", "border" or "reflection").
            sampling_mode: type of interpolation used to sample the texture.
                    Corresponds to the mode parameter in PyTorch's
                    grid_sample ("nearest" or "bilinear").

        The align_corners and padding_mode arguments correspond to the arguments
        of the `grid_sample` torch function. There is an informative illustration of
        the two align_corners options at
        https://discuss.pytorch.org/t/22663/9 .

        An example of how the indexing into the maps, with align_corners=True,
        works is as follows.
        If maps[i] has shape [1001, 101] and the value of verts_uvs[i][j]
        is [0.4, 0.3], then a value of j in faces_uvs[i] means a vertex
        whose color is given by maps[i][700, 40]. padding_mode affects what
        happens if a value in verts_uvs is less than 0 or greater than 1.
        Note that increasing a value in verts_uvs[..., 0] increases an index
        in maps, whereas increasing a value in verts_uvs[..., 1] _decreases_
        an _earlier_ index in maps.

        If align_corners=False, an example would be as follows.
        If maps[i] has shape [1000, 100] and the value of verts_uvs[i][j]
        is [0.405, 0.2995], then a value of j in faces_uvs[i] means a vertex
        whose color is given by maps[i][700, 40].
        When align_corners=False, padding_mode even matters for values in
        verts_uvs slightly above 0 or slightly below 1. In this case, the
        padding_mode matters if the first value is outside the interval
        [0.0005, 0.9995] or if the second is outside the interval
        [0.005, 0.995].
        """

        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.sampling_mode = sampling_mode
        if isinstance(faces_uvs, (list, tuple)):
            for fv in faces_uvs:
                if fv.ndim != 2 or fv.shape[-1] != 3:
                    msg = "Expected faces_uvs to be of shape (F, 3); got %r"
                    raise ValueError(msg % repr(fv.shape))
            self._faces_uvs_list = faces_uvs
            self._faces_uvs_padded = None
            self.device = torch.device("cpu")

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(faces_uvs)
            self._num_faces_per_mesh = [len(fv) for fv in faces_uvs]

            if self._N > 0:
                self.device = faces_uvs[0].device

        elif torch.is_tensor(faces_uvs):
            if faces_uvs.ndim != 3 or faces_uvs.shape[-1] != 3:
                msg = "Expected faces_uvs to be of shape (N, F, 3); got %r"
                raise ValueError(msg % repr(faces_uvs.shape))
            self._faces_uvs_padded = faces_uvs
            self._faces_uvs_list = None
            self.device = faces_uvs.device

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(faces_uvs)
            max_F = faces_uvs.shape[1]
            self._num_faces_per_mesh = [max_F] * self._N
        else:
            raise ValueError("Expected faces_uvs to be a tensor or list")

        if isinstance(verts_uvs, (list, tuple)):
            for fv in verts_uvs:
                if fv.ndim != 2 or fv.shape[-1] != 2:
                    msg = "Expected verts_uvs to be of shape (V, 2); got %r"
                    raise ValueError(msg % repr(fv.shape))
            self._verts_uvs_list = verts_uvs
            self._verts_uvs_padded = None

            if len(verts_uvs) != self._N:
                raise ValueError(
                    "verts_uvs and faces_uvs must have the same batch dimension"
                )
            if not all(v.device == self.device for v in verts_uvs):
                raise ValueError("verts_uvs and faces_uvs must be on the same device")

        elif torch.is_tensor(verts_uvs):
            if (
                verts_uvs.ndim != 3
                or verts_uvs.shape[-1] != 2
                or verts_uvs.shape[0] != self._N
            ):
                msg = "Expected verts_uvs to be of shape (N, V, 2); got %r"
                raise ValueError(msg % repr(verts_uvs.shape))
            self._verts_uvs_padded = verts_uvs
            self._verts_uvs_list = None

            if verts_uvs.device != self.device:
                raise ValueError("verts_uvs and faces_uvs must be on the same device")
        else:
            raise ValueError("Expected verts_uvs to be a tensor or list")

        self._maps_ids_padded, self._maps_ids_list = self._format_maps_ids(maps_ids)

        if isinstance(maps, (list, tuple)):
            self._maps_list = maps
        else:
            self._maps_list = None
        self._maps_padded = self._format_maps_padded(maps)

        if self._maps_padded.device != self.device:
            raise ValueError("maps must be on the same device as verts/faces uvs.")
        self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)

    def _format_maps_ids(
        self,
        maps_ids: Optional[
            Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
        ],
    ) -> Tuple[
        Optional[torch.Tensor], Optional[Union[List[torch.Tensor], Tuple[torch.Tensor]]]
    ]:
        if maps_ids is None:
            return None, None
        elif isinstance(maps_ids, (list, tuple)):
            for mid in maps_ids:
                if mid.ndim != 1:
                    msg = "Expected maps_ids to be of shape (F,); got %r"
                    raise ValueError(msg % repr(mid.shape))
            if len(maps_ids) != self._N:
                raise ValueError(
                    "map_ids, faces_uvs and verts_uvs must have the same batch dimension"
                )
            if not all(mid.device == self.device for mid in maps_ids):
                raise ValueError(
                    "maps_ids and verts/faces uvs must be on the same device"
                )

            if not all(
                mid.shape[0] == nfm
                for mid, nfm in zip(maps_ids, self._num_faces_per_mesh)
            ):
                raise ValueError(
                    "map_ids and faces_uvs must have the same number of faces per mesh"
                )
            if not all(mid.device == self.device for mid in maps_ids):
                raise ValueError(
                    "maps_ids and verts/faces uvs must be on the same device"
                )
            if not self._num_faces_per_mesh:
                return torch.Tensor(), maps_ids
            return list_to_padded(maps_ids, pad_value=0), maps_ids
        elif isinstance(maps_ids, torch.Tensor):
            if maps_ids.ndim != 2 or maps_ids.shape[0] != self._N:
                msg = "Expected maps_ids to be of shape (N, F); got %r"
                raise ValueError(msg % repr(maps_ids.shape))
            maps_ids_padded = maps_ids
            max_F = max(self._num_faces_per_mesh)
            if not maps_ids.shape[1] == max_F:
                raise ValueError(
                    "map_ids and faces_uvs must have the same number of faces per mesh"
                )
            if maps_ids.device != self.device:
                raise ValueError(
                    "maps_ids and verts/faces uvs must be on the same device"
                )
            return maps_ids_padded, None
        raise ValueError("Expected maps_ids to be a tensor or list")

    def _format_maps_padded(
        self, maps: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        maps_ids_none = self._maps_ids_padded is None
        if isinstance(maps, torch.Tensor):
            if not maps_ids_none:
                if maps.ndim != 5 or maps.shape[0] != self._N:
                    msg = "Expected maps to be of shape (N, M, H, W, C); got %r"
                    raise ValueError(msg % repr(maps.shape))
            elif maps.ndim != 4 or maps.shape[0] != self._N:
                msg = "Expected maps to be of shape (N, H, W, C); got %r"
                raise ValueError(msg % repr(maps.shape))
            return maps

        if isinstance(maps, (list, tuple)):
            if len(maps) != self._N:
                raise ValueError("Expected one texture map per mesh in the batch.")
            if self._N > 0:
                ndim = 3 if maps_ids_none else 4
                if not all(map.ndim == ndim for map in maps):
                    raise ValueError("Invalid number of dimensions in texture maps")
                if not all(map.shape[-1] == maps[0].shape[-1] for map in maps):
                    raise ValueError("Inconsistent number of channels in maps")
                maps_padded = (
                    _pad_texture_maps(maps, align_corners=self.align_corners)
                    if maps_ids_none
                    else _pad_texture_multiple_maps(
                        maps, align_corners=self.align_corners
                    )
                )
            else:
                if maps_ids_none:
                    maps_padded = torch.empty(
                        (self._N, 0, 0, 3), dtype=torch.float32, device=self.device
                    )
                else:
                    maps_padded = torch.empty(
                        (self._N, 0, 0, 0, 3), dtype=torch.float32, device=self.device
                    )
            return maps_padded

        raise ValueError("Expected maps to be a tensor or list of tensors.")

    def clone(self) -> "TexturesUV":
        tex = self.__class__(
            self.maps_padded().clone(),
            self.faces_uvs_padded().clone(),
            self.verts_uvs_padded().clone(),
            maps_ids=(
                self._maps_ids_padded.clone()
                if self._maps_ids_padded is not None
                else None
            ),
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
            sampling_mode=self.sampling_mode,
        )
        if self._maps_list is not None:
            tex._maps_list = [m.clone() for m in self._maps_list]
        if self._verts_uvs_list is not None:
            tex._verts_uvs_list = [v.clone() for v in self._verts_uvs_list]
        if self._faces_uvs_list is not None:
            tex._faces_uvs_list = [f.clone() for f in self._faces_uvs_list]
        if self._maps_ids_list is not None:
            tex._maps_ids_list = [f.clone() for f in self._maps_ids_list]
        num_faces = (
            self._num_faces_per_mesh.clone()
            if torch.is_tensor(self._num_faces_per_mesh)
            else self._num_faces_per_mesh
        )
        tex._num_faces_per_mesh = num_faces
        tex.valid = self.valid.clone()
        return tex

    def detach(self) -> "TexturesUV":
        tex = self.__class__(
            self.maps_padded().detach(),
            self.faces_uvs_padded().detach(),
            self.verts_uvs_padded().detach(),
            maps_ids=(
                self._maps_ids_padded.detach()
                if self._maps_ids_padded is not None
                else None
            ),
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
            sampling_mode=self.sampling_mode,
        )
        if self._maps_list is not None:
            tex._maps_list = [m.detach() for m in self._maps_list]
        if self._verts_uvs_list is not None:
            tex._verts_uvs_list = [v.detach() for v in self._verts_uvs_list]
        if self._faces_uvs_list is not None:
            tex._faces_uvs_list = [f.detach() for f in self._faces_uvs_list]
        if self._maps_ids_list is not None:
            tex._maps_ids_list = [mi.detach() for mi in self._maps_ids_list]
        num_faces = (
            self._num_faces_per_mesh.detach()
            if torch.is_tensor(self._num_faces_per_mesh)
            else self._num_faces_per_mesh
        )
        tex._num_faces_per_mesh = num_faces
        tex.valid = self.valid.detach()
        return tex

    def __getitem__(self, index) -> "TexturesUV":
        props = [
            "faces_uvs_list",
            "verts_uvs_list",
            "maps_list",
            "maps_ids_list",
            "_num_faces_per_mesh",
        ]
        new_props = self._getitem(index, props)
        faces_uvs = new_props["faces_uvs_list"]
        verts_uvs = new_props["verts_uvs_list"]
        maps = new_props["maps_list"]
        maps_ids = new_props["maps_ids_list"]

        # if index has multiple values then faces/verts/maps may be a list of tensors
        if all(isinstance(f, (list, tuple)) for f in [faces_uvs, verts_uvs, maps]):
            if maps_ids is not None and not isinstance(maps_ids, (list, tuple)):
                raise ValueError(
                    "Maps ids are  not in the correct format expected list or tuple"
                )
            new_tex = self.__class__(
                faces_uvs=faces_uvs,
                verts_uvs=verts_uvs,
                maps=maps,
                maps_ids=maps_ids,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                sampling_mode=self.sampling_mode,
            )
        elif all(torch.is_tensor(f) for f in [faces_uvs, verts_uvs, maps]):
            if maps_ids is not None and not torch.is_tensor(maps_ids):
                raise ValueError(
                    "Maps ids are not in the correct format expected tensor"
                )
            new_tex = self.__class__(
                faces_uvs=[faces_uvs],
                verts_uvs=[verts_uvs],
                maps=[maps],
                maps_ids=[maps_ids] if maps_ids is not None else None,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                sampling_mode=self.sampling_mode,
            )
        else:
            raise ValueError("Not all values are provided in the correct format")
        new_tex._num_faces_per_mesh = new_props["_num_faces_per_mesh"]
        return new_tex

    def faces_uvs_padded(self) -> torch.Tensor:
        if self._faces_uvs_padded is None:
            if self.isempty():
                self._faces_uvs_padded = torch.zeros(
                    (self._N, 0, 3), dtype=torch.float32, device=self.device
                )
            else:
                self._faces_uvs_padded = list_to_padded(
                    self._faces_uvs_list, pad_value=0.0
                )
        return self._faces_uvs_padded

    def faces_uvs_list(self) -> List[torch.Tensor]:
        if self._faces_uvs_list is None:
            if self.isempty():
                self._faces_uvs_list = [
                    torch.empty((0, 3), dtype=torch.float32, device=self.device)
                ] * self._N
            else:
                self._faces_uvs_list = padded_to_list(
                    self._faces_uvs_padded, split_size=self._num_faces_per_mesh
                )
        return self._faces_uvs_list

    def verts_uvs_padded(self) -> torch.Tensor:
        if self._verts_uvs_padded is None:
            if self.isempty():
                self._verts_uvs_padded = torch.zeros(
                    (self._N, 0, 2), dtype=torch.float32, device=self.device
                )
            else:
                self._verts_uvs_padded = list_to_padded(
                    self._verts_uvs_list, pad_value=0.0
                )
        return self._verts_uvs_padded

    def verts_uvs_list(self) -> List[torch.Tensor]:
        if self._verts_uvs_list is None:
            if self.isempty():
                self._verts_uvs_list = [
                    torch.empty((0, 2), dtype=torch.float32, device=self.device)
                ] * self._N
            else:
                # The number of vertices in the mesh and in verts_uvs can differ
                # e.g. if a vertex is shared between 3 faces, it can
                # have up to 3 different uv coordinates.
                self._verts_uvs_list = list(self._verts_uvs_padded.unbind(0))
        return self._verts_uvs_list

    def maps_ids_padded(self) -> Optional[torch.Tensor]:
        return self._maps_ids_padded

    def maps_ids_list(self) -> Optional[List[torch.Tensor]]:
        if self._maps_ids_list is not None:
            return self._maps_ids_list
        elif self._maps_ids_padded is not None:
            return self._maps_ids_padded.unbind(0)
        else:
            return None

    # Currently only the padded maps are used.
    def maps_padded(self) -> torch.Tensor:
        return self._maps_padded

    def maps_list(self) -> List[torch.Tensor]:
        if self._maps_list is not None:
            return self._maps_list
        return self._maps_padded.unbind(0)

    def extend(self, N: int) -> "TexturesUV":
        new_props = self._extend(
            N,
            [
                "maps_padded",
                "verts_uvs_padded",
                "faces_uvs_padded",
                "maps_ids_padded",
                "_num_faces_per_mesh",
            ],
        )
        new_tex = self.__class__(
            maps=new_props["maps_padded"],
            faces_uvs=new_props["faces_uvs_padded"],
            verts_uvs=new_props["verts_uvs_padded"],
            maps_ids=new_props["maps_ids_padded"],
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            sampling_mode=self.sampling_mode,
        )

        new_tex._num_faces_per_mesh = new_props["_num_faces_per_mesh"]
        return new_tex

    # pyre-fixme[14]: `sample_textures` overrides method defined in `TexturesBase`
    #  inconsistently.
    def sample_textures(self, fragments, **kwargs) -> torch.Tensor:
        """
        Interpolate a 2D texture map using uv vertex texture coordinates for each
        face in the mesh. First interpolate the vertex uvs using barycentric coordinates
        for each pixel in the rasterized output. Then interpolate the texture map
        using the uv coordinate for each pixel.

        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordinates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: tensor of shape (N, H, W, K, C) giving the interpolated
            texture for each pixel in the rasterized image.
        """
        
        if self.isempty():
            faces_verts_uvs = torch.zeros(
                (self._N, 3, 2), dtype=torch.float32, device=self.device
            )
        else:
            packing_list = [
                i[j] for i, j in zip(self.verts_uvs_list(), self.faces_uvs_list())
            ]
            faces_verts_uvs = torch.cat(packing_list)

        # pixel_uvs: (N, H, W, K, 2)
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )

        N, H_out, W_out, K = fragments.pix_to_face.shape

        texture_maps = self.maps_padded()
        maps_ids_padded = self.maps_ids_padded()
        if maps_ids_padded is None:
            # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
            pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)
            N, H_in, W_in, C = texture_maps.shape  # 3 for RGB

            # textures.map:
            #   (N, H, W, C) -> (N, C, H, W) -> (1, N, C, H, W)
            #   -> expand (K, N, C, H, W) -> reshape (N*K, C, H, W)
            texture_maps = (
                texture_maps.permute(0, 3, 1, 2)[None, ...]
                .expand(K, -1, -1, -1, -1)
                .transpose(0, 1)
                .reshape(N * K, C, H_in, W_in)
            )
            # Textures: (N*K, C, H, W), pixel_uvs: (N*K, H, W, 2)
            # Now need to format the pixel uvs and the texture map correctly!
            # From pytorch docs, grid_sample takes `grid` and `input`:
            #   grid specifies the sampling pixel locations normalized by
            #   the input spatial dimensions It should have most
            #   values in the range of [-1, 1]. Values x = -1, y = -1
            #   is the left-top pixel of input, and values x = 1, y = 1 is the
            #   right-bottom pixel of input.

            # map to a range of [-1, 1] and flip the y axis
            pixel_uvs = torch.lerp(
                pixel_uvs.new_tensor([-1.0, 1.0]),
                pixel_uvs.new_tensor([1.0, -1.0]),
                pixel_uvs,
            )

            if texture_maps.device != pixel_uvs.device:
                texture_maps = texture_maps.to(pixel_uvs.device)
            texels = F.grid_sample(
                texture_maps,
                pixel_uvs,
                mode=self.sampling_mode,
                align_corners=self.align_corners,
                padding_mode=self.padding_mode,
            )
            # texels now has shape (NK, C, H_out, W_out)
            texels = texels.reshape(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)
            return texels
        else:
            # We have maps_ids_padded: (N, F), textures_map: (N, M, Hi, Wi, C),fragmenmts.pix_to_face: (N, Ho, Wo, K)
            # Get pixel_to_map_ids: (N, K, Ho, Wo) by indexing pix_to_face into maps_ids
            N, M, H_in, W_in, C = texture_maps.shape  # 3 for RGB

            mask = fragments.pix_to_face < 0
            pix_to_face = fragments.pix_to_face.clone()
            pix_to_face[mask] = 0

            pixel_to_map_ids = (
                maps_ids_padded.flatten()
                .gather(0, pix_to_face.flatten())
                .view(N, H_out, W_out, K, 1)
                .permute(0, 3, 1, 2, 4)
            )  # N x H_out x W_out x K x 1

            # Normalize between -1 and 1 with M (number of maps)
            pixel_to_map_ids = (2.0 * pixel_to_map_ids.float() / float(M - 1)) - 1
            pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4)
            pixel_uvs = torch.lerp(
                pixel_uvs.new_tensor([-1.0, 1.0]),
                pixel_uvs.new_tensor([1.0, -1.0]),
                pixel_uvs,
            )  # N x H_out x W_out x K x 2

            # N x H_out x W_out x K x 3
            pixel_uvms = torch.cat((pixel_uvs, pixel_to_map_ids), dim=4)
            # (N, M, H, W, C) -> (N, C, M, H, W)
            texture_maps = texture_maps.permute(0, 4, 1, 2, 3)
            if texture_maps.device != pixel_uvs.device:
                texture_maps = texture_maps.to(pixel_uvs.device)
            texels = F.grid_sample(
                texture_maps,
                pixel_uvms,
                mode=self.sampling_mode,
                align_corners=self.align_corners,
                padding_mode=self.padding_mode,
            )
            # (N, C, K, H_out, W_out) -> (N, H_out, W_out, K, C)
            texels = texels.permute(0, 3, 4, 2, 1).contiguous()
            return texels

    def faces_verts_textures_packed(self) -> torch.Tensor:
        """
        Samples texture from each vertex and for each face in the mesh.
        For N meshes with {Fi} number of faces, it returns a
        tensor of shape sum(Fi)x3xC (C = 3 for RGB).
        You can use the utils function in structures.utils to convert the
        packed representation to a list or padded.
        """
        if self.isempty():
            return torch.zeros(
                (0, 3, self.maps_padded().shape[-1]),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            packing_list = [
                i[j] for i, j in zip(self.verts_uvs_list(), self.faces_uvs_list())
            ]
            faces_verts_uvs = _list_to_padded_wrapper(
                packing_list, pad_value=0.0
            )  # Nxmax(Fi)x3x2
        # map to a range of [-1, 1] and flip the y axis
        faces_verts_uvs = torch.lerp(
            faces_verts_uvs.new_tensor([-1.0, 1.0]),
            faces_verts_uvs.new_tensor([1.0, -1.0]),
            faces_verts_uvs,
        )
        texture_maps = self.maps_padded()  # NxHxWxC or NxMxHxWxC
        maps_ids_padded = self.maps_ids_padded()
        if maps_ids_padded is None:
            texture_maps = texture_maps.permute(0, 3, 1, 2)  # NxCxHxW
        else:
            M = texture_maps.shape[1]
            # (N, M, H, W, C) -> (N, C, M, H, W)
            texture_maps = texture_maps.permute(0, 4, 1, 2, 3)
            # expand maps_ids to (N, F, 3, 1)
            maps_ids_padded = maps_ids_padded[:, :, None, None].expand(-1, -1, 3, -1)
            maps_ids_padded = (2.0 * maps_ids_padded.float() / float(M - 1)) - 1.0

            # (N, F, 3, 2+1) -> (N, 1, F, 3, 3)
            faces_verts_uvs = torch.cat(
                (faces_verts_uvs, maps_ids_padded), dim=3
            ).unsqueeze(1)
            # (N, M, H, W, C) -> (N, C, H, W, M)
            # texture_maps = texture_maps.permute(0, 4, 2, 3, 1)
        textures = F.grid_sample(
            texture_maps,
            faces_verts_uvs,
            mode=self.sampling_mode,
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
        )  # (N, C, max(Fi), 3)
        if maps_ids_padded is not None:
            textures = textures.squeeze(dim=2)
        # (N, C, max(Fi), 3) -> (N, max(Fi), 3, C)
        textures = textures.permute(0, 2, 3, 1)
        textures = _padded_to_list_wrapper(
            textures, split_size=self._num_faces_per_mesh
        )  # list of N {Fix3xC} tensors
        return list_to_packed(textures)[0]

    def join_batch(self, textures: List["TexturesUV"]) -> "TexturesUV":
        """
        Join the list of textures given by `textures` to
        self to create a batch of textures. Return a new
        TexturesUV object with the combined textures.

        Args:
            textures: List of TexturesUV objects

        Returns:
            new_tex: TexturesUV object with the combined
            textures from self and the list `textures`.
        """
        if self.maps_ids_padded() is not None:
            # TODO
            raise NotImplementedError(
                "join_batch does not support TexturesUV with multiple maps"
            )
        tex_types_same = all(isinstance(tex, TexturesUV) for tex in textures)
        if not tex_types_same:
            raise ValueError("All textures must be of type TexturesUV.")

        padding_modes_same = all(
            tex.padding_mode == self.padding_mode for tex in textures
        )
        if not padding_modes_same:
            raise ValueError("All textures must have the same padding_mode.")
        align_corners_same = all(
            tex.align_corners == self.align_corners for tex in textures
        )
        if not align_corners_same:
            raise ValueError("All textures must have the same align_corners value.")
        sampling_mode_same = all(
            tex.sampling_mode == self.sampling_mode for tex in textures
        )
        if not sampling_mode_same:
            raise ValueError("All textures must have the same sampling_mode.")

        verts_uvs_list = []
        faces_uvs_list = []
        maps_list = []
        faces_uvs_list += self.faces_uvs_list()
        verts_uvs_list += self.verts_uvs_list()
        maps_list += self.maps_list()
        num_faces_per_mesh = self._num_faces_per_mesh.copy()
        for tex in textures:
            verts_uvs_list += tex.verts_uvs_list()
            faces_uvs_list += tex.faces_uvs_list()
            num_faces_per_mesh += tex._num_faces_per_mesh
            maps_list += tex.maps_list()

        new_tex = self.__class__(
            maps=maps_list,
            faces_uvs=faces_uvs_list,
            verts_uvs=verts_uvs_list,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            sampling_mode=self.sampling_mode,
        )
        new_tex._num_faces_per_mesh = num_faces_per_mesh
        return new_tex

    def _place_map_into_single_map(
        self, single_map: torch.Tensor, map_: torch.Tensor, location: PackedRectangle
    ) -> None:
        """
        Copy map into a larger tensor single_map at the destination specified by location.
        If align_corners is False, we add the needed border around the destination.

        Used by join_scene.

        Args:
            single_map: (total_H, total_W, C)
            map_: (H, W, C) source data
            location: where to place map
        """
        do_flip = location.flipped
        source = map_.transpose(0, 1) if do_flip else map_
        border_width = 0 if self.align_corners else 1
        lower_u = location.x + border_width
        lower_v = location.y + border_width
        upper_u = lower_u + source.shape[0]
        upper_v = lower_v + source.shape[1]
        single_map[lower_u:upper_u, lower_v:upper_v] = source

        if self.padding_mode != "zeros" and not self.align_corners:
            single_map[lower_u - 1, lower_v:upper_v] = single_map[
                lower_u, lower_v:upper_v
            ]
            single_map[upper_u, lower_v:upper_v] = single_map[
                upper_u - 1, lower_v:upper_v
            ]
            single_map[lower_u:upper_u, lower_v - 1] = single_map[
                lower_u:upper_u, lower_v
            ]
            single_map[lower_u:upper_u, upper_v] = single_map[
                lower_u:upper_u, upper_v - 1
            ]
            single_map[lower_u - 1, lower_v - 1] = single_map[lower_u, lower_v]
            single_map[lower_u - 1, upper_v] = single_map[lower_u, upper_v - 1]
            single_map[upper_u, lower_v - 1] = single_map[upper_u - 1, lower_v]
            single_map[upper_u, upper_v] = single_map[upper_u - 1, upper_v - 1]

    def join_scene(self) -> "TexturesUV":
        """
        Return a new TexturesUV amalgamating the batch.

        We calculate a large single map which contains the original maps,
        and find verts_uvs to point into it. This will not replicate
        behavior of padding for verts_uvs values outside [0,1].

        If align_corners=False, we need to add an artificial border around
        every map.

        We use the function `pack_unique_rectangles` to provide a layout for
        the single map. This means that if self was created with a list of maps,
        and to() has not been called, and there were two maps which were exactly
        the same tensor object, then they will become the same data in the unified map.
        _place_map_into_single_map is used to copy the maps into the single map.
        The merging of verts_uvs and faces_uvs is handled locally in this function.
        """
        if self.maps_ids_padded() is not None:
            # TODO
            raise NotImplementedError("join_scene does not support multiple maps.")
        maps = self.maps_list()
        heights_and_widths = []
        extra_border = 0 if self.align_corners else 2
        for map_ in maps:
            heights_and_widths.append(
                Rectangle(
                    map_.shape[0] + extra_border, map_.shape[1] + extra_border, id(map_)
                )
            )
        merging_plan = pack_unique_rectangles(heights_and_widths)
        C = maps[0].shape[-1]
        single_map = maps[0].new_zeros((*merging_plan.total_size, C))
        verts_uvs = self.verts_uvs_list()
        verts_uvs_merged = []

        for map_, loc, uvs in zip(maps, merging_plan.locations, verts_uvs):
            new_uvs = uvs.clone()
            if loc.is_first:
                self._place_map_into_single_map(single_map, map_, loc)
            do_flip = loc.flipped
            x_shape = map_.shape[1] if do_flip else map_.shape[0]
            y_shape = map_.shape[0] if do_flip else map_.shape[1]

            if do_flip:
                # Here we have flipped / transposed the map.
                # In uvs, the y values are decreasing from 1 to 0 and the x
                # values increase from 0 to 1. We subtract all values from 1
                # as the x's become y's and the y's become x's.
                new_uvs = 1.0 - new_uvs[:, [1, 0]]
                if TYPE_CHECKING:
                    new_uvs = torch.Tensor(new_uvs)

            # If align_corners is True, then an index of x (where x is in
            # the range 0 .. map_.shape[1]-1) in one of the input maps
            # was hit by a u of x/(map_.shape[1]-1).
            # That x is located at the index loc[1] + x in the single_map, and
            # to hit that we need u to equal (loc[1] + x) / (total_size[1]-1)
            # so the old u should be mapped to
            #   { u*(map_.shape[1]-1) + loc[1] } / (total_size[1]-1)

            # Also, an index of y (where y is in
            # the range 0 .. map_.shape[0]-1) in one of the input maps
            # was hit by a v of 1 - y/(map_.shape[0]-1).
            # That y is located at the index loc[0] + y in the single_map, and
            # to hit that we need v to equal 1 - (loc[0] + y) / (total_size[0]-1)
            # so the old v should be mapped to
            #   1 - { (1-v)*(map_.shape[0]-1) + loc[0] } / (total_size[0]-1)
            # =
            # { v*(map_.shape[0]-1) + total_size[0] - map.shape[0] - loc[0] }
            #        / (total_size[0]-1)

            # If align_corners is False, then an index of x (where x is in
            # the range 1 .. map_.shape[1]-2) in one of the input maps
            # was hit by a u of (x+0.5)/(map_.shape[1]).
            # That x is located at the index loc[1] + 1 + x in the single_map,
            # (where the 1 is for the border)
            # and to hit that we need u to equal (loc[1] + 1 + x + 0.5) / (total_size[1])
            # so the old u should be mapped to
            #   { loc[1] + 1 + u*map_.shape[1]-0.5 + 0.5 } / (total_size[1])
            #  = { loc[1] + 1 + u*map_.shape[1] } / (total_size[1])

            # Also, an index of y (where y is in
            # the range 1 .. map_.shape[0]-2) in one of the input maps
            # was hit by a v of 1 - (y+0.5)/(map_.shape[0]).
            # That y is located at the index loc[0] + 1 + y in the single_map,
            # (where the 1 is for the border)
            # and to hit that we need v to equal 1 - (loc[0] + 1 + y + 0.5) / (total_size[0])
            # so the old v should be mapped to
            #   1 - { loc[0] + 1 + (1-v)*map_.shape[0]-0.5 + 0.5 } / (total_size[0])
            #  = { total_size[0] - loc[0] -1 - (1-v)*map_.shape[0]  }
            #         / (total_size[0])
            #  = { total_size[0] - loc[0] - map.shape[0] - 1 + v*map_.shape[0] }
            #         / (total_size[0])

            # We change the y's in new_uvs for the scaling of height,
            # and the x's for the scaling of width.
            # That is why the 1's and 0's are mismatched in these lines.
            one_if_align = 1 if self.align_corners else 0
            one_if_not_align = 1 - one_if_align
            denom_x = merging_plan.total_size[0] - one_if_align
            scale_x = x_shape - one_if_align
            denom_y = merging_plan.total_size[1] - one_if_align
            scale_y = y_shape - one_if_align
            new_uvs[:, 1] *= scale_x / denom_x
            new_uvs[:, 1] += (
                merging_plan.total_size[0] - x_shape - loc.x - one_if_not_align
            ) / denom_x
            new_uvs[:, 0] *= scale_y / denom_y
            new_uvs[:, 0] += (loc.y + one_if_not_align) / denom_y

            verts_uvs_merged.append(new_uvs)

        faces_uvs_merged = []
        offset = 0
        for faces_uvs_, verts_uvs_ in zip(self.faces_uvs_list(), verts_uvs):
            faces_uvs_merged.append(offset + faces_uvs_)
            offset += verts_uvs_.shape[0]

        return self.__class__(
            maps=[single_map],
            faces_uvs=[torch.cat(faces_uvs_merged)],
            verts_uvs=[torch.cat(verts_uvs_merged)],
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
            sampling_mode=self.sampling_mode,
        )

    def centers_for_image(self, index: int) -> torch.Tensor:
        """
        Return the locations in the texture map which correspond to the given
        verts_uvs, for one of the meshes. This is potentially useful for
        visualizing the data. See the texturesuv_image_matplotlib and
        texturesuv_image_PIL functions.

        Args:
            index: batch index of the mesh whose centers to return.

        Returns:
            centers: coordinates of points in the texture image
                - a FloatTensor of shape (V,2)
        """
        if self.maps_ids_padded() is not None:
            # TODO: invent a visualization for the multiple maps case
            raise NotImplementedError("This function does not support multiple maps.")
        if self._N != 1:
            raise ValueError(
                "This function only supports plotting textures for one mesh."
            )
        texture_image = self.maps_padded()
        verts_uvs = self.verts_uvs_list()[index][None]
        _, H, W, _3 = texture_image.shape
        coord1 = torch.arange(W).expand(H, W)
        coord2 = torch.arange(H)[:, None].expand(H, W)
        coords = torch.stack([coord1, coord2])[None]
        with torch.no_grad():
            # Get xy cartesian coordinates based on the uv coordinates
            centers = F.grid_sample(
                torch.flip(coords.to(texture_image), [2]),
                # Convert from [0, 1] -> [-1, 1] range expected by grid sample
                verts_uvs[:, None] * 2.0 - 1,
                mode=self.sampling_mode,
                align_corners=self.align_corners,
                padding_mode=self.padding_mode,
            ).cpu()
            centers = centers[0, :, 0].T
        return centers

    def check_shapes(
        self, batch_size: int, max_num_verts: int, max_num_faces: int
    ) -> bool:
        """
        Check if the dimensions of the verts/faces uvs match that of the mesh
        """
        # (N, F) should be the same
        # (N, V) is not guaranteed to be the same
        return (self.faces_uvs_padded().shape[0:2] == (batch_size, max_num_faces)) and (
            self.verts_uvs_padded().shape[0] == batch_size
        )

    def submeshes(
        self,
        vertex_ids_list: List[List[torch.LongTensor]],
        faces_ids_list: List[List[torch.LongTensor]],
    ) -> "TexturesUV":
        """
        Extract a sub-texture for use in a submesh.

        If the meshes batch corresponding to this  TexturesUV contains
        `n = len(faces_ids_list)` meshes, then self.faces_uvs_padded()
        will be of length n. After submeshing, we obtain a batch of
        `k = sum(len(f) for f in faces_ids_list` submeshes (see Meshes.submeshes). This
        function creates a corresponding  TexturesUV object with `faces_uvs_padded`
        of length `k`.

        Args:
            vertex_ids_list: Not used when submeshing TexturesUV.

            face_ids_list: A list of length equal to self.faces_uvs_padded. Each
                element is a LongTensor listing the face ids that the submesh keeps in
                each respective mesh.


        Returns:
            A  "TexturesUV in which faces_uvs_padded, verts_uvs_padded, and maps_padded
            have length sum(len(faces) for faces in faces_ids_list)
        """
        if self.maps_ids_padded() is not None:
            # TODO
            raise NotImplementedError("This function does not support multiple maps.")
        if len(faces_ids_list) != len(self.faces_uvs_padded()):
            raise IndexError(
                "faces_uvs_padded must be of " "the same length as face_ids_list."
            )

        sub_faces_uvs, sub_verts_uvs, sub_maps = [], [], []
        for faces_ids, faces_uvs, verts_uvs, map_ in zip(
            faces_ids_list,
            self.faces_uvs_padded(),
            self.verts_uvs_padded(),
            self.maps_padded(),
        ):
            for faces_ids_submesh in faces_ids:
                sub_faces_uvs.append(faces_uvs[faces_ids_submesh])
                sub_verts_uvs.append(verts_uvs)
                sub_maps.append(map_)

        return self.__class__(
            maps=sub_maps,
            faces_uvs=sub_faces_uvs,
            verts_uvs=sub_verts_uvs,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            sampling_mode=self.sampling_mode,
        )





































import torch
from pytorch3d.io import IO
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    BlendParams,
    HardPhongShader,
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
    def __init__(self, device="cpu", blend_params=None, texture=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.texture=texture
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

os.chdir("/home/thanos/OneDrive/mybackupOneDrive/PhD/3dif_objaverse-xl-subset/Minecraft Grass Block/") # Minecraft Grass Block, earth
mesh = IO().load_mesh("untitled.obj", device=device)


# os.chdir()
# mesh = IO().load_mesh("mq-9_reaper.obj", device=device)

# Define the parameters for different views (all lists have the same length)
d_list = [3.5-.8] #.6
e_list = [45-20+20] #40
a_list = [180-135+135] #180
desired_origin = torch.tensor([0, -.7-.5, 0.0]) #torch.tensor([-1.2, -.5, 0.0]) 
#
# d_list = [3.7, .6]
# e_list = [40, 40]
# a_list = [180, 180]
# desired_origin = torch.tensor([-1.2, -.5, 0.0]) 

# Define the resolutions
# 1024 for rgb (through the encoder) and depth
# 128 for the pixel_uvs_np in the latent space
resolutions = [128] # [1024, 128]

# Initialize an empty dictionary to store the results
views = {res: {'rgb_view': [], 'fragments': [], 'depth_map_255': [], 'pixel_uvs_np': [], 'd': [], 'e': [], 'a': []} for res in resolutions}

for d, e, a in zip(d_list, e_list, a_list):
    for res in resolutions:
        # Initialize the camera (R:rotation, T:translation)
        R, T = look_at_view_transform(dist=d, elev=e, azim=a)
        T = T + desired_origin.view(1, 3) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(image_size=res, blur_radius=0.0, faces_per_pixel=1) # 1024
        
        texture = TexturesUV(
            maps=mesh.textures.maps_padded(),
            faces_uvs=mesh.textures.faces_uvs_padded(),
            verts_uvs=mesh.textures.verts_uvs_padded()
        )
        
        # RGB Renderer
        rgb_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=NoLightShader(device=device, texture=texture))
        
        # Depth Renderer
        depth_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=PointLights(device=device)))
    
        # Render the RGB view
        rendered_view = rgb_renderer(mesh)
        
        # Get fragments for depth calculation
        fragments = rgb_renderer.rasterizer(mesh)
        depth_map = fragments.zbuf[0, ..., 0].cpu().numpy()
        
        # Normalize the depth map and convert to 8-bit (0-255) scale
        depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map_255 = (depth_map_normalized * 255).astype(np.uint8)
    
        # Get pixel UVs using the provided function
        # texture = TexturesUV(
        #     maps=mesh.textures.maps_padded(),
        #     faces_uvs=mesh.textures.faces_uvs_padded(),
        #     verts_uvs=mesh.textures.verts_uvs_padded()
        # )
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

def scale_coordinates(array, old_min=0, old_max=127, new_min=0, new_max=16383):
    scaled_array = ((array - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min
    return np.round(scaled_array).astype(np.int64)

# Plot each view's RGB view, depth map, and texture UV points
for res in views:
    for i in range(len(views[res]['rgb_view'])):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot RGB view
        ax1.imshow(views[res]['rgb_view'][i][0,:,:,:3].cpu().numpy())
        ax1.set_title(f'RGB View (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        #
        # ax1.set_xticks(range(0, views[res]['rgb_view'][i].shape[2], 1))  # Set x ticks to 1 pixel
        # ax1.set_yticks(range(0, views[res]['rgb_view'][i].shape[1], 1))  # Set y ticks to 1 pixel
        # ax1.grid(True, which='both', color='white', linewidth=0.5, linestyle='--')  # Add grid

        
        # Plot depth map
        ax2.imshow(views[res]['depth_map_255'][i], cmap='gray')
        ax2.set_title(f'Depth Map (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        
        # # Plot texture map with pixel UVs
        # ax3.scatter(views[res]['pixel_uvs_np'][i][:,:,1], views[res]['pixel_uvs_np'][i][:,:,0], c='blue', marker='x', s=5)
        # ax3.set_title(f'Texture Map with UV Points (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        #
        # Plot UV texture map with UV points overlay
        # if res != 1024:
        #     uv_texture = mesh.textures.maps_padded()[0]
        #     uv_texture_resized = F.interpolate(uv_texture[None].permute(0,-1,1,2), size=(res, res), mode='bilinear', align_corners=False)
        #     uv_texture_resized_np = uv_texture_resized.squeeze().permute(1, 2, 0).cpu().numpy()  # Shape [128, 128, 3]
        #     # uv_texture_resized_np = np.ones((uv_texture_resized_np.shape))
        #     ax3.imshow(uv_texture_resized_np)
        # else:
        #     uv_texture = mesh.textures.maps_padded()[0].cpu().numpy()  # Extract UV texture
        #     # uv_texture = np.ones((uv_texture.shape))
        #     ax3.imshow(uv_texture)
        # temp = views[res]['pixel_uvs_np'][i]
        #
        uv_texture = mesh.textures.maps_padded()[0].cpu().numpy()  # Extract UV texture
        ax3.imshow(uv_texture)
        temp = scale_coordinates(views[res]['pixel_uvs_np'][i], old_max=res-1, new_max=1023)
        #
        # Overlay pixel UV points on UV texture map
        ax3.scatter(temp[:,:,0], temp[:,:,1], c='blue', marker='x', s=5)
        ax3.set_title(f'UV Texture Map with UV Points (d={views[res]["d"][i]}, e={views[res]["e"][i]}, a={views[res]["a"][i]}, res={res})')
        #
        # ax3.set_xticks(range(0, res, 1))  # Set x ticks to 1 pixel
        # ax3.set_yticks(range(0, res, 1))  # Set y ticks to 1 pixel
        # ax3.grid(True, which='both', color='white', linewidth=0.5, linestyle='--')  # Add grid

        
        plt.show()



        
        
        
# %% just testing

# original one
def get_texels_for_pixel(fragments, texture, pixel_x, pixel_y):
    """
    Get the texels covered by a specific screen-space pixel with perspective correction
    """
    # Get the face ID for this pixel
    face_id = fragments.pix_to_face[0, pixel_y, pixel_x, 0]
    if face_id < 0:  # pixel doesn't hit any face
        return None
    
    # Get UV coordinates of the face vertices
    faces_uvs = texture.faces_uvs_list()[0]
    verts_uvs = texture.verts_uvs_list()[0]
    face_uvs = verts_uvs[faces_uvs[face_id]]  # Shape: [3, 2]
    
    # Define pixel corners in screen space (relative to pixel center)
    pixel_corners = torch.tensor([
        [-0.5, -0.5],  # top-left
        [0.5, -0.5],   # top-right
        [0.5, 0.5],    # bottom-right
        [-0.5, 0.5],   # bottom-left
    ], device=fragments.pix_to_face.device)
    
    # Get barycentric coordinates for each corner
    corners_uvs = []
    for dx, dy in pixel_corners:
        # Adjust position for the corner
        px = pixel_x + dx
        py = pixel_y + dy
        
        # Get interpolated barycentric coordinates
        if (px >= 0 and px < fragments.pix_to_face.shape[2] and 
            py >= 0 and py < fragments.pix_to_face.shape[1]):
            
            # Check if this corner hits the same face
            corner_face_id = fragments.pix_to_face[0, int(py), int(px), 0]
            if corner_face_id == face_id:  # Only use corners that hit the same face
                bary = fragments.bary_coords[0, int(py), int(px), 0]
                corner_uv = (face_uvs * bary.view(-1, 1)).sum(dim=0)
                
                # Convert to texture space coordinates
                corner_uv = torch.lerp(
                    corner_uv.new_tensor([-1.0, 1.0]),
                    corner_uv.new_tensor([1.0, -1.0]),
                    corner_uv,
                )
                
                corners_uvs.append(corner_uv)
            else:
                # If corner hits different face, use the center's UV coordinate
                bary = fragments.bary_coords[0, pixel_y, pixel_x, 0]
                corner_uv = (face_uvs * bary.view(-1, 1)).sum(dim=0)
                corner_uv = torch.lerp(
                    corner_uv.new_tensor([-1.0, 1.0]),
                    corner_uv.new_tensor([1.0, -1.0]),
                    corner_uv,
                )
                corners_uvs.append(corner_uv)
    
    if len(corners_uvs) < 3:  # Need at least 3 points for a valid polygon
        return None
        
    corners_uvs = torch.stack(corners_uvs)
    
    # Get texture dimensions
    H, W = texture.maps_padded().shape[1:3]
    
    # Convert normalized coordinates to pixel coordinates and clamp to texture bounds
    corners_texels = torch.zeros_like(corners_uvs)
    corners_texels[:, 0] = torch.clamp(((corners_uvs[:, 0] + 1) * (W - 1)) / 2, 0, W-1)
    corners_texels[:, 1] = torch.clamp(((corners_uvs[:, 1] + 1) * (H - 1)) / 2, 0, H-1)
    
    return {
        'corners_uvs': corners_uvs.cpu(),
        'corners_texels': corners_texels.cpu(),
    }



# has 1/z
def get_texels_for_pixel2(fragments, texture, pixel_x, pixel_y):
    """
    Get the texels covered by a specific screen-space pixel with perspective correction
    """
    # Get the face ID for this pixel
    face_id = fragments.pix_to_face[0, pixel_y, pixel_x, 0]
    if face_id < 0:
        return None
    
    # Get UV coordinates and z-depths
    faces_uvs = texture.faces_uvs_list()[0]
    verts_uvs = texture.verts_uvs_list()[0]
    face_uvs = verts_uvs[faces_uvs[face_id]]
    
    # Get z-depths from the fragments
    zbuf = fragments.zbuf[0, :, :, 0]  # [H, W]
    
    pixel_corners = torch.tensor([
        [-0.5, -0.5],  # top-left
        [0.5, -0.5],   # top-right
        [0.5, 0.5],    # bottom-right
        [-0.5, 0.5],   # bottom-left
    ], device=fragments.pix_to_face.device)
    
    corners_uvs = []
    for dx, dy in pixel_corners:
        px = pixel_x + dx
        py = pixel_y + dy
        
        if (px >= 0 and px < fragments.pix_to_face.shape[2] and 
            py >= 0 and py < fragments.pix_to_face.shape[1]):
            
            corner_face_id = fragments.pix_to_face[0, int(py), int(px), 0]
            if corner_face_id == face_id:
                # Get barycentric coordinates and z-depth
                bary = fragments.bary_coords[0, int(py), int(px), 0]
                z = zbuf[int(py), int(px)]
                
                # Perspective-correct the barycentric coordinates
                w = 1.0 / z
                bary_perspective = bary * w
                bary_perspective = bary_perspective / bary_perspective.sum()
                
                # Compute UV with perspective-correct interpolation
                corner_uv = (face_uvs * bary_perspective.view(-1, 1)).sum(dim=0)
                
                corner_uv = torch.lerp(
                    corner_uv.new_tensor([-1.0, 1.0]),
                    corner_uv.new_tensor([1.0, -1.0]),
                    corner_uv,
                )
                corners_uvs.append(corner_uv)
            else:
                # Use center's coordinates with perspective correction
                bary = fragments.bary_coords[0, pixel_y, pixel_x, 0]
                z = zbuf[pixel_y, pixel_x]
                w = 1.0 / z
                bary_perspective = bary * w
                bary_perspective = bary_perspective / bary_perspective.sum()
                
                corner_uv = (face_uvs * bary_perspective.view(-1, 1)).sum(dim=0)
                corner_uv = torch.lerp(
                    corner_uv.new_tensor([-1.0, 1.0]),
                    corner_uv.new_tensor([1.0, -1.0]),
                    corner_uv,
                )
                corners_uvs.append(corner_uv)
    
    if len(corners_uvs) < 3:
        return None
        
    corners_uvs = torch.stack(corners_uvs)
    
    H, W = texture.maps_padded().shape[1:3]
    corners_texels = torch.zeros_like(corners_uvs)
    corners_texels[:, 0] = torch.clamp(((corners_uvs[:, 0] + 1) * (W - 1)) / 2, 0, W-1)
    corners_texels[:, 1] = torch.clamp(((corners_uvs[:, 1] + 1) * (H - 1)) / 2, 0, H-1)
    
    return {
        'corners_uvs': corners_uvs.cpu(),
        'corners_texels': corners_texels.cpu(),
    }







def plot_all_pixel_footprints(fragments, texture):
    """
    Plot all pixel footprints on the UV map with random colors per pixel
    """
    # Get image dimensions
    H, W = fragments.pix_to_face.shape[1:3]
    
    # Create figure
    plt.figure(figsize=(15, 15))
    
    # Plot the texture map
    uv_texture = texture.maps_padded()[0].cpu().numpy()
    plt.imshow(uv_texture) 
    
    # Create random colors for each pixel
    np.random.seed(42)  # for reproducibility, remove if you want different colors each time
    colors = np.random.rand(H * W, 3)  # Random RGB values between 0 and 1
    
    # Plot footprint for each pixel
    pixel_idx = 0
    for y in range(H):
        for x in range(W):
            print(y,x)
            texel_info = get_texels_for_pixel(fragments, texture, x, y)
            if texel_info is not None:
                coords = texel_info['corners_texels']
                coords_closed = torch.cat([coords, coords[0:1]])
                plt.plot(coords_closed[:, 0], coords_closed[:, 1], 
                        color=colors[pixel_idx], 
                        alpha=0.91, 
                        linewidth=0.5)
            pixel_idx += 1
    
    plt.title('All Pixel Footprints in UV Space')
    plt.xlabel('Texture U coordinate')
    plt.ylabel('Texture V coordinate')
    plt.show()

# Call the function
plot_all_pixel_footprints(fragments, texture)









# def plot_pixel_texel_polygon(fragments, texture, pixel_x, pixel_y):
#     """
#     Plot the polygon formed by the texels that correspond to a screen-space pixel
#     """
#     texel_info = get_texels_for_pixel(fragments, texture, pixel_x, pixel_y)
    
#     if texel_info is None:
#         print(f"No face found at pixel ({pixel_x}, {pixel_y})")
#         return
    
#     # Get the texture map and move to CPU
#     uv_texture = texture.maps_padded()[0].cpu().numpy()
    
#     # Create figure
#     plt.figure(figsize=(10, 10))
    
#     # Plot the texture map
#     # plt.imshow(uv_texture)
    
#     # Get the texel coordinates and add the first point again to close the polygon
#     coords = texel_info['corners_texels']
#     coords_closed = torch.cat([coords, coords[0:1]])
    
#     # Plot the polygon










#     plt.plot(coords_closed[:, 0], coords_closed[:, 1], 'r-', linewidth=2, label='Pixel footprint')
    
#     # Add labels and title
#     plt.title(f'Texels covered by pixel ({pixel_x}, {pixel_y})')
#     plt.xlabel('Texture U coordinate')
#     plt.ylabel('Texture V coordinate')
#     plt.legend()
    
#     plt.show()

# x = 60
# y = 110

# plot_pixel_texel_polygon(fragments, texture, x, y)
# get_texels_for_pixel(fragments, texture, x, y)

 
















# %% testing....





























