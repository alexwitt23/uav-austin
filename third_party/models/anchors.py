""" """

from typing import List

import torch
import numpy as np


class AnchorGenerator(torch.nn.Module):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        pyramid_levels: List[int],
        anchor_scales: List[float],
        use_cuda: bool = False,
    ):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.cuda = use_cuda
        self.aspect_ratios = [0.5, 1, 2]
        self.sizes = [2 ** (level + 2) for level in pyramid_levels]
        self.strides = [2 ** level for level in pyramid_levels]
        self.anchors_per_cell = self._generate_cell_anchors(
            self.sizes, self.aspect_ratios, anchor_scales
        )
        # Get the number of anchors in each cell
        self.num_anchors_per_cell = len(anchor_scales) * len(self.aspect_ratios)

        grid_sizes = np.array(
            [[np.ceil(img_height / s), np.ceil(img_width / s)] for s in self.strides]
        )
        self.anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        self.all_anchors = torch.cat(self.anchors_over_all_feature_maps)
        if use_cuda:
            self.all_anchors = self.all_anchors.cuda()

    def _generate_cell_anchors(
        self,
        anchor_sizes: List[int],
        aspect_ratios: List[int],
        anchor_scales: List[int],
    ) -> List[torch.Tensor]:
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).
        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):
        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """
        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors: List[torch.Tensor] = []
        # Generate anchors for each feature pyramid
        for anchor_size in anchor_sizes:
            pyramid_anchors = []
            for anchor_scale in anchor_scales:
                # The area of the box
                area = (anchor_size * anchor_scale) ** 2
                for aspect_ratio in aspect_ratios:
                    # s * s = w * h
                    # a = h / w
                    # ... some algebra ...
                    # w = sqrt(s * s / a)
                    # h = a * w
                    w = np.sqrt(area / aspect_ratio)
                    h = aspect_ratio * w
                    x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                    pyramid_anchors.append([x0, y0, x1, y1])

            anchors.append(torch.Tensor(pyramid_anchors).float())

        return anchors

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.anchors_per_cell
        ):
            shift_x, shift_y = self._create_grid_offsets(size, stride, 0.5, self.cuda)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def _create_grid_offsets(
        self, size: List[int], stride: int, offset: float, cuda: bool
    ):
        grid_height, grid_width = size
        shifts_x = torch.arange(
            offset * stride,
            grid_width * stride,
            step=stride,
            dtype=torch.float32,
            device="cpu" if cuda else "cpu",
        )
        shifts_y = torch.arange(
            offset * stride,
            grid_height * stride,
            step=stride,
            dtype=torch.float32,
            device="cpu" if cuda else "cpu",
        )

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        return shift_x, shift_y
