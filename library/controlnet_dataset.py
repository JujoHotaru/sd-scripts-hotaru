"""ControlNet dataset: paired training images and conditioning (control) images.

Extracted from ``library.train_util`` as part of the dataset-split refactor;
imports the abstract :class:`~library.dataset.BaseDataset` and its
:class:`~library.subset.ControlNetSubset` configuration class.
"""

import logging
import os

import numpy as np

from typing import Any, List, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator

from library.dataset import (
    IMAGE_TRANSFORMS,
    BaseDataset,
    glob_images,
    load_image,
)
from library.dreambooth_dataset import DreamBoothDataset
from library.subset import ControlNetSubset, DreamBoothSubset
from library.utils import resize_image, setup_logging, trim_and_resize_if_required

setup_logging()
logger = logging.getLogger(__name__)


class ControlNetDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[ControlNetSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        train_inpainting: bool,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str] = None,
        skip_image_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            resolution,
            network_multiplier,
            train_inpainting,
            debug_dataset,
            resize_interpolation,
            skip_image_resolution,
        )

        db_subsets = []
        for subset in subsets:
            assert (
                not subset.random_crop
            ), "random_crop is not supported in ControlNetDataset / random_cropはControlNetDatasetではサポートされていません"
            db_subset = DreamBoothSubset(
                subset.image_dir,
                False,
                None,
                subset.caption_extension,
                subset.cache_info,
                False,
                subset.num_repeats,
                subset.shuffle_caption,
                subset.caption_separator,
                subset.keep_tokens,
                subset.keep_tokens_separator,
                subset.secondary_separator,
                subset.enable_wildcard,
                subset.color_aug,
                subset.flip_aug,
                subset.face_crop_aug_range,
                subset.random_crop,
                subset.caption_dropout_rate,
                subset.caption_dropout_every_n_epochs,
                subset.caption_tag_dropout_rate,
                subset.caption_prefix,
                subset.caption_suffix,
                subset.token_warmup_min,
                subset.token_warmup_step,
                resize_interpolation=subset.resize_interpolation,
            )
            db_subsets.append(db_subset)

        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            True,
            batch_size,
            resolution,
            network_multiplier,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            1.0,
            train_inpainting,
            debug_dataset,
            validation_split,
            validation_seed,
            resize_interpolation,
            skip_image_resolution,
        )

        # config_util等から参照される値をいれておく（若干微妙なのでなんとかしたい）
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.controlnet_subsets = subsets
        self.batch_size = batch_size
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.validation_split = validation_split
        self.validation_seed = validation_seed
        self.resize_interpolation = resize_interpolation

        # assert all conditioning data exists
        missing_imgs = []
        cond_imgs_with_pair = set()
        for image_key, info in self.dreambooth_dataset_delegate.image_data.items():
            db_subset = self.dreambooth_dataset_delegate.image_to_subset[image_key]
            subset = None
            for s in subsets:
                if s.image_dir == db_subset.image_dir:
                    subset = s
                    break
            assert subset is not None, "internal error: subset not found"

            if not os.path.isdir(subset.conditioning_data_dir):
                logger.warning(f"not directory: {subset.conditioning_data_dir}")
                continue

            img_basename = os.path.splitext(os.path.basename(info.absolute_path))[0]
            ctrl_img_path = glob_images(subset.conditioning_data_dir, img_basename)
            if len(ctrl_img_path) < 1:
                missing_imgs.append(img_basename)
                continue
            ctrl_img_path = ctrl_img_path[0]
            ctrl_img_path = os.path.abspath(ctrl_img_path)  # normalize path

            info.cond_img_path = ctrl_img_path
            cond_imgs_with_pair.add(os.path.splitext(ctrl_img_path)[0])  # remove extension because Windows is case insensitive

        extra_imgs = []
        for subset in subsets:
            conditioning_img_paths = glob_images(subset.conditioning_data_dir, "*")
            conditioning_img_paths = [os.path.abspath(p) for p in conditioning_img_paths]  # normalize path
            extra_imgs.extend([p for p in conditioning_img_paths if os.path.splitext(p)[0] not in cond_imgs_with_pair])

        assert (
            len(missing_imgs) == 0
        ), f"missing conditioning data for {len(missing_imgs)} images / 制御用画像が見つかりませんでした: {missing_imgs}"
        if len(extra_imgs) > 0:
            logger.warning(f"extra conditioning data for {len(extra_imgs)} images / 余分な制御用画像があります: {extra_imgs}")

        self.conditioning_image_transforms = IMAGE_TRANSFORMS

    def get_conditioning_image_tensor(self, image_info, target_size_hw, original_size_hw, flipped):
        cond_img = load_image(image_info.cond_img_path)

        if self.dreambooth_dataset_delegate.enable_bucket:
            assert (
                cond_img.shape[0] == original_size_hw[0] and cond_img.shape[1] == original_size_hw[1]
            ), f"size of conditioning image is not match / 画像サイズが合いません: {image_info.absolute_path}"
            cond_img = resize_image(
                cond_img,
                original_size_hw[1],
                original_size_hw[0],
                target_size_hw[1],
                target_size_hw[0],
                self.resize_interpolation,
            )
            height, width = target_size_hw
            top = (cond_img.shape[0] - height) // 2
            left = (cond_img.shape[1] - width) // 2
            cond_img = cond_img[top : top + height, left : left + width]
        elif cond_img.shape[0] != target_size_hw[0] or cond_img.shape[1] != target_size_hw[1]:
            cond_img = resize_image(
                cond_img,
                cond_img.shape[1],
                cond_img.shape[0],
                target_size_hw[1],
                target_size_hw[0],
                self.resize_interpolation,
            )

        if flipped:
            cond_img = cond_img[:, ::-1, :].copy()
        return self.conditioning_image_transforms(cond_img)

    def align_addift_mask(self, mask_img, target_size_hw, original_size_hw, flipped):
        if mask_img.ndim == 3:
            mask_img = mask_img[:, :, 0]
        if self.dreambooth_dataset_delegate.enable_bucket:
            mask_img = resize_image(
                mask_img[:, :, None],
                original_size_hw[1],
                original_size_hw[0],
                target_size_hw[1],
                target_size_hw[0],
                self.resize_interpolation,
            )
            if mask_img.ndim == 3:
                mask_img = mask_img[:, :, 0]
            height, width = target_size_hw
            top = (mask_img.shape[0] - height) // 2
            left = (mask_img.shape[1] - width) // 2
            mask_img = mask_img[top : top + height, left : left + width]
        elif mask_img.shape[0] != target_size_hw[0] or mask_img.shape[1] != target_size_hw[1]:
            mask_img = resize_image(
                mask_img[:, :, None],
                mask_img.shape[1],
                mask_img.shape[0],
                target_size_hw[1],
                target_size_hw[0],
                self.resize_interpolation,
            )
            if mask_img.ndim == 3:
                mask_img = mask_img[:, :, 0]
        if flipped:
            mask_img = mask_img[:, ::-1].copy()
        return mask_img.astype(np.float32) / 255.0

    def load_addift_file_mask(self, mask_path, target_size_hw, original_size_hw, flipped):
        return self.align_addift_mask(load_image(mask_path), target_size_hw, original_size_hw, flipped)

    def load_addift_alpha_mask(self, image_path, target_size_hw, original_size_hw, flipped):
        image = load_image(image_path, True)
        if image.ndim == 3 and image.shape[2] >= 4:
            mask_img = image[:, :, 3]
        else:
            mask_img = np.full(image.shape[:2], 255, dtype=np.uint8)
        return self.align_addift_mask(mask_img, target_size_hw, original_size_hw, flipped)

    def get_addift_alpha_mask(self, image_info, mode, target_size_hw, original_size_hw, flipped):
        target_mask = None
        source_mask = None
        if mode in ("target", "union", "intersection", "difference"):
            target_mask = self.load_addift_alpha_mask(image_info.absolute_path, target_size_hw, original_size_hw, flipped)
        if mode in ("source", "union", "intersection", "difference"):
            source_mask = self.load_addift_alpha_mask(image_info.cond_img_path, target_size_hw, original_size_hw, flipped)
        if mode == "target":
            return target_mask
        if mode == "source":
            return source_mask
        if mode == "union":
            return np.maximum(target_mask, source_mask)
        if mode == "intersection":
            return np.minimum(target_mask, source_mask)
        if mode == "difference":
            return np.abs(target_mask - source_mask)
        raise ValueError(f"Unknown ADDifT alpha mask mode: {mode}")
    def set_current_strategies(self):
        return self.dreambooth_dataset_delegate.set_current_strategies()

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        return self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)

    def new_cache_text_encoder_outputs(self, models: List[Any], is_main_process: bool):
        return self.dreambooth_dataset_delegate.new_cache_text_encoder_outputs(models, is_main_process)

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]
        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        conditioning_images = []
        cached_conditioning_latents = []
        addift_masks = []
        pair_weights = []
        pair_multipliers = []
        pair_reverse_weights = []
        pair_reverse_multipliers = []

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]
            target_size_hw = example["target_sizes_hw"][i]
            original_size_hw = example["original_sizes_hw"][i]
            flipped = example["flippeds"][i]

            if hasattr(image_info, "addift_conditioning_latents_npz"):
                with np.load(image_info.addift_conditioning_latents_npz) as data:
                    key = "latents_flipped" if flipped else "latents"
                    cached_conditioning_latents.append(torch.from_numpy(data[key].copy()).float())
            else:
                conditioning_images.append(
                    self.get_conditioning_image_tensor(image_info, target_size_hw, original_size_hw, flipped)
                )

            mask_path = getattr(image_info, "addift_mask_path", None)
            alpha_mask_mode = getattr(image_info, "addift_alpha_mask", None)
            if mask_path is not None:
                mask = self.load_addift_file_mask(mask_path, target_size_hw, original_size_hw, flipped)
                addift_masks.append(torch.from_numpy(mask).float().unsqueeze(0))
            elif alpha_mask_mode is not None:
                mask = self.get_addift_alpha_mask(
                    image_info, alpha_mask_mode, target_size_hw, original_size_hw, flipped
                )
                addift_masks.append(torch.from_numpy(mask).float().unsqueeze(0))

            pair_weights.append(getattr(image_info, "addift_pair_weight", 1.0))
            pair_multipliers.append(getattr(image_info, "addift_pair_multiplier", 1.0))
            pair_reverse_weights.append(getattr(image_info, "addift_pair_reverse_weight", 1.0))
            pair_reverse_multipliers.append(getattr(image_info, "addift_pair_reverse_multiplier", -1.0))

        if cached_conditioning_latents:
            assert (
                len(cached_conditioning_latents) == bucket_batch_size and not conditioning_images
            ), "mixed cached and uncached ADDifT conditioning latents in a batch"
            example["addift_conditioning_latents"] = torch.stack(cached_conditioning_latents)
        else:
            example["conditioning_images"] = torch.stack(conditioning_images).to(
                memory_format=torch.contiguous_format
            ).float()

        example["addift_pair_weights"] = torch.tensor(pair_weights, dtype=torch.float32)
        example["addift_pair_multipliers"] = torch.tensor(pair_multipliers, dtype=torch.float32)
        example["addift_pair_reverse_weights"] = torch.tensor(pair_reverse_weights, dtype=torch.float32)
        example["addift_pair_reverse_multipliers"] = torch.tensor(pair_reverse_multipliers, dtype=torch.float32)
        if addift_masks:
            assert len(addift_masks) == bucket_batch_size, "missing ADDifT masks in a batch"
            example["addift_masks"] = torch.stack(addift_masks)

        return example