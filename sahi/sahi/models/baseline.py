# OBSS SAHI Tool
# Code written by Fatih C Akyon and Kadir Nar, 2021.

import logging
from typing import Any, List, Optional
import torch
import os, sys

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

ocsort_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "OC_SORT"))  # noqa
sys.path.append(ocsort_path)
from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, get_model_info, postprocess

logger = logging.getLogger(__name__)


class BaselineModel(DetectionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.nmsthre = 0.5

    def check_dependencies(self) -> None:
        check_requirements(["torch"])

    def load_model(self):
        print("Already loaded when initializing")
        pass

    def set_model(self, model: Any):
        """
        Sets the underlying TorchVision model.
        Args:
            model: Any
                A TorchVision model
        """
        # check_requirements(["torch", "torchvision"])

        model.eval()
        self.model = model.to(self.device)

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")


        if image.shape != (640, 640, 3):
            self._original_predictions = torch.zeros((1, 0, 6)).to(self.device)
            self._original_shape = image.shape
            return

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        # convert np image to cv2 format
        
        image, ratio = preproc(image, (height, width), self.rgb_means, self.std)
        # img_info["ratio"] = ratio
        image = torch.from_numpy(image).unsqueeze(0).float().to(self.device)
        # if self.fp16:
        #     image = image.half()  # to FP16
        
        with torch.no_grad():
            outputs = self.model(image)
            # print(f"outputs= {outputs}")
            # if self.decoder is not None:
            #     outputs = self.decoder(outputs, dtype=outputs.type())
            # print(f"outputs= {outputs[:, 4] > self.confthre}")
            post_outputs = postprocess(
                outputs, self.num_categories, self.confidence_threshold, self.nmsthre
            )
            # print(f"outputs= {outputs[0].cpu().numpy()[0]}")
        
        if post_outputs[0] is None:
            prediction = outputs[0]
        else:
            prediction = torch.cat([
                post_outputs[0][:, :4],
                post_outputs[0][:, 4].unsqueeze(-1),
                torch.ones_like(post_outputs[0][:, 4]).unsqueeze(-1),
            ], dim=-1)
        prediction = prediction.unsqueeze(0)  # [1, num_queries, 6]
        # print(f"prediction= {prediction.shape}")

        self._original_predictions = prediction
        self._original_shape = image.shape

    def convert_original_predictions(
        self,
        shift_amount: Optional[List[List[int]]] = [[0, 0]],
        full_shape: Optional[List[List[int]]] = None,
    ):
        """
        Converts original predictions of the detection model to a list of
        prediction.ObjectPrediction object. Should be called after perform_inference().
        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        self._create_object_prediction_list_from_original_predictions(
            shift_amount_list=shift_amount,
            full_shape_list=full_shape,
        )
        # if self.category_remapping:
        #     self._apply_category_remapping()

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatibility for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []

        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # Extract boxes and optional masks/obb
            boxes = image_predictions.data.cpu().detach().numpy()
            masks_or_points = None

            # Process each prediction
            for pred_ind, prediction in enumerate(boxes):
                # Get bbox coordinates
                bbox = prediction[:4].tolist()
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = 'bird'

                # Fix box coordinates
                bbox = [max(0, coord) for coord in bbox]
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # Ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                # Get segmentation or OBB points
                segmentation = None
                if masks_or_points is not None:
                    if self.has_mask:
                        bool_mask = masks_or_points[pred_ind]
                        # Resize mask to original image size
                        bool_mask = cv2.resize(
                            bool_mask.astype(np.uint8), (self._original_shape[1], self._original_shape[0])
                        )
                        segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
                    else:  # is_obb
                        obb_points = masks_or_points[pred_ind]  # Get OBB points for this prediction
                        segmentation = [obb_points.reshape(-1).tolist()]

                    if len(segmentation) == 0:
                        continue

                # Create and append object prediction
                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    segmentation=segmentation,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=self._original_shape[:2] if full_shape is None else full_shape,  # (height, width)
                )
                object_prediction_list.append(object_prediction)

            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

    @property
    def object_prediction_list(self) -> List[List[ObjectPrediction]]:
        if self._object_prediction_list_per_image is None:
            return []
        if len(self._object_prediction_list_per_image) == 0:
            return []
        return self._object_prediction_list_per_image[0]

    @property
    def object_prediction_list_per_image(self) -> List[List[ObjectPrediction]]:
        return self._object_prediction_list_per_image or []

    @property
    def original_predictions(self):
        return self._original_predictions


    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return 1

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return False

    @property
    def category_names(self):
        return list(self.category_mapping.values())