import cv2
import numpy as np
import onnxruntime as ort
import albumentations as A
from scipy.spatial import distance
from albumentations.pytorch import ToTensorV2

class FeatureExtraction:
    def __init__(
        self,
        onnx_path="./osnet/osnet_x1_0.onnx",
        device="cpu",
    ):
        self.onnx_path = onnx_path
        self.device = device
        if self.device.lower() == "cpu":
            if ort.get_device() == "CUDA":
                print("CUDA available, if you want to switch your device into CUDA")
            self.ort_session = ort.InferenceSession(
                self.onnx_path, providers=["CPUExecutionProvider"]
            )
        elif self.device.lower() == "cuda":
            self.ort_session = ort.InferenceSession(
                self.onnx_path,
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
            )
        else:
            raise ValueError("Choose between CPU or CUDA!")
        self.model_height, self.model_width = self.ort_session.get_inputs()[0].shape[
            2:4
        ]
        self.image_augmentation = A.Compose(
            [A.Resize(self.model_height, self.model_width), A.Normalize(), ToTensorV2()]
        )

    def prdict_multi_input(self, imgs, id):
        preds = np.zeros((1, 512))
        
        for img in imgs:
            pred = self.predict_img(img)[0]
            preds = np.concatenate((preds, pred), axis=0)
            
        if len(preds) == 1:
            return ([], [])
        else : 
            return (preds[1:,], id)

    def predict_img(self, img):
        image = self._preprocessing_img(img)
        input_onnx = self.ort_session.get_inputs()[0].name
        output_onnx = self.ort_session.run(None, {input_onnx: image})
        return output_onnx

    def _preprocessing_img(self, img):
        image = np.array(img, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_augmentation(image=np.array(image))["image"]
        image = np.expand_dims(image, axis=0)
        return image
        