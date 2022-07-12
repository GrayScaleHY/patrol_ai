# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
from inference.lib_inference  import load_predictor, inference
import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get configuration
        boxes_config = pb_utils.get_output_config_by_name(model_config, "detection_boxes")
        num_config = pb_utils.get_output_config_by_name(model_config, "num_detections")
        scores_config = pb_utils.get_output_config_by_name(model_config, "detection_scores")
        classes_config = pb_utils.get_output_config_by_name(model_config, "detection_classes")


        # Convert Triton types to numpy types
        self.boxes_dtype = pb_utils.triton_string_to_numpy(boxes_config["data_type"])
        self.num_dtype = pb_utils.triton_string_to_numpy(num_config["data_type"])
        self.scores_dtype = pb_utils.triton_string_to_numpy(scores_config["data_type"])
        self.classes_dtype = pb_utils.triton_string_to_numpy(classes_config["data_type"])

        # Instantiate the PyTorch model
        self.predictor = load_predictor()
        

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        boxes_dtype = self.boxes_dtype
        num_dtype = self.num_dtype
        scores_dtype = self.scores_dtype
        classes_dtype = self.classes_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "image")

            ## 图片前处理
            img = in_0.as_numpy().astype(np.uint8)
            img = img[0,:,:,:]

            ## 推理
            contours, boxes, (masks, classes) = inference(self.predictor, img)

            ## 对推理后的结果做后处理
            num = np.array([len(boxes)],dtype=np.int32)
            boxes = np.append(boxes, np.zeros([100-len(boxes), 4]), axis=0)
            scores = np.append(np.ones([len(classes)]), np.zeros([100-len(classes)]))
            classes = np.append(classes, np.zeros([100-len(classes)]))

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_num = pb_utils.Tensor("num_detections", num.astype(num_dtype))
            out_tensor_boxes = pb_utils.Tensor("detection_boxes", boxes.astype(boxes_dtype))
            out_tensor_scores = pb_utils.Tensor("detection_scores", scores.astype(scores_dtype))
            out_tensor_classes = pb_utils.Tensor("detection_classes", classes.astype(classes_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_boxes, out_tensor_scores, out_tensor_classes, out_tensor_num])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")