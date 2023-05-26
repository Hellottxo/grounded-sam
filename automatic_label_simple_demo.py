import cv2
import numpy as np
import supervision as sv
from typing import List
from PIL import Image
import uuid
import torch

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

# Tag2Text
import sys
sys.path.append('Tag2Text')
from Tag2Text.models import tag2text
from Tag2Text import inference
import torchvision
import torchvision.transforms as TS

def segmentToimg(SOURCE_IMAGE_PATH):

    # Hyper-Params
    # SOURCE_IMAGE_PATH = "./outputs/raw_image.jpg"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

    TAG2TEXT_CHECKPOINT_PATH = "Tag2Text/tag2text_swin_14m.pth"

    TAG2TEXT_THRESHOLD = 0.64
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.2
    IOU_THRESHOLD = 0.5

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)


    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam_predictor = SamPredictor(sam)

    # Tag2Text
    # initialize Tag2Text
    normalize = TS.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = TS.Compose(
        [
            TS.Resize((384, 384)),
            TS.ToTensor(), 
            normalize
        ]
    )

    DELETE_TAG_INDEX = []  # filter out attributes and action which are difficult to be grounded
    for idx in range(3012, 3429):
        DELETE_TAG_INDEX.append(idx)

    tag2text_model = tag2text.tag2text_caption(
        pretrained=TAG2TEXT_CHECKPOINT_PATH,
        image_size=384,
        vit='swin_b',
        delete_tag_index=DELETE_TAG_INDEX
    )
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    tag2text_model.threshold = TAG2TEXT_THRESHOLD
    tag2text_model.eval()
    tag2text_model = tag2text_model.to(DEVICE)


    # load image
    image = cv2.cvtColor(SOURCE_IMAGE_PATH, cv2.COLOR_RGB2BGR)
    # image = SOURCE_IMAGE_PATH  # bgr
    image_pillow = Image.fromarray(image)  # rgb

    image_pillow = image_pillow.resize((384, 384))
    image_pillow = transform(image_pillow).unsqueeze(0).to(DEVICE)

    specified_tags='None'
    res = inference.inference(image_pillow , tag2text_model, specified_tags)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    AUTOMATIC_CLASSES=res[0].split(" | ")
    caption=res[2]

    print(f"Caption: {caption}")
    print(f"Tags: {res[0].replace(' |', ',')}")


    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=AUTOMATIC_CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        IOU_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    # annotate image with detections
    # box_annotator = sv.BoxAnnotator()
    # labels = [
    #     f"{AUTOMATIC_CLASSES[class_id]} {confidence:0.2f}" 
    #     for _, _, confidence, class_id, _ 
    #     in detections]
    # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # # save the annotated grounding dino image
    # cv2.imwrite("groundingdino_auto_annotated_image.jpg", annotated_frame)

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        
        # image_embedding = sam_predictor.get_image_embedding().cpu().numpy()
        # # 将 image_embedding 保存为 .npy 文件，用于segment官网选中示例
        # np.save('image_embedding.npy', image_embedding)

        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)


    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # 生成label
    labels = [
        f"{AUTOMATIC_CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]

    id = uuid.uuid1()
    res = []
    # 循环处理每个掩码并分割出物体并保存
    for i, mask in enumerate(detections.mask):
        # 将掩码应用于原始图像
        img = image.copy()
        mask = mask.astype(np.uint8) * 255


        masked_img = cv2.bitwise_and(img, img, mask=mask)

        # 将不属于目标像素的像素设置为透明色，保留目标像素
        rgba = cv2.cvtColor(masked_img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)[:, :, 0]
        # 保存结果图像
        filename = f"outputs/{labels[i]}_{id}.png"
        res.append(filename)
        cv2.imwrite(filename, rgba)
    return res

    # annotate image with detections
    # box_annotator = sv.BoxAnnotator()
    # mask_annotator = sv.MaskAnnotator()

    # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # save the annotated grounded-sam image
    # cv2.imwrite("grounded_sam_auto_annotated_image.jpg", annotated_image)

if __name__ == "__main__":
    segmentToimg("./outputs/raw_image.jpg")