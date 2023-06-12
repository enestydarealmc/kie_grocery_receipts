import os
import json
from PIL import Image
from pathlib import Path
from uuid import uuid4
from paddleocr.paddleocr import PaddleOCR
import cv2
import argparse
import sys
from bbox_utils import four_points_to_oriented_topleft

# Append utils path to be located by Python
sys.path.append(r'/home/enestydarealmc/Git/Thesis/tools/')

# Define argument parser
parser = argparse.ArgumentParser(description='Pre annotate Indian Receipt Images with PaddleOCR for post-annotation correction.')
parser.add_argument('indian_img_path', type=str, help='Indian receipt images path')
parser.add_argument('port', type=int, help='port to serve Indian images in final output')
parser.add_argument('output', type=int, help='output Label Studio Json format for OCR task')
args = parser.parse_args()

indian_img_path = args.indian_img_path
port = args.port
output = args.output

def create_image_url(filepath):
    """
    Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
    if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8082/filename.png
    Otherwise you can build links like /data/upload/filename.png to refer to the files
    """
    filename = os.path.basename(filepath)
    return f'http://localhost:{port}/{filename}'


def convert_to_ls(image_path, model, thresh_hold=0.5):
    """
    :param image: PIL image object
    :param tesseract_output: the output from tesseract
    :param per_level: control the granularity of bboxes from tesseract
    :return: tasks.json ready to be imported into Label Studio with "Optical Character Recognition" template
    """
    
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    result = model.ocr(image)
    paddle_boxes = [line[0] for line in result[0]]
    txts = [str(line[1][0]) for line in result[0]]
    scores = [line[1][1] for line in result[0]]
    results = []
    all_scores = []
    for text, paddle_box, score in zip(txts, paddle_boxes, scores):
        (x,y), box_width, box_height, angle = four_points_to_oriented_topleft(paddle_box)
        bbox = {
            'x': 100 * x / image_width,
            'y': 100 * y / image_height,
            'width': 100 * box_width / image_width,
            'height': 100 * box_height / image_height,
            'rotation': angle
        }

        text = text.strip()

        if not text or score < thresh_hold:
            continue
        region_id = str(uuid4())[:10]
        bbox_result = {
            'id': region_id, 'from_name': 'bbox', 'to_name': 'image', 'type': 'rectangle',
            'value': bbox}
        transcription_result = {
            'id': region_id, 'from_name': 'transcription', 'to_name': 'image', 'type': 'textarea',
            'value': dict(text=[text], **bbox), 'score': score}
        results.extend([bbox_result, transcription_result])
        all_scores.append(score)

    return {
        'data': {
            'ocr': create_image_url(os.path.basename(image_path))
        },
        'predictions': [{
            'result': results,
            'score': sum(all_scores) / len(all_scores) if all_scores else 0
        }]
    }

tasks = []


# collect the receipt images from the image directory

model = PaddleOCR(use_angle_cls=True, lang='en',
                  cls_image_shape='3, 70, 200', cls_thresh=0.75,
                  structure_version='PP-Structure',
                  use_gpu=False,
                  det_pse_box_thresh = 0.75, 
                  det_limit_side_len = 4000,
                  det_box_type='quad',
                  use_dilation=True, use_tensorrt=True, image_orientation=True, 
                  det_db_score_mode='slow', use_pse=True,
                  det_db_box_thresh=0.3,
                  det_db_thresh=0.3, det_db_unclip_ratio=1.8,
                  drop_score=0.5, text_thresh=0.5
                  )

for f in Path(indian_img_path).glob('*'):
    # track down progress
    print(f)
    task = convert_to_ls(str(f), model)
    tasks.append(task)

# create a file to import into Label Studio
with open(output, mode='w') as f:
    json.dump(tasks, f, indent=2)
