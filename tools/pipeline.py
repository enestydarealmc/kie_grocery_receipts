import matplotlib.pyplot as plt
import re
import matplotlib.ticker as ticker
from paddleocr.paddleocr import PaddleOCR
import cv2
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Polygon
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

from transformers import AutoModelForTokenClassification
from transformers import AutoProcessor
import torch

from PIL import Image, ImageDraw, ImageFont
import sys

sys.path.append(r'/home/enestydarealmc/Git/Thesis/tools/')

from bbox_utils import normalize_coord, split_rectangle, unnormalize_coord, axis_align_bound_box

def iob_to_label(label):
    if not label:
      return 'other'
    return label

def OCR(ocr_model: PaddleOCR, img: np.ndarray):
    ocr_results = ocr_model.ocr(img)
    height, width = img.shape[:2]
    
    boxes = [line[0] for line in ocr_results[0]]
    txts = [str(line[1][0]) for line in ocr_results[0]]
    
    splitted_boxes = []
    splited_txts = []
    punctuation = ":"
    for box, text in zip(boxes, txts):
        if punctuation in text:
            left_text, right_text = text.split(punctuation, maxsplit=1)
            if left_text.strip() != '' and right_text.strip() != '':
                splited_box = split_rectangle(box, text, punctuation)
                
                splitted_boxes.append(splited_box[0][0])
                splited_txts.append(splited_box[0][1])
                splitted_boxes.append(splited_box[1][0])
                splited_txts.append(splited_box[1][1])
            else:
                splitted_boxes.append(box)
                splited_txts.append(text)
        else:
            splitted_boxes.append(box)
            splited_txts.append(text)
    
    # convert to axis-aligned boxes and normalize boxes 
    
    aligned_boxes = [axis_align_bound_box(box) for box in splitted_boxes]

    
    normalized_boxes = [normalize_coord(box, width, height) for box in aligned_boxes]
    
    
    return normalized_boxes, splited_txts

def inference(img_path, inference_model, processor, ocr_model, use_cuda = True):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    boxes, words = OCR(ocr_model, img)
    filler_tags = [0 for _ in range(len(boxes))]
    
    encoding = processor(img, [words], boxes=[boxes], word_labels=[filler_tags], return_tensors="pt", truncation=True).to("cuda")
    
    
    with torch.no_grad():
        outputs = inference_model(**encoding)
        
    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    labels = encoding.labels.squeeze().tolist()
    
    token_boxes = encoding.bbox.squeeze().tolist()
    
    label_preds = [inference_model.config.id2label[pred] for pred, label in zip(predictions, labels) if label != - 100]
    box_preds = [unnormalize_coord(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]
    
    return words, box_preds, label_preds

            
    