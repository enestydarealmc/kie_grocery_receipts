import json
import numpy as np
import sys
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

sys.path.append(r'/home/enestydarealmc/Git/Thesis/tools/')
from bbox_utils import unnormalize_bbox, clip_coordinates

# Define argument parser
parser = argparse.ArgumentParser(description='Convert Label Studio JSON-min format to Wildreceipt')
parser.add_argument('json_min_path', type=str, help='LB JSON-min path')
parser.add_argument('img_path_prefix', type=str, help='Prefix of path where the images are located')
parser.add_argument('output_path', type=int, help='output path of Wildreceipt format')
args = parser.parse_args()


indian2wildreceipt_text = {
    "Business_name_key": "Store_name_key",
    "Business_name_value": "Store_name_value",
    "CGST_key": "Tax_key",
    "CGST_value": "Tax_value",
    "Currency": "Others",
    "Date_key": "Date_key",
    "Date_value": "Date_value",
    "GSTIN_key": "Others",
    "GSTIN_value": "Others",
    "IGST_key": "Tax_key",
    "IGST_value": "Tax_value",
    "Prod_item_key": "Prod_item_key",
    "Prod_item_value": "Prod_item_value",
    "Prod_price_key": "Others",
    "Prod_price_value": "Others",
    "Prod_quantity_key": "Prod_quantity_key",
    "Prod_quantity_value": "Prod_quantity_value",
    "Prod_total_price_key": "Prod_price_key",
    "Prod_total_price_value": "Prod_price_value",
    "Receipt_borders": "Ignore",
    "SGST_key": "Tax_key",
    "SGST_value": "Tax_value",
    "Store_addr_key": "Store_addr_key",
    "Store_addr_value": "Store_addr_value",
    "Subtotal_key": "Subtotal_key",
    "Subtotal_value": "Subtotal_value",
    "Tel_key": "Tel_key",
    "Tel_value": "Tel_value",
    "Text": "Others",
    "Time_key": "Time_key",
    "Time_value": "Time_value",
    "Tips_key": "Tips_key",
    "Tips_value": "Tips_value",
    "Total_key": "Total_key",
    "Total_value": "Total_value",
    "Trade_name_key": "Store_name_key",
    "Trade_name_value": "Store_name_value"
}

wildreceipt_num2wildreceipt_text = {
        0: "Ignore",
        1: "Store_name_value",
        2: "Store_name_key",
        3: "Store_addr_value",
        4: "Store_addr_key",
        5: "Tel_value",
        6: "Tel_key",
        7: "Date_value",
        8: "Date_key",
        9: "Time_value",
        10: "Time_key",
        11: "Prod_item_value",
        12: "Prod_item_key",
        13: "Prod_quantity_value",
        14: "Prod_quantity_key",
        15: "Prod_price_value",
        16: "Prod_price_key",
        17: "Subtotal_value",
        18: "Subtotal_key",
        19: "Tax_value",
        20: "Tax_key",
        21: "Tips_value",
        22: "Tips_key",
        23: "Total_value",
        24: "Total_key",
        25: "Others"
    }

wildreceipt_text2wildreceipt_num = {value: key for key, value in wildreceipt_num2wildreceipt_text.items()}

json_min_path = args.json_min_path
indian_path_prefix = args.img_path_prefix
output_path = args.output_path

with open(json_min_path, "r") as f:
    receipts = json.load(f)

for receipt in receipts:
    img_name = receipt['ocr'].split('/')[-1]
    if img_name == '96.png':
        continue
    img_path = indian_path_prefix + r'img/' + img_name
    # image 
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # Create a figure and axis
    
    result = {"file_name": r'img/' + img_name,
                "height": height,
                "width": width,
                "annotations":[]}
    
    for bbox, text, label in zip(receipt['bbox'], receipt['transcription'], receipt['label']):
        if "border" in text.lower():
            continue
        x, y, box_width, box_height, rotation, box_label = bbox['x'], bbox['y'], bbox['width'], bbox['height'], np.deg2rad(bbox['rotation']), label['labels'][0]
        
        x, y, box_width, box_height = unnormalize_bbox(x, y, box_width, box_height, width, height)
        
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)

        
        # Rotate the points clockwise around the bottom-left point
        rotation_matrix = np.array([[cos_theta, sin_theta],
                                    [-sin_theta, cos_theta]])
        
        if 0 <= rotation < np.pi/2 or 3*np.pi/2 <= rotation < 2*np.pi:
            points = np.array([[x, y], [x + box_width, y], [x + box_width, y + box_height], [x, y + box_height]])
        elif np.pi <= rotation < 3*np.pi/2 or np.pi/2 <= rotation < np.pi:
            points = np.array([[x + box_width, y + box_height], [x, y + box_height], [x, y], [x + box_width, y]])

        points = np.dot(points - [x, y], rotation_matrix) + [x, y]
        
        points = clip_coordinates(points, width, height)
        
        points = points.astype(int)
        
        num_label = wildreceipt_text2wildreceipt_num[indian2wildreceipt_text[box_label]]
        
        # text = text.replace('\u00a', '')
        
        result["annotations"].append({"box": points.flatten().tolist(),
                                "text": text,
                                "label": num_label})
        
        # splitted_points = split_bbox(points, text, 0.25)
        
        # words = text.split()
        # num_label = wildreceipt_text2wildreceipt_num[indian2wildreceipt_text[box_label]]
        
        # for sub_word, sub_bbox in zip(words, splitted_points):
        #     result["annotations"].append({"box": sub_bbox.astype(int).flatten().tolist(),
        #                             "text": sub_word,
        #                             "label": num_label})
    with open(output_path, "a") as f:
            f.write(json.dumps(result)+"\n")
        
