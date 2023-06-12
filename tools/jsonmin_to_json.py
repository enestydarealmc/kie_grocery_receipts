import json
import uuid
import argparse

# Define argument parser
parser = argparse.ArgumentParser(description='Transform JSON-min to JSON.')
parser.add_argument('input_json_min', type=str, help='Input JSON-min file path')
parser.add_argument('output_json', type=str, help='Output JSON file path')
args = parser.parse_args()

with open(args.input_json_min, "r") as f:
    receipts = json.load(f)
    
# Convert Label Studio JSON-min annotation format to JSON format

label_studio_data = []

for i, receipt in enumerate(receipts):
    label_studio_data.append(
        {
            "id": i,
            "annotations": [
                {
                    "id": i,
                    "completed_by": 1,
                    "result": [
                        # TODO: to be added more results
                    ],
                    # TODO: append to result
                    "was_cancelled": False,
                    "ground_truth": False,
                    "created_at": "2023-04-11T19:48:43.724764Z",
                    "updated_at": "2023-04-11T19:48:43.724785Z",
                    "lead_time": 44.464,
                    "prediction": {},
                    "result_count": 0,
                    "unique_id": str(uuid.uuid4())[:10],
                    "last_action": None,
                    "task": i,
                    "project": 10,
                    "updated_by": 1,
                    "parent_prediction": None,
                    "parent_annotation": None,
                    "last_created_by": None,
                }
            ],
            "file_upload": f"aibek.json",
            "drafts": [],
            "predictions": [],
            "data": {"ocr": receipt['ocr']},
            "meta": {},
            "created_at": "2023-04-11T19:43:52.168320Z",
            "updated_at": "2023-04-11T19:48:43.754831Z",
            "inner_id": i,
            "total_annotations": 1,
            "cancelled_annotations": 0,
            "total_predictions": 0,
            "comment_count": 0,
            "unresolved_comment_count": 0,
            "last_comment_updated_at": None,
            "project": 10,
            "updated_by": 1,
            "comment_authors": [],
        }
        )

    for bbox, trans, label in zip(receipt['bbox'], receipt['transcription'], receipt['label']):
        # TODO: id and inner_id are the same
        annotation_id = str(uuid.uuid4())[:10]
        cur_annotation = [
            {
                            "original_width": bbox['original_width'],
                            "original_height": bbox['original_height'],
                            "image_rotation": 0,
                            "value": {
                                "x": bbox['x'],
                                "y": bbox['y'],
                                "width": bbox['width'],
                                "height": bbox['height'],
                                "rotation": bbox['rotation'],
                            },
                            "id": annotation_id,
                            "from_name": "bbox",
                            "to_name": "image",
                            "type": "rectangle",
                            "origin": "manual",
                        },
                        {
                            "original_width": bbox['original_width'],
                            "original_height": bbox['original_height'],
                            "image_rotation": 0,
                            "value": {
                                "x": bbox['x'],
                                "y": bbox['y'],
                                "width": bbox['width'],
                                "height": bbox['height'],
                                "rotation": bbox['rotation'],
                                "labels": label['labels'],
                            },
                            "id": annotation_id,
                            "from_name": "label",
                            "to_name": "image",
                            "type": "labels",
                            "origin": "manual",
                        },
                        {
                            "original_width": bbox['original_width'],
                            "original_height": bbox['original_height'],
                            "image_rotation": 0,# Convert annotations to Label Studio JSON format
                            "value": {
                                "x": bbox['x'],
                                "y": bbox['y'],
                                "width": bbox['width'],
                                "height": bbox['height'],
                                "rotation": bbox['rotation'],
                                "text": [trans],
                            },
                            "id": annotation_id,
                            "from_name": "transcription",
                            "to_name": "image",
                            "type": "textarea",
                            "origin": "manual",
                        }
        ]

        label_studio_data[i]["annotations"][0]["result"].extend(cur_annotation)

# Save output
with open(args.output_json, "w") as f:
    json.dump(label_studio_data, f, indent=4)