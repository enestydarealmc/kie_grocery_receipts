import cv2
import numpy as np
import math

def normalize_coord(bbox: list, width, height) -> list:
    x0, y0, x2, y2 = bbox

    x0 = int(1000 * (x0 / width))
    x2 = int(1000 * (x2 / width))
    y0 = int(1000 * (y0 / height))
    y2 = int(1000 * (y2 / height))

    return [x0, y0, x2, y2]

def normalize_bbox(x, y, box_width, box_height, width, height):

    x = int(x / 1000 * width)
    box_width = int(box_width / 1000 * width)
    y = int(y / 1000 * height)
    box_height = int(box_height / 1000 * height)

    return [x, y, box_width, box_height]

def unnormalize_coord(bbox, width, height):
    x0, y0, x2, y2 = bbox

    x0 = int(x0 / 1000 * width)
    x2 = int(x2 / 1000 * width)
    y0 = int(y0 / 1000 * height)
    y2 = int(y2 / 1000 * height)

    return [x0, y0, x2, y2]

def unnormalize_bbox(x, y, box_width, box_height, width, height):

    x = int(x / 100 * width)
    y = int(y / 100 * height)
    box_width = int(box_width / 100 * width)
    box_height = int(box_height / 100 * height)

    return [x, y, box_width, box_height]

def axis_align_bound_box(points):
    points = np.array(points, dtype=np.float32)
    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    
    return [x,y, x+w, y+h]

def split_rectangle(points, text, punctuation):
    # split text and get length
    left_text, right_text = text.split(punctuation, maxsplit=1)
    total_length = len(text.replace(' ', '').replace(punctuation, ''))
    left_length = len(left_text.replace(' ', ''))
    
    # calculate split ratio
    split_ratio = left_length / total_length
    
    # calculate split point
    p0, p1, p2, p3 = np.array(points)
    split_point = p0 + (p1 - p0) * split_ratio
    split_point_2 = p3 + (p2 - p3) * split_ratio
    
    left_points = np.array([p0.tolist(), split_point.tolist(), split_point_2.tolist(), p3.tolist()]).tolist()
    
    right_points = np.array([split_point.tolist(), p1.tolist(), p2.tolist(), split_point_2.tolist()]).tolist()
    
    # return split rectangles and texts
    return [left_points, left_text.strip()], [right_points, right_text.strip()]

def unit_vector(vector):
    if np.sum(np.abs(vector)) == 0:
        return vector
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def findClockwiseAngle(self, other):
    unit_self = unit_vector(self)
    unit_other = unit_vector(other)
    # using cross-product formula
    return -math.degrees(math.asin((unit_self[0] * unit_other[1] - unit_self[1] * unit_other[0])))

def four_points_to_oriented_topleft(points):
    """Convert rectangle format from array of points to [top-left point, width, height, angle] 

    Args:
        points (list): [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        
    Returns:
        _type_: [top-left point, width, height, angle] 
    """
    width = points[1][0] - points[0][0]
    height = points[3][1] - points[0][1]
    
    hor = (points[1][0]-points[0][0], points[1][1]-points[0][1])
    angle = findClockwiseAngle(hor, (1, 0))
    if -360 <= angle < 0:
        angle += 360
    elif angle < -360:
        # input points have invalid angle
        raise ArithmeticError
    
    return points[0], width, height, angle

def sort_rectangle_points(points):
    # get center point
    center_x = sum([p[0] for p in points]) / 4
    center_y = sum([p[1] for p in points]) / 4

    # determine relative positions of corners
    corners = np.array(points)
    for p in points:
        if p[0] < center_x and p[1] < center_y:
            corners[0] = p
        elif p[0] > center_x and p[1] < center_y:
            corners[1] = p
        elif p[0] > center_x and p[1] > center_y:
            corners[2] = p
        elif p[0] < center_x and p[1] > center_y:
            corners[3] = p
    
    return corners

def calculate_iou(box1, box2):
    x1_0, y1_0, x1_2, y1_2 = box1
    x2_0, y2_0, x2_2, y2_2 = box2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_intersection_0 = max(x1_0, x2_0)
    y_intersection_0 = max(y1_0, y2_0)
    x_intersection_2 = min(x1_2, x2_2)
    y_intersection_2 = min(y1_2, y2_2)

    # Compute the area of intersection rectangle
    intersection_area = max(0, x_intersection_2 - x_intersection_0 + 1) * max(0, y_intersection_2 - y_intersection_0 + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (x1_2 - x1_0 + 1) * (y1_2 - y1_0 + 1)
    box2_area = (x2_2 - x2_0 + 1) * (y2_2 - y2_0 + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    # Return the intersection over union value
    return iou

def compare_boxes(box1, box2, threshold=0.9):
    # Calculate the area of the two bounding boxes
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)

    # Calculate the intersection area of the two bounding boxes
    intersection_area = calculate_intersection_area(box1, box2)

    # Calculate the containment and matching ratios
    containment_ratio = intersection_area / min(area1, area2)
    matching_ratio = intersection_area / max(area1, area2)

    # Check if one bounding box contains the other
    # This must be checked first
    if containment_ratio > threshold:
        return "Containment"

    # Check if the two bounding boxes match each other
    if matching_ratio > threshold:
        return "Matching"

    # No containment or matching
    return "No match"

def calculate_area(box):
    # Extract the coordinates of the bounding box
    x0, y0, x2, y2 = box

    # Calculate the width and height of the bounding box
    width = x2 - x0
    height = y2 - y0

    # Calculate the area of the bounding box
    area = width * height

    return area

def calculate_intersection_area(box1, box2):
    # Find the intersection coordinates
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the width and height of the intersection
    width = max(0, x2 - x0)
    height = max(0, y2 - y0)

    # Calculate the intersection area
    area = width * height

    return area

def calculate_union_area(box1, box2):
    # Calculate the area of each bounding box
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)

    # Calculate the intersection area
    intersection_area = calculate_intersection_area(box1, box2)

    # Calculate the union area
    union_area = area1 + area2 - intersection_area

    return union_area
    
def calculate_union_area(boxes):
    union_area = 0
    for i, box in enumerate(boxes):
        # Calculate the area of the current bounding box
        area = calculate_area(box)
        
        # Subtract the intersection areas with previous boxes
        for prev_box in boxes[:i]:
            intersection_area = calculate_intersection_area(box, prev_box)
            area -= intersection_area
        
        # Add the area of the current box to the union area
        union_area += area

    return union_area

    
    
    
    
    
    
    
    
    
    