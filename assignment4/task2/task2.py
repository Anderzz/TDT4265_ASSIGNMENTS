import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x1 = max(prediction_box[0], gt_box[0])
    y1 = max(prediction_box[1], gt_box[1])
    x2 = min(prediction_box[2], gt_box[2])
    y2 = min(prediction_box[3], gt_box[3])

    # Compute intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)  

    box1_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    box2_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    # Compute union
    union = box1_area + box2_area - intersection  

    iou = intersection / union

    #END OF YOUR CODE

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    # YOUR CODE HERE
    return num_tp / (num_tp + num_fp) if num_tp + num_fp != 0 else 1


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    # YOUR CODE HERE
    return num_tp / (num_tp + num_fn) if num_tp + num_fn != 0 else 0


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # YOUR CODE HERE
    # Find all possible matches with an IoU >= iou_threshold
    matches = []
    for i, pred_box in enumerate(prediction_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                matches.append((i, j, iou))
    if not matches:
        return np.array([]), np.array([])

    # Sort all matches on IoU in descending order
    matches.sort(key=lambda x: x[2], reverse=True)

    # Find all matches with the highest IoU threshold
    seen_pred = set()
    seen_gt = set()
    final_pred_boxes = []
    final_gt_boxes = []
    for i, j, iou in matches:
        if i not in seen_pred and j not in seen_gt:
            final_pred_boxes.append(prediction_boxes[i])
            final_gt_boxes.append(gt_boxes[j])
            seen_pred.add(i)
            seen_gt.add(j)

    return np.array(final_pred_boxes), np.array(final_gt_boxes)



def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    # YOUR CODE HERE
    # Find all possible matches
    matched_prediction_boxes, _ = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    # Calculate true positives, false positives and false negatives
    tp = len(matched_prediction_boxes)
    fp = len(prediction_boxes) - tp
    fn = len(gt_boxes) - tp

    return {"true_pos": tp, "false_pos": fp, "false_neg": fn}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # YOUR CODE HERE
    # Calculate true positives, false positives and false negatives for all images
    #all_results = [calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold) for prediction_boxes, gt_boxes in zip(all_prediction_boxes, all_gt_boxes)]
    tp, fp, fn = 0, 0, 0
    for pred_box, gt_box in zip(all_prediction_boxes, all_gt_boxes):
        res = calculate_individual_image_result(pred_box, gt_box, iou_threshold)
        tp += res["true_pos"]
        fp += res["false_pos"]
        fn += res["false_neg"]

    precision = calculate_precision(tp, fp, fn)
    recall = calculate_recall(tp, fp, fn)

    return (precision, recall)

def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []
    for threshold in confidence_thresholds:
        # Filter out all boxes with confidence score below threshold
        filtered_prediction_boxes = [prediction_boxes[confidence_scores[i] > threshold] for i, prediction_boxes in enumerate(all_prediction_boxes)]

        # Calculate precision and recall
        precision, recall = calculate_precision_recall_all_images(filtered_prediction_boxes, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    average_precision = 0
    for level in recall_levels[:-1]:
        precisions_above_recall_level = precisions[recalls >= level]
        if len(precisions_above_recall_level) == 0:
            precision = 0
        else:
            precision = np.max(precisions_above_recall_level)
        average_precision += precision / 11

    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))

def main():
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)

if __name__ == "__main__":
    main()
