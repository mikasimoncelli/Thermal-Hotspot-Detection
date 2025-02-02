import cv2
import numpy as np
from pathlib import Path
import argparse

def merge_boxes(boxes, distance_threshold=20):
    """
    Combine boxes that are near each other.
    Args:
        boxes: List of (x, y, w, h) tuples.
        distance_threshold: How close boxes need to be to merge them.
    Returns:
        List of merged (x, y, w, h) tuples.
    """
    if not boxes:
        return []
    
    # Convert tuples to lists to modify them easily
    boxes = [list(box) for box in boxes]
    merged_boxes = []
    
    while boxes:
        current_box = boxes.pop(0)
        current_x, current_y, current_w, current_h = current_box
        
        i = 0
        while i < len(boxes):
            x, y, w, h = boxes[i]
            
            # Determine centers for both boxes
            current_center_x = current_x + current_w / 2
            current_center_y = current_y + current_h / 2
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Determine the distance between the centers
            distance = ((current_center_x - center_x) ** 2 + 
                        (current_center_y - center_y) ** 2) ** 0.5
            
            # If the boxes are close enough merge them
            if distance < distance_threshold:
                # Create a new box that covers both boxes
                new_x = min(current_x, x)
                new_y = min(current_y, y)
                new_w = max(current_x + current_w, x + w) - new_x
                new_h = max(current_y + current_h, y + h) - new_y
                
                current_box = [new_x, new_y, new_w, new_h]
                boxes.pop(i)
            else:
                i += 1
        
        merged_boxes.append(tuple(current_box))
    
    return merged_boxes

def detect_hotspots(frame):
    """
    Identify thermal hotspots in the frame by focusing on red and purple hues.
    Returns a list of bounding boxes in the form [(x, y, w, h)].
    """
    # Convert the frame to HSV color space for more color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for purple/magenta.
    lower_purple = np.array([130, 50, 150])
    upper_purple = np.array([170, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Define the HSV ranges for red
    lower_red1 = np.array([0, 50, 150])
    upper_red1 = np.array([20, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([160, 50, 150])
    upper_red2 = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Merge the red and purple masks together
    combined_mask = cv2.bitwise_or(purple_mask, red_mask1)
    combined_mask = cv2.bitwise_or(combined_mask, red_mask2)
    
    # Exclude the area where the temperature gauge usually is
    gauge_mask = create_gauge_mask(frame.shape[0], frame.shape[1])
    combined_mask = cv2.bitwise_and(combined_mask, gauge_mask)
    
    # Clean up the mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size and color intensity
    min_area = 10  # Consider contours with an area smaller than 10 as noise
    max_area = 1000  # Ignore contours that are too large
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract the region of interest from the HSV image
            roi = hsv[y:y+h, x:x+w]
            
            # Check if the ROI has enough saturated and bright pixels
            sat_bright_pixels = np.sum((roi[:, :, 1] > 50) & (roi[:, :, 2] > 150))
            sat_bright_ratio = sat_bright_pixels / (w * h)
            
            # Only consider areas with at least 30% colored pixels
            if sat_bright_ratio > 0.3:
                # Add a bit of padding around the box
                padding = 3
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                boxes.append((x, y, w, h))
    
    # Combine boxes that are near each other
    boxes = merge_boxes(boxes)
    
    return boxes

def create_gauge_mask(height, width):
    """
    Create a mask to ignore the temperature gauge area.
    """
    mask = np.ones((height, width), dtype=np.uint8)
    # Zero out the left ~15% of the image where the gauge is
    gauge_width = int(width * 0.15)
    mask[:, :gauge_width] = 0
    return mask

def process_video(input_path, output_path=None):
    """
    Process the input video to detect hotspots.
    If an output path is provided, an annotated video is saved.
    Returns a list of tuples: (frame_number, [(x, y, w, h)]).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get the video properties (FPS, dimensions, total frame count)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up the video writer if an output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    results = []
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run hotspot detection on the current frame
        boxes = detect_hotspots(frame)
        
        # Save the frame's detections if any hotspots are found
        if boxes:
            results.append((frame_number, boxes))
            
        # If an annotated video is desired draw overlays
        if writer is not None:
            # Draw rectangles around detected hotspots
            if boxes:
                for x, y, w, h in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Get dimensions for proper text placement
            height, width = frame.shape[:2]
            
            # Overlay the frame number in the bottom right corner
            frame_text = f"Frame: {frame_number}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(frame_text, font, font_scale, thickness)
            
            text_x = width - text_width - 10
            text_y = height - 10
            cv2.putText(frame, frame_text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
            
            # Overlay the hotspot detection status
            if boxes:
                status_text = "Hotspot Detected"
                color = (0, 0, 255)  # Red indicates detection
            else:
                status_text = "No Hotspot"
                color = (0, 255, 0)  # Green means clear
            
            (status_width, _), _ = cv2.getTextSize(status_text, font, font_scale, thickness)
            status_x = width - status_width - 10
            status_y = text_y - text_height - 10
            cv2.putText(frame, status_text, (status_x, status_y), font, font_scale, color, thickness)
            
            writer.write(frame)
        
        frame_number += 1
        
        # Print out progress every 30 frames
        if frame_number % 30 == 0:
            print(f"Processed {frame_number}/{total_frames} frames")
    
    # Release resources after processing
    cap.release()
    if writer:
        writer.release()
    
    return results

def summarize_detections(results):
    """
    Group frame-by-frame detections into continuous hotspot sequences.
    Returns a list of tuples: (hotspot_id, start_frame, end_frame).
    """
    if not results:
        return []
    
    # Make sure the results are in order by frame number
    results.sort(key=lambda x: x[0])
    
    sequences = []
    current_sequence = None
    
    for frame_num, boxes in results:
        if boxes:  # There's a hotspot in this frame
            if current_sequence is None:
                # Start a new hotspot sequence
                current_sequence = [len(sequences) + 1, frame_num, frame_num]
            else:
                # Update the sequence with the current frame
                current_sequence[2] = frame_num
        else:
            if current_sequence is not None:
                # End the current sequence since no hotspot was detected
                sequences.append(tuple(current_sequence))
                current_sequence = None
    
    # If a sequence was active at the end add it
    if current_sequence is not None:
        sequences.append(tuple(current_sequence))
    
    return sequences

def write_summary(sequences, output_path):
    """
    Write the hotspot sequences to a summary text file.
    """
    with open(output_path, 'w') as f:
        if not sequences:
            f.write("No hotspots detected in video.\n")
            return
        
        f.write(f"Found {len(sequences)} hotspot sequence(s):\n\n")
        for hotspot_id, start_frame, end_frame in sequences:
            duration = end_frame - start_frame + 1
            f.write(f"Hotspot {hotspot_id}: frames {start_frame}-{end_frame} ({duration} frames)\n")

def main():
    # Set up the argument parser for command line inputs
    parser = argparse.ArgumentParser(description='Detect thermal hotspots in video footage.')
    parser.add_argument('input', type=str, help='Input video file path')
    parser.add_argument('-o', '--output', type=str, help='Output video file path (optional)')
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Define the input and output file paths
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        # Default to a file name with an "_annotated" suffix
        output_path = input_path.with_name(f"{input_path.stem}_Annotated_Output{input_path.suffix}")
    
    # Define the path for the summary output file
    summary_path = input_path.with_name(f"{input_path.stem}_Summary.txt")
    
    try:
        print(f"\nProcessing {input_path}")
        
        # Analyze the video to detect hotspots
        results = process_video(str(input_path), str(output_path))
        
        # Summarize detections into sequences and write the summary
        sequences = summarize_detections(results)
        write_summary(sequences, summary_path)
        
        # Print out the detection results
        if sequences:
            print(f"\nFound {len(sequences)} hotspot sequence(s):")
            for hotspot_id, start_frame, end_frame in sequences:
                duration = end_frame - start_frame + 1
                print(f"Hotspot {hotspot_id}: frames {start_frame}-{end_frame} ({duration} frames)")
        else:
            print("\nNo hotspots detected in video.")
                
        print(f"\nProcessing complete:")
        print(f"- Annotated video: {output_path}")
        print(f"- Detection summary: {summary_path}")
        
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()
