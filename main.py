from ultralytics import YOLO
import cv2

# Load the object detection model
detector_model = YOLO('detection_model.pt')

# Load the image classification model
dish_classifier_model = YOLO('dish_classification_model.pt')
tray_classifier_model = YOLO('tray_classification_model.pt')

# Load your image
cap = cv2.VideoCapture("AU_15.mp4")

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Read and display each frame of the stream
while True:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("video_result.mp4", fourcc, 15.0, (1920, 1080))
    ret, frame = cap.read()
    if not ret:
        continue
    # Run object detection
    detector_results = detector_model.track(frame, stream=True, imgsz=1920, persist=True)
    for detector_result in detector_results:
        # get the classes names
        classes_names = detector_result.names
        # Iterate over detection results
        for box in detector_result.boxes:
            # Extract bounding box coordinates
            if box.conf[0] > 0.6:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]

                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                label_detector = classes_names[cls]

                # get the respective colour
                colour = getColours(cls)

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), 2)

                # Run image classification on the cropped image
                crop_img = frame[y1:y2, x1:x2]
                if label_detector == "dish":
                    classifier_results = dish_classifier_model(crop_img)
                else:
                    classifier_results = tray_classifier_model(crop_img)
                classifer_list = classifier_results[0].names

                # Get classification label (assuming top-1 classification)
                label = classifer_list[classifier_results[0].probs.top1]
                
                # Combine results by annotating the original image
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)  # Draw bounding box
                cv2.putText(frame, f"{label_detector} {box.conf[0]:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)  # Add label
                cv2.putText(frame, label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)  # Add label
    cv2.imshow("Frame",frame)
    
    # Wait for 1ms for key press to continue or exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()