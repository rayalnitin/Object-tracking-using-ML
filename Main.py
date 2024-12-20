import cv2
import glob
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog, local_binary_pattern

def load_annotations(annotation_file):
    """
    Load annotations from the text file.
    Format: x,y,width,height
    """
    annotations = {}
    try:
        with open(annotation_file, 'r') as f:
            for frame_idx, line in enumerate(f, start=1):
                values = list(map(int, line.strip().split(',')))
                annotations[frame_idx] = (values[0], values[1], values[2], values[3])
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return None
    return annotations

def extract_features(frame, bbox):
    """
    Extract combined HOG and LBP features from the region of interest (ROI).
    """
    x, y, w, h = map(int, bbox)
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    # Resize ROI for consistent feature extraction
    roi_resized = cv2.resize(roi, (64, 64))

    # Convert ROI to grayscale for HOG
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    hog_features = hog(roi_gray,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       visualize=False,
                       feature_vector=True)

    # Extract LBP features
    lbp = local_binary_pattern(roi_gray, P=8, R=1, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum()  # Normalize the histogram

    # Combine HOG and LBP features
    return np.concatenate((hog_features, lbp_hist))

def train_tracker(sequence_path, annotation_file):
    """
    Train a classifier using annotated frames with HOG and LBP features.
    """
    frame_files = sorted(glob.glob(os.path.join(sequence_path, "*.jpg")))
    annotations = load_annotations(annotation_file)

    if not annotations:
        print("No annotations found")
        return None

    X = []  # Features
    y = []  # Labels

    for frame_idx, frame_file in enumerate(frame_files, start=1):
        frame = cv2.imread(frame_file)
        if frame is None:
            continue

        if frame_idx in annotations:
            bbox = annotations[frame_idx]
            features = extract_features(frame, bbox)
            if features is not None:
                X.append(features)
                y.append(1)  # Positive example

            # Generate challenging negative examples
            for _ in range(5):
                h, w = frame.shape[:2]
                random_x = np.random.randint(max(0, bbox[0] - 20), min(w, bbox[0] + 20))
                random_y = np.random.randint(max(0, bbox[1] - 20), min(h, bbox[1] + 20))
                random_bbox = (random_x, random_y, bbox[2], bbox[3])

                features = extract_features(frame, random_bbox)
                if features is not None:
                    X.append(features)
                    y.append(0)  # Negative example

    if len(X) == 0:
        print("No features extracted")
        return None

    # Normalize features
    X = np.array(X)
    y = np.array(y)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate performance
    y_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    return classifier

def track_object(sequence_path, annotation_file, classifier=None):
    frame_files = sorted(glob.glob(os.path.join(sequence_path, "*.jpg")))
    annotations = load_annotations(annotation_file)

    if not frame_files:
        print(f"No frames found in {sequence_path}")
        return None

    # Read the first frame
    frame = cv2.imread(frame_files[0])
    if frame is None:
        print("Failed to read frames from the sequence")
        return None

    # Get the initial window either from annotations or user selection
    if annotations and 1 in annotations:
        track_window = annotations[1]
    else:
        track_window = cv2.selectROI("Frame", frame, False)
        cv2.destroyWindow("Frame")

    x, y, w, h = map(int, track_window)

    # Setup the ROI for tracking
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calculate histogram for the ROI
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup termination criteria
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    for frame_idx, frame_file in enumerate(frame_files, start=1):
        frame = cv2.imread(frame_file)
        if frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply Mean-Shift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window

        # Verify tracking result using classifier if available
        if classifier is not None:
            features = extract_features(frame, (x, y, w, h))
            if features is not None and classifier.predict([features])[0] == 1:
                # Confirm tracking if classifier detects object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
                cv2.putText(frame, "OBJECT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                # If classifier fails to detect object, don't draw a bounding box
                pass
        else:
            # If no classifier is used, just show the blue bounding box (we assume object is tracked)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
            cv2.putText(frame, "OBJECT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Draw ground truth if available
        if frame_idx in annotations:
            gt_bbox = annotations[frame_idx]
            cv2.rectangle(frame, 
                        (int(gt_bbox[0]), int(gt_bbox[1])),
                        (int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])),
                        (0, 255, 0), 2)  # Green box for ground truth

        # Display the processed frame
        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

    cv2.destroyAllWindows()

def process_sequences(sequences_path, annotations_path):
    sequence_folders = [f for f in os.listdir(sequences_path) if os.path.isdir(os.path.join(sequences_path, f))]

    for sequence in sequence_folders:
        sequence_path = os.path.join(sequences_path, sequence)
        annotation_file = os.path.join(annotations_path, f"{sequence}.txt")

        print(f"\nProcessing sequence: {sequence}")
        classifier = train_tracker(sequence_path, annotation_file)
        track_object(sequence_path, annotation_file, classifier)

# Update paths to your data
sequences_path = "sequences"
annotations_path = "annotations"

process_sequences(sequences_path, annotations_path)