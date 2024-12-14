import cv2
import numpy as np


def preprocess_image(image):
    # Load and preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return image, thresh

def segment_pieces(thresh):
    # Find contours of pieces
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #pieces = [cv2.boundingRect(cnt) for cnt in contours]
    return contours#, pieces

def extract_features(piece_image):
    # Use keypoint detection or shape descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(piece_image, None)
    return keypoints, descriptors

def match_pieces(features1, features2):
    # Match features between pieces
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(features1[1], features2[1], k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good_matches


img = cv2.imread('../data/10b.jpg')     
assert img is not None, "file could not be read, check with os.path.exists()"

image, thresh = preprocess_image(img)
#contours= segment_pieces(thresh)
#features = [extract_features(cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)) for x, y, w, h in pieces]
#contour_image = img.copy()

cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow("thresh", thresh)
cv2.imshow("Contours", contour_image)
#cv2.imshow("pieces", pieces)
#cv2.imshow("features", features)
cv2.waitKey(0)
cv2.destroyAllWindows()
