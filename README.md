# Team8-ItemSegmentation
Project for Code Coogs Team 8. Item segmentation will be used for a tbd use

cv2.imread("image.jpg")                         # Load an image from disk
cv2.imwrite("out.jpg", img)                     # Save an image to disk
cv2.imshow("Window", img)                       # Display an image in a window
cv2.waitKey(0)                                  # Wait indefinitely for a key press
cv2.waitKey(1)                                  # Wait briefly (used for video frames)
cv2.destroyAllWindows()                         # Close all OpenCV windows

cv2.VideoCapture(0)                             # Open the default webcam
cap.read()                                      # Capture a single frame from the camera
cap.isOpened()                                  # Check if the camera opened correctly
cap.release()                                   # Release the camera resource

cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           # Convert image to grayscale
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            # Convert BGR to RGB format
cv2.cvtColor(img, cv2.COLOR_BGR2HSV)            # Convert image to HSV color space

cv2.resize(img, (w, h))                         # Resize image to new width and height
cv2.GaussianBlur(img, (5,5), 0)                 # Blur image to reduce noise
cv2.medianBlur(img, 5)                          # Remove noise while preserving edges
cv2.bilateralFilter(img, 9, 75, 75)             # Smooth image without blurring edges

cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)# Convert image to black and white
cv2.threshold(gray, 0, 255, 
              cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Auto threshold selection
cv2.adaptiveThreshold(gray, 255,
                      cv2.ADAPTIVE_THRESH_MEAN_C,
                      cv2.THRESH_BINARY, 11, 2) # Threshold based on local regions

cv2.Canny(gray, 100, 200)                       # Detect edges in the image
cv2.Sobel(gray, cv2.CV_64F, 1, 0)                # Detect horizontal edges
cv2.Laplacian(gray, cv2.CV_64F)                  # Detect edges using second derivatives

cv2.erode(img, kernel, iterations=1)            # Shrink white regions (remove noise)
cv2.dilate(img, kernel, iterations=1)           # Expand white regions
cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)   # Remove small objects (erosion + dilation)
cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # Close small holes in objects

cv2.findContours(binary,
                 cv2.RETR_EXTERNAL,
                 cv2.CHAIN_APPROX_SIMPLE)       # Find object outlines in a binary image
cv2.drawContours(img, contours, -1, (0,255,0), 2)# Draw detected contours
cv2.contourArea(cnt)                            # Compute area of a contour
cv2.boundingRect(cnt)                           # Get bounding box around a contour

cv2.line(img, pt1, pt2, color, thickness)       # Draw a line on the image
cv2.rectangle(img, pt1, pt2, color, thickness)  # Draw a rectangle
cv2.circle(img, center, radius, color, thickness)#
