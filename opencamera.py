## importing cv2 and numpy
import cv2
##using a lot of cv2 functions
import numpy as np

def main():
    # open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Using KNN algorithm which is more robust for detecting people
    # Lower learning rate and better threshold for face detection, takes longer
    ## history is how long it remembers the background
    ## dist2Threshold is how different a pixel must be to be considered foreground
    ## detectshadow not important here
    bg_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)
    
    print("Camera opened successfully!")
    print("Press 'q' to quit")
    print("Press 'r' to reset background model")
    
    ##starts the background pixel count
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        if frame_count > 50:  # Longer initialization for better background model
            fg_mask = bg_subtractor.apply(frame, learningRate=0.001)  # Lower learning rate to prevent person from being learned as background
            
            # Convert shadow pixels (value 127) to foreground (255)
            ## so you can keep your shadow with you
            fg_mask[fg_mask == 127] = 255
            
            # Clean up the foreground mask with better morphological operations
            ## close fill small holes, open removes noise
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_small)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small)
            
            # Fill larger holes and smooth the mask
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)
            
            # Dilate slightly to ensure we capture edges
            fg_mask = cv2.dilate(fg_mask, kernel_small, iterations=1)
            
            # Create output frame showing person separated from background
            # Person will be shown in original colors, background will have red tint
            output = frame.copy()
            
            # Create red-tinted version of the frame for background
            red_tint = frame.copy()
            # Increase red channel and decrease green/blue channels for red tint effect
            red_tint[:, :, 2] = np.clip(red_tint[:, :, 2] * 1.3, 0, 255).astype(np.uint8)  # Increase red
            red_tint[:, :, 0] = np.clip(red_tint[:, :, 0] * 0.7, 0, 255).astype(np.uint8)  # Decrease blue
            red_tint[:, :, 1] = np.clip(red_tint[:, :, 1] * 0.7, 0, 255).astype(np.uint8)  # Decrease green
            
            # Create inverse mask for background areas
            bg_mask = cv2.bitwise_not(fg_mask)
            
            # Apply red tint to background areas
            output[bg_mask > 0] = red_tint[bg_mask > 0]
            
            # Create side-by-side display
            combined = np.hstack([frame, output])
            
            # Add labels
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Person Separated", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Person Separation', combined)
        else:
            # Show original frame while background model is learning
            cv2.putText(frame, f"Initializing background model... ({frame_count}/50)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Stay still and don't move!", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Person Separation', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset background model
            bg_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)
            frame_count = 0
            print("Background model reset!")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Program ended.")

if __name__ == "__main__":
    main()
