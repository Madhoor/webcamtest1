import cv2
import mediapipe as mp
import numpy as np

# A helpful utility function to calculate the angle between three points.
# This is useful for determining the orientation of a body part (e.g., the torso).
def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (landmarks).
    a: First landmark
    b: Midpoint landmark (vertex of the angle)
    c: Third landmark
    Returns the angle in degrees.
    """
    a = np.array(a)  # First coordinate
    b = np.array(b)  # Second coordinate (vertex)
    c = np.array(c)  # Third coordinate
    
    # Calculate the vectors from the midpoint to the other two points
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    
    # Ensure the angle is between 0 and 180 degrees
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- Main Function for Live Detection ---
def detect_person_and_fall_live():
    """
    Detects people and their skeletal poses in a live video stream using MediaPipe.
    It then uses pose data to detect if a person has fallen.
    """
    # Initialize MediaPipe's drawing and pose solution
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Initialize video capture. Use 0 for the default webcam.
    # If you have multiple webcams, you might need to change this to 1, 2, etc.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- Fall Detection Variables ---
    fall_status = "Monitoring the patient"
    fall_frames_counter = 0
    torso_angle = 0.0
    
    # --- TUNABLE PARAMETERS ---
    # To find the best threshold for your setup:
    # 1. Run the script.
    # 2. Watch the "Torso Angle" value on the screen.
    # 3. Note the angle when you are standing vs. when you are lying down ("fallen").
    # 4. Set the FALL_ANGLE_THRESHOLD to a value that is between your standing and fallen angles.
    FALL_ANGLE_THRESHOLD = 55.0 # change this value acc to threashold 
    
    # Number of consecutive frames the angle must be over the threshold to confirm a fall
    CONSECUTIVE_FRAMES_THRESHOLD = 15 

    # Use MediaPipe Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Resize for faster processing and flip for a natural selfie-view
            frame = cv2.resize(frame, (1280, 720))
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image to get pose landmarks
            results = pose.process(image)
            
            # Revert color space and writeable flag for drawing
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check if any pose landmarks were detected
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # --- Improved Fall Detection Logic ---
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates for key joints
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
                    
                    # Calculate the center of the hips and shoulders
                    hip_center = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
                    shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]

                    # Define a vertical reference point directly above the hip center
                    vertical_ref_point = [hip_center[0], hip_center[1] - 1] 

                    # Calculate the angle of the torso with respect to the vertical axis
                    torso_angle = calculate_angle(shoulder_center, hip_center, vertical_ref_point)
                    
                    # --- State Logic ---
                    # If the torso angle suggests a fall
                    if torso_angle > FALL_ANGLE_THRESHOLD:
                        fall_frames_counter += 1
                        # If the "fallen" state persists, confirm the fall
                        if fall_frames_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                            fall_status = "Fallen"
                    else:
                        # If the person is upright, reset the counter and status
                        fall_frames_counter = 0
                        fall_status = "Monitoring the patient"

                except Exception as e:
                    # Handle cases where landmarks are not fully visible
                    # print(f"Could not process landmarks: {e}")
                    pass
            
            # Display the fall status on the screen
            text_color = (0, 0, 255) if fall_status == "Fallen" else (0, 255, 0)
            cv2.putText(image, fall_status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3, cv2.LINE_AA)

            # --- DISPLAY FOR DEBUGGING ---
            # Display the calculated torso angle on the screen to help with tuning
            cv2.putText(image, f"Torso Angle: {torso_angle:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)


            # Display the resulting frame
            cv2.imshow('Live Fall Detection', image)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
# Execute the function
if __name__ == '__main__':
    detect_person_and_fall_live()

