from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from configparser import ConfigParser
import google.generativeai as genai
import markdown


# Constants
DISTANCE_THRESHOLD = 30
STOP_THRESHOLD = 10
COLLISION_DISTANCE_THRESHOLD = 250
OUT_OF_ARENA_THRESHOLD = 60
FLOW_MAGNITUDE = 2

# Load the YOLO model
MODEL = YOLO("weight/best_yolov10n_dran_vs_wizard.pt")

# Initialize battle data
battle_data = {
    "type1": None,
    "type2": None,
    "winner": None,
    "game_over_time": None,
    "winner_spinning_time": None,
    "win_reason": None,
    "collision_count_approx": None
}

analysis_response = ""

# Initialize battle log
battle_log = {
    "frame": [],
    "beyblade1": [],
    "beyblade2": [],
    "beyblade1_position": [],
    "beyblade2_position": [],
    "collision_status": [],
    "beyblade1_status": [],
    "beyblade2_status": []
}

# Initialize state variables
previous_centers = [None, None]
distance_count_frames = [0, 0] # Number of frames Beyblade has moved a small distance [type1, type2]
aspect_ratio_frames = [0, 0] # Index of beyblade [type1, type2]
not_detected_frames = 0
loser_frame = None 
stopped_index = [None, None] # Index of Beyblade that has stopped spinning [Loser, Winner]  
collision_count = 0
collision_active = False
status = ["Spinning", "Spinning"]

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Create an uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define your existing functions here (calculate_center, detect_collisions, etc.)
def calculate_center(bbox):
    """
    Calculate the center of a bounding box.
    
    Args:
        bbox: Bounding box coordinates (x1, y1, x2, y2).
    
    Returns:
        A numpy array containing the x and y coordinates of the center.
    """
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return np.array([center_x, center_y])

def detect_collisions(bboxes):
    """
    Detect collisions between two bounding boxes.
    If distance between centers of two bounding boxes is less than COLLISION_DISTANCE_THRESHOLD, a collision is detected.
    Before detecting a new collision, the collision_active flag is set to False when distance is greater than COLLISION_DISTANCE_THRESHOLD.
    
    Args:
        bboxes: List of bounding boxes.
    """
    global collision_count, collision_active
    
    if len(bboxes) < 2:
        return
    
    center1 = calculate_center(bboxes[0])
    center2 = calculate_center(bboxes[1])
    distance = np.linalg.norm(center1 - center2)
    if distance < COLLISION_DISTANCE_THRESHOLD:
        if not collision_active:
            collision_count += 1
            collision_active = True
            return True # collision detected
    else:
        collision_active = False
        
    return False # no collision

def has_beyblade_stopped(results, bboxes, frame_count, fps):
    """
    Check if any of the Beyblades has stopped.

    If a center of a Beyblade only moves a small distance less than DISTANCE_THRESHOLD for STOP_THRESHOLD frames
    AND
    To ensure the Beyblade is not spinning in place, we also check the aspect ratio of the bounding box.
    If the aspect ratio of the bounding box is not within 0.85 and 1.2 for STOP_THRESHOLD frames,
    it is considered stopped.
    
    Args:
        results: Detected objects with names (list).
        bboxes: List of bounding boxes for each detected object.
        frame_count: Current frame count.
        fps: Frames per second of the video.
    """
    global loser_frame, winner_frame
    
    for i, bbox in enumerate(bboxes):
        # Calculate the center of the bounding box
        center = calculate_center(bbox)
        
        # Check distance moved by the Beyblade center
        if previous_centers[i] is not None:
            distance = np.linalg.norm(center - previous_centers[i])
            #print(f"Distance moved by Beyblade {i}: {distance}")
            if distance < DISTANCE_THRESHOLD:
                distance_count_frames[i] += 1
            else:
                distance_count_frames[i] = 0
                
        previous_centers[i] = center
        
        # Calculate the aspect ratio of the bounding box
        x1, y1, x2, y2 = map(int, bbox)
        width, height = x2 - x1, y2 - y1
        aspect_ratio = width / height

        # Identify Beyblade type based on detection
        beyblade_type = results[0].names[i]
        beyblade_idx = 0 if beyblade_type == battle_data["type1"] else 1
        
        
        # Determine if the Beyblade has stopped
        if (
            distance_count_frames[beyblade_idx] >= STOP_THRESHOLD
            ):
            
            if (stopped_index[0] is not None and stopped_index[1] is None and beyblade_type == battle_data["winner"] ): # for winner beyblade
                print(f"Winner beyblade {beyblade_type} stopped spinning.")
                stopped_index[1] = beyblade_idx
                winner_frame = frame_count - STOP_THRESHOLD
                battle_data["winner_spinning_time"] = winner_frame / fps
                status[beyblade_idx] = "Stopped"
            
            if stopped_index[0] is None and battle_data["game_over_time"] is None:  # for loser beyblade
                print(f"Loser beyblade {beyblade_type} stopped spinning.")
                stopped_index[0] = beyblade_idx
                loser_frame = frame_count - STOP_THRESHOLD
                battle_data["winner"] = battle_data["type1"] if beyblade_idx == 1 else battle_data["type2"]
                battle_data["win_reason"] = "Opponent stopped spinning"
                battle_data["game_over_time"] = loser_frame / fps
                status[beyblade_idx] = "Stopped"
    
    return status
    
"""
def has_beyblade_stopped(frame, prev_frame, results, bboxes, frame_count, fps):
    
    '''
    Check if any of the Beyblades has stopped using optical flow.

    If the average motion of the Beyblade is below a certain threshold for STOP_THRESHOLD frames,
    it is considered stopped.

    Args:
        frame: Current video frame.
        prev_frame: Previous video frame for optical flow calculation.
        results: Detected objects with names (list).
        bboxes: List of bounding boxes for each detected object.
        frame_count: Current frame count.
        fps: Frames per second of the video.
    '''
    global loser_frame, winner_frame
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    for i, bbox in enumerate(bboxes):
               
        # Calculate the average flow vector for the bounding box
        x1, y1, x2, y2 = map(int, bbox)
        flow_region = flow[y1:y2, x1:x2]
        
        # Calculate the mean flow in the bounding box
        mean_flow = np.mean(flow_region, axis=(0, 1))
        flow_magnitude = np.linalg.norm(mean_flow)
        

        # Identify Beyblade type based on detection
        beyblade_type = results[0].names[i]
        #print(f"Beyblade type {i}: {beyblade_type}")
        beyblade_idx = 0 if beyblade_type == battle_data["type1"] else 1
        
        print(f"{beyblade_type} : {flow_magnitude}")
        # Check if the flow magnitude is below the stopping threshold
        if flow_magnitude < FLOW_MAGNITUDE:
            distance_count_frames[beyblade_idx] += 1
        else:
            distance_count_frames[beyblade_idx] = 0

        # Determine if the Beyblade has stopped
        if distance_count_frames[beyblade_idx] >= STOP_THRESHOLD:
            if (stopped_index[0] is not None and stopped_index[1] is None and beyblade_type == battle_data["winner"]):
                print(f"Winner beyblade {beyblade_type} stopped spinning.")
                stopped_index[1] = beyblade_idx
                winner_frame = frame_count - STOP_THRESHOLD
                battle_data["winner_spinning_time"] = winner_frame / fps
                status[beyblade_idx] = "Stopped"
            
            if stopped_index[0] is None and battle_data["game_over_time"] is None:
                print(f"Loser beyblade {beyblade_type} stopped spinning.")
                stopped_index[0] = beyblade_idx
                loser_frame = frame_count - STOP_THRESHOLD
                battle_data["winner"] = battle_data["type1"] if beyblade_idx == 1 else battle_data["type2"]
                battle_data["win_reason"] = "Opponent stopped spinning"
                battle_data["game_over_time"] = loser_frame / fps
                status[beyblade_idx] = "Stopped"
    
    return status
    
"""




def has_beyblade_out_of_arena(results, bboxes, frame_count, fps):
    """
    Check if any of the Beyblades has gone out of the arena.
    If only one Beyblade is detected for more than threshold, missing Beyblade is considered out of the arena.
    
    Args:
        results: YOLO detection results.
        bboxes: List of bounding boxes.
        frame_count: Current frame count.
        fps: Frames per second of the video.
    """
    global not_detected_frames, loser_frame
    START_TOLERATE = 60 # 2 seconds
    if frame_count > START_TOLERATE and len(bboxes) <= 1:
        not_detected_frames += 1
        #print(f"Opponent not detected for {not_detected_frames} frames.")
        if not_detected_frames > OUT_OF_ARENA_THRESHOLD and battle_data["game_over_time"] is None:
            #print("Opponent out of the arena.")
            loser_frame = frame_count - (OUT_OF_ARENA_THRESHOLD+START_TOLERATE)
            if len(results[0].boxes.cls) > 0:
                class_index = int(results[0].boxes.cls[0])
                if class_index < len(results[0].names):
                    battle_data["winner"] = results[0].names[class_index]
                    status[1- class_index] = "Out of Arena"
                    battle_data["game_over_time"] = loser_frame / fps
                    battle_data["win_reason"] = "Opponent out of the arena"
                    
                else:
                    print("Class index out of range for names list.")
            else:
                print("No classes detected in results.")
            
    else:
        not_detected_frames = 0

def extract_video_segment(input_video_path, output_video_path, start_frame, end_frame, fps):
    """
    Extract a segment of the video.
    
    Args:
        input_video_path: Path to the input video file.
        output_video_path: Path to save the extracted video segment.
        start_frame: Start frame for the segment.
        end_frame: End frame for the segment.
        fps: Frames per second of the video.
    """
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame + 1):
        success, frame = cap.read()
        if not success:
            break
        out.write(frame)
    cap.release()
    out.release()

    #print(f"Extracted segment saved to {output_video_path}")

def analyze_battle_using_LLM():
    """
    Analyze the Beyblade battle using the Gemini LLM model.
    """
    DEFAULT_SYSTEM_PROMPT = """
    
    You are the judge and expert in the beyblade battle. 
    You will be given battle result data along with game logs. 
    Give a detailed analysis. 
    You can also add information related to the specifications of the beyblade used based on your knowledge.
    
    Instruction:
    - Analyze the battle result data and game logs.
    - Provide a detailed analysis of the battle.
    - Add information related to the specifications of the beyblade used.
    
    Output:
    - Detailed analysis of the battle.
    
    """
    
    battle_result = open('output/battle_data_result.csv', 'r', encoding='utf-8').read()
    battle_log = open('output/battle_log.csv', 'r', encoding='utf-8').read()
    
    user_content = f""" Here is the battle result {battle_result}
    
    and here is the battle log
    {battle_log}"""
    
    config = ConfigParser()
    config.read('config.ini')
    gemini_api_key = config['GEMINI']['api_key']
    
    genai.configure(api_key=gemini_api_key)
    
    model=genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=DEFAULT_SYSTEM_PROMPT)
    response = model.generate_content(user_content)
    #print(response.text)
    
    return response.text


# Include the main video processing logic in a function
def process_video(video_path):
    global battle_data, battle_log, analysis_response

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_video_path = "static/video_output/result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    # Read the first frame and initialize prev_frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)  # Convert first frame to grayscale

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        results = MODEL.track(frame, conf=0.7, max_det=2, verbose=False, tracker="botsort.yaml")
        annotated_frame = results[0].plot()
        bboxes = results[0].boxes.xyxy.cpu().numpy()

 
        if battle_data["type1"] is None and battle_data["type2"] is None and len(results[0].names) > 1:
            battle_data["type2"] = results[0].names[1]
            battle_data["type1"] = results[0].names[0]

        # Detect collisions and check for out-of-arena condition
        beyblade_positions = [calculate_center(bboxes[i]) for i in range(len(bboxes))]
        collision_status = detect_collisions(bboxes)
        has_beyblade_out_of_arena(results, bboxes, frame_count, fps)
        beyblade_status = has_beyblade_stopped(results, bboxes, frame_count, fps)
        #beyblade_status = has_beyblade_stopped(frame, prev_frame, results, bboxes, frame_count, fps)

        # Update battle log
        battle_log["frame"].append(frame_count)
        battle_log["beyblade1"].append(battle_data["type1"])
        battle_log["beyblade2"].append(battle_data["type2"])
        battle_log["beyblade1_position"].append(beyblade_positions[0] if len(bboxes) > 0 else None)
        battle_log["beyblade2_position"].append(beyblade_positions[1] if len(bboxes) > 1 else None)
        battle_log["collision_status"].append(collision_status)
        battle_log["beyblade1_status"].append(beyblade_status[0])
        battle_log["beyblade2_status"].append(beyblade_status[1])

        out.write(annotated_frame)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update prev_frame to the current frame for the next iteration
        prev_frame = gray_frame.copy()

    if battle_data["winner_spinning_time"] is None:
        #print("Winner beyblade not stopped spinning.")
        battle_data["winner_spinning_time"] = frame_count / fps

    battle_data["collision_count_approx"] = collision_count

    if loser_frame is not None:
        frames_before = int(fps * 3)  # Two seconds before losing frame
        frames_after = int(fps * 2)  # One second after losing frame
        start_frame = max(0, loser_frame - frames_before)
        end_frame = min(frame_count, loser_frame + frames_after)
        extract_video_segment(video_path, "static/video_output/highlight.mp4", start_frame, end_frame, fps)

    # Save battle log to CSV
    df_battle_log = pd.DataFrame(battle_log)
    df_battle_log.to_csv("output/battle_log.csv", index=False)

    # Save battle data
    #print(battle_data)
    df_battle_data = pd.DataFrame([battle_data])
    df_battle_data.to_csv('output/battle_data_result.csv', index=False)

    cap.release()
    out.release()

    analysis_response = markdown.markdown(analyze_battle_using_LLM())

    

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    # Clear the battle log before processing a new video
    for key in battle_log.keys():
        battle_log[key].clear()

    # Reset battle_data to its initial state
    global battle_data
    battle_data = {
        "type1": None,
        "type2": None,
        "winner": None,
        "game_over_time": None,
        "winner_spinning_time": None,
        "win_reason": None,
        "collision_count_approx": None
    }

    global status
    status = ["Spinning", "Spinning"]
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        process_video(video_path)  # Call the processing function
        return redirect(url_for('results'))  # Redirect to results page


    
@app.route('/results')
def results():
    global battle_data, analysis_response
    output_video = "static/video_output/result.mp4"
    highlight_video = "static/video_output/highlight.mp4"
    return render_template('results.html', battle_data=battle_data, analysis_response=analysis_response, output_video=output_video, highlight_video=highlight_video)


if __name__ == '__main__':
    app.run(debug=True)
