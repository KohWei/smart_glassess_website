import threading
from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
from urllib.parse import urlparse
import cv2
import requests
import pyodbc
import os
from collections import defaultdict
from imageClassification import ImageClassifier
from objectDetection import ObjectDetection
import base64
from threading import Thread, Lock
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Database connection details
server = r'LAPTOP-UUABUEJT\SQLEXPRESS'
database = 'rwDb'
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes"
model_stored_path = r"C:\Users\User\Intern\RealWear2\model"

room_cameras = {}
room_inference_engine = {}

video_stream_ready = {}

sequence_progress = {}  

room_lock = Lock()



model_classes = {
    'image classification': ImageClassifier,
    'object detection': ObjectDetection
}


# Access the database
def get_db_connection():
    try:
        conn = pyodbc.connect(connection_string)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def get_smart_glasses_list():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT camera_id, device_name, rtsp_url FROM camera")
        cameras = cursor.fetchall()
        conn.close()
        return [{'camera_id': row.camera_id, 
                 'device_name': row.device_name, 
                 'rtsp_url': row.rtsp_url} 
                 for row in cameras]
    else:
        return []
    
def get_sop_name_list():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sop_id,sop_name FROM sop")
        sop = cursor.fetchall()
        conn.close()
        return [{'sop_id': row[0],
                 'sop_name': row[1]
                } 
                for row in sop]  # Access the first element of the tuple
    else:
        return []

def get_room_list():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                room.room_name, 
                sop.sop_name, 
                COUNT(room.camera_id) as number_of_camera, 
                STRING_AGG(camera.device_name, ', ') AS device_names
            FROM room
            INNER JOIN camera ON room.camera_id = camera.camera_id
            INNER JOIN sop ON room.sop_id = sop.sop_id
            GROUP BY room.room_name, sop.sop_name;
        """)
        rooms = cursor.fetchall()  # Fetch all rows as tuples
        conn.close()

        # Build a list of dictionaries from the fetched rows
        return [
            {
                'room_name': row[0],
                'sop_name': row[1],
                'number_of_camera': row[2],
                'device_name': row[3]  # This will now contain the comma-separated camera names
            }
            for row in rooms
        ]
    else:
        return []

def show_live_stream_list():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            query = """
                SELECT r.room_name, c.rtsp_url, c.device_name, s.sop_name
                FROM room r
                JOIN camera c ON r.camera_id = c.camera_id
                JOIN sop s ON r.sop_id = s.sop_id
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            room_dict = {}

            for row in rows:
                room_name, rtsp_url, device_name, sop_name = row
                
                if room_name not in room_dict:
                    room_dict[room_name] = {
                        'sop_name': sop_name,
                        'cameras': []
                    }

                room_dict[room_name]['cameras'].append({
                    'device_name': device_name,
                    'rtsp_url': rtsp_url
                })

            return room_dict  # Dict of rooms, each containing multiple cameras

        except Exception as e:
            print(f"Database error: {e}")
            return {}

    return {}

def get_model_list():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT model_id,model_path,model_class_label_path,model_type FROM model")
        sop = cursor.fetchall()
        conn.close()
        return [{'model_id': row[0],
                 'model_path': row[1],
                 'model_class_label_path': row[2],
                 'model_type': row[3]
                } 
                for row in sop]  # Access the first element of the tuple
    else:
        return []
    
def get_procs_list():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT proc_id, sop_id, model_id, model_class,output, message,sequence FROM procedures")
        procedures = cursor.fetchall()
        conn.close()
        return [{'proc_id': row[0],
                 'sop_id': row[1],
                 'model_id': row[2],
                 'model_class': row[3],
                 'output': row[4],
                 'message': row[5],
                 'sequence': row[6]
                } 
                for row in procedures]  # Access the first element of the tuple
    else:
        return []


# Read the html file
@app.route('/')
def smart_glasses_management():
    smart_glasses_list = get_smart_glasses_list()
    return render_template('smart_glasses_management.html', smart_glasses=smart_glasses_list)

@app.route('/room_management')
def room_management():
    smart_glasses_list = get_smart_glasses_list()
    room_list = get_room_list()
    sop_list = get_sop_name_list()

    
    return render_template('room_management.html',devices=smart_glasses_list, rooms=room_list, sops=sop_list)

@app.route('/live_stream_management')
def live_stream_management():
    devices = show_live_stream_list()
    return render_template('live_stream_management.html',device=devices)


@app.route('/procedure_management')
def procedure_management():
    sop_list = get_sop_name_list()
    model_list = get_model_list()
    proc_list = get_procs_list()
    return render_template('procedure_management.html', sops = sop_list, models = model_list, procs = proc_list)


@app.route('/inference_engine_management')
def inference_engine_management():
    room_list = get_room_list()
    return render_template('inference_engine_management.html', rooms= room_list)



# Smart Glassess Management
def add_camera(camera_data):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()

            device_name = camera_data.get('device_name')
            rtsp_url = camera_data.get('rtsp_url')

            # Validate the data
            if not device_name or not rtsp_url:
                return {'status': 'error', 'message': 'Device Name and RTSP URL are required'}
            
            if cursor.execute("SELECT * FROM camera where device_name = ?", (device_name,)).fetchone():
                return {'status': 'error', 'message': 'Camera already exists'}
            
            # Insert data into the 'camera' table with the correct number of parameters
            cursor.execute("""INSERT INTO camera (device_name, rtsp_url) VALUES (?, ?)""",
                           (device_name, rtsp_url))  

            conn.commit()  # Commit the changes
            return {'status': 'success', 'message': 'Camera added successfully'}
        else:
            return {'status': 'error', 'message': 'Failed to connect to the database'}
    except Exception as e:
        print(f"Error: {e}")
        return {'status': 'error', 'message': 'An error occurred while adding the camera'}
    
@app.route('/start_stream', methods=['POST'])
def start_stream():
    rtsp_url = request.form['rtsp_url']
    
    try:
        # Attempt to validate RTSP URL by opening it
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({"status": "error", "message": "Invalid RTSP URL or could not open the video stream"}), 400
        else:
            parsed_url = urlparse(rtsp_url)
            device_id = parsed_url.path.split('/')[-1]

            # Prepare camera data for database insertion
            camera_data = {
                "device_name": device_id,  
                "rtsp_url": rtsp_url
            }

            # Attempt to add camera to the database
            add_camera_response = add_camera(camera_data)
            if add_camera_response['status'] == 'success':
                return jsonify({"status": "success", "message": "RTSP stream validated and camera added successfully"}), 200
            else:
                return jsonify({
                    "status": "error",
                    "message": f"RTSP validated, but database insertion failed: {add_camera_response['message']}"
                }), 400

    except Exception as e:
        # Handle unexpected errors
        print(f"Error in start_stream: {e}")
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500
    

@app.route('/remove_camera', methods=['POST'])
def remove_camera():
    camera_id = request.form['camera_id']

    conn = get_db_connection()
    if conn:
            cursor = conn.cursor()   
            cursor.execute("DELETE from camera where camera_id=?", (camera_id,))
                           
            conn.commit()
            return jsonify({'status': 'success', 'message': 'Delete the camera sucessfully'})
    else:
            return jsonify({'status': 'error', 'message': 'Failed to delete the camera'})


# Room Management
@app.route('/remove_room', methods=['POST'])
def remove_room():
    room_name = request.form['room_name']

    conn = get_db_connection()
    if conn:
            cursor = conn.cursor()   
            cursor.execute("DELETE from room where room_name=?", (room_name,))
                           
            conn.commit()
            return jsonify({'status': 'success', 'message': 'Delete the room sucessfully'})
    else:
            return jsonify({'status': 'error', 'message': 'Failed to delete the room'})

@app.route('/add_room', methods=['POST'])
def add_room():
    try:
        room_name = request.form['room_name']
        camera_ids = request.form.getlist('camera_id')  # Get camera IDs as a list
        sop_ids = request.form.getlist('sop_id')  # Get SOP IDs as a list

        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            if cursor.execute("SELECT * FROM room where room_name = ?", (room_name,)).fetchone():
                return {'status': 'error', 'message': 'The room already exists. Please join the existing room.'}
            else:
                for camera_id in camera_ids:
                    for sop_id in sop_ids:
                        cursor.execute("""
                            INSERT INTO room (camera_id, sop_id, room_name)
                            VALUES (?, ?, ?)
                        """, (camera_id, sop_id, room_name))
                
                conn.commit()  # Commit the changes

                return jsonify({'status': 'success', 'message': 'Room added successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to connect to the database'})
    except Exception as e:
        print(f"Error adding room: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while adding the room'})

@app.route('/update_room', methods=['POST'])
def update_room():
    try:
        # Get room name
        room_name = request.form['room_name']
        print("Room Name:", room_name)

        # Get selected device names and SOP names
        device_names = request.form.getlist('device_names[]')  
        print("Selected Device Names:", device_names)

        sop_names = request.form.getlist('sop_names[]')  
        print("Selected SOP Names:", sop_names)

        camera_ids = []
        sop_ids = []

    
        # Connect to database
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            # Retrieve camera IDs from device names
            for device_name in device_names:
                cursor.execute("SELECT camera_id FROM camera WHERE device_name = ?", (device_name,))
                result = cursor.fetchone()
                if result:  
                    camera_ids.append(result[0])  # Extract the first value from the tuple
            
            # Retrieve SOP IDs from SOP names
            for sop_name in sop_names:
                cursor.execute("SELECT sop_id FROM sop WHERE sop_name = ?", (sop_name,))
                result = cursor.fetchone()
                if result:
                    sop_ids.append(result[0])  # Extract the first value from the tuple
            
            print("Camera IDs to update:", camera_ids)
            print("SOP IDs to update:", sop_ids)

            cursor.execute("DELETE FROM room WHERE room_name = ?", (room_name,))


            if camera_ids and sop_ids:
                for camera_id in camera_ids:
                    for sop_id in sop_ids:
                        cursor.execute(
                            "INSERT INTO room (camera_id,sop_id, room_name) VALUES (?, ?, ?)",
                        (camera_id, sop_id, room_name)
                        )

                conn.commit()  # Save changes
                conn.close()
                return jsonify({'status': 'success', 'message': 'Room updated successfully'})

        
        else:
            return jsonify({'status': 'error', 'message': 'Failed to connect to the database'})
    
    except Exception as e:
        print(f"Error updating room: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while updating the room'})




# Procedure Management
@app.route('/remove_sop', methods=['POST'])
def remove_sop():
    sop_id = request.form['sop_id']

    conn = get_db_connection()
    if conn:
            cursor = conn.cursor()   
            cursor.execute("DELETE from sop where sop_id=?", (sop_id,))
                           
            conn.commit()
            return jsonify({'status': 'success', 'message': 'Delete the SOP sucessfully'})
    else:
            return jsonify({'status': 'error', 'message': 'Failed to SOP the model'})



@app.route('/remove_model', methods=['POST'])
def remove_model():
    model_id = request.form['model_id']

    conn = get_db_connection()
    if conn:
            cursor = conn.cursor()   
            cursor.execute("DELETE from model where model_id=?", (model_id,))
                           
            conn.commit()
            return jsonify({'status': 'success', 'message': 'Delete the model sucessfully'})
    else:
            return jsonify({'status': 'error', 'message': 'Failed to delete the model'})


@app.route('/add_sop', methods=['POST'])
def add_sop():
    try:
        sop_name = request.form['sop_name']

        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            if cursor.execute("SELECT * FROM sop where sop_name = ?", (sop_name,)).fetchone():
                return {'status': 'error', 'message': 'The SOP already exists.'}
            else:
                # Insert data into the 'camera' table with the correct number of parameters
                cursor.execute("""INSERT INTO sop (sop_name)VALUES (?)""",
                            (sop_name))  

                conn.commit()  # Commit the changes

                return jsonify({'status': 'success', 'message': 'SOP added successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to connect to the database'})
    except Exception as e:
        print(f"Error adding room: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while adding the SOP'})



@app.route('/add_model', methods=['POST'])
def add_model():
    # Get form fields and uploaded files
    sop_name = request.form.get('sop_name')
    model_file = request.files.get('model_path')
    class_file = request.files.get('model_class_path')
    model_type = request.form.get('model_type')

    if model_type == '1':
        model_type = 'image classification'
    else:
        model_type = 'object detection'
    
    print(sop_name, model_file, class_file, model_type)

    if not sop_name or not model_file or not class_file:
        return jsonify({"error": "Missing required fields"}), 400

    # Create a folder for the model based on the sop_name
    model_folder = os.path.join(model_stored_path, sop_name)
    os.makedirs(model_folder, exist_ok=True)

    # Save files
    model_path = os.path.join(model_folder, model_file.filename)
    class_path = os.path.join(model_folder, class_file.filename)
    model_file.save(model_path)
    class_file.save(class_path)

    # Connect to database
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    cursor = conn.cursor()

    # Check if the SOP already exists in the sop table
    cursor.execute("SELECT sop_id FROM sop WHERE sop_name = ?", (sop_name,))
    row = cursor.fetchone()
    if row:
        sop_id = row[0]
    else:
        # Insert new SOP record and get its id
        cursor.execute("INSERT INTO sop (sop_name) VALUES (?)", (sop_name,))

    # Insert a record in the model table with the file paths
    cursor.execute(
        "INSERT INTO model (model_path, model_class_label_path, model_type) VALUES (?, ?, ?)",
        (model_path, class_path, model_type)
    )

    conn.commit()
    conn.close()

    return jsonify({
        "status": "success",
        "message": f"Files uploaded successfully to {model_folder} and records saved"
    }), 200


# Inference Engine

def outputEmit(sequence, predicted_class, message, frame, room):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer.tobytes()).decode('utf-8')

        data = {
            'sequence': sequence,
            'predicted_class': predicted_class,
            'message': message,
            'image': encoded_frame
        }

        # print(f"📡 Sending data to {target_device} in {room}")
        socketio.emit('data', data, to=room)

    except Exception as e:
        print(f"Error in outputEmit: {e}")


def handle_preprocessing(room, camera_name, sop_name):
    """ Function to preprocess input in a separate thread """
    conn = get_db_connection()
    if not conn:
        print(f"[{room} - {camera_name}] Database connection failed")
        return

    try:
        print(f"Received Data -> Room: {room}, Camera: {camera_name}, SOP: {sop_name}")

        rtsp_url = None
        models = defaultdict(list)

        cursor = conn.cursor()

        # Fetch RTSP URL for the specific device
        cursor.execute("SELECT rtsp_url FROM camera WHERE device_name = ?", (camera_name,))
        camera_row = cursor.fetchone()

        if camera_row:
            rtsp_url = camera_row[0]
            print(f"RTSP URL: {rtsp_url}")
        else:
            print(f"No RTSP URL found for device: {camera_name}")
            return

        # Fetch SOP ID
        cursor.execute("SELECT sop_id FROM sop WHERE sop_name = ?", (sop_name,))
        sop_row = cursor.fetchone()

        if sop_row:
            sop_id = sop_row[0]
            print(f"SOP ID: {sop_id}")
        else:
            print(f"SOP '{sop_name}' not found.")
            return

        # Fetch associated models for the SOP
        cursor.execute("""
            SELECT DISTINCT p.model_class, p.output, p.message, p.sequence, 
                   m.model_path, m.model_class_label_path, m.model_type 
            FROM procedures p
            JOIN model m ON p.model_id = m.model_id
            WHERE p.sop_id = ?
            ORDER BY p.sequence ASC
        """, (sop_id,))
        
        procedure_rows = cursor.fetchall()
        if procedure_rows:
            print(procedure_rows)
        else:
            print(f"No models found for SOP: {sop_name}")
            return

        for record in procedure_rows:
            model_class, output, message, sequence, model_path, label_path, model_type = record
            model_type = model_type.strip().lower()
            print(model_type)

            if model_type in model_classes:
                classifier = model_classes[model_type](model_path, label_path)
            else:
                print(f"⚠️ Warning: Unrecognized model type '{model_type}'. Skipping entry.")
                continue 

            models[sequence].append({
                'classifier': classifier,
                'model_class': model_class.strip().lower(),
                'expected_output': output.strip().lower(),
                'message': message
            })

        print(f"Models loaded for {camera_name} in room {room}: {models}")

        # Store models **for each camera individually inside the room**
        room_inference_engine.setdefault(room, {})[camera_name] = models  # 🔹 Each camera gets its own model instance
        room_cameras.setdefault(room, set()).add(camera_name)

        # Ensure each camera has its own video processing thread
        if room not in video_stream_ready:
            video_stream_ready[room] = {}

        if camera_name not in video_stream_ready[room]:
            video_stream_ready[room][camera_name] = True
            video_thread = Thread(target=read_video_stream, args=(room, camera_name, rtsp_url), daemon=True)
            video_thread.start()

    except Exception as e:
        print(f"[{room} - {camera_name}] Database error: {e}")

    finally:
        conn.close()


@app.route('/preprocess_input', methods=['POST'])
def preprocess_input():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    room = data.get('room')
    camera_name = data.get('camera_name')
    sop_name = data.get('sop_name')

    if not room or not camera_name or not sop_name:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    # Start a new thread for preprocessing
    preprocess_thread = Thread(target=handle_preprocessing, args=(room, camera_name, sop_name), daemon=True)
    preprocess_thread.start()

    return jsonify({'status': 'success', 'message': 'Preprocessing started in the background'})



def read_video_stream(room, camera_name, rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error opening RTSP stream: {rtsp_url}")
        return

    print(f"📡 Processing RTSP stream for {camera_name} in {room}")

    models = room_inference_engine.get(room, {}).get(camera_name, {})

    if room not in sequence_progress:
        sequence_progress[room] = {}

    if camera_name not in sequence_progress[room]:
        sequence_progress[room][camera_name] = 0  # Start from sequence 0 (before any step)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame from {camera_name} RTSP stream")
            break

        for sequence in sorted(models.keys()):  # Ensure sequence order
            if sequence != sequence_progress[room][camera_name] + 1:
                continue  # Skip if it's not the expected next sequence

            sequence_validated = False

            while not sequence_validated:
                for model in models[sequence]:
                    classifier = model['classifier']
                    expected_class = model['model_class']
                    expected_output = model['expected_output']
                    message = model['message']

                    # 🔍 Predict the class
                    predicted_class = classifier.predict(frame)

                    if isinstance(predicted_class, (tuple, list)):
                        predicted_class = predicted_class[0] if predicted_class else None

                    if predicted_class:
                        predicted_class = predicted_class.strip().lower()
                        expected_class = expected_class.strip().lower()
                        expected_output = expected_output.strip().lower()

                        print(f"[{camera_name}] Expected: {expected_output}, Predicted: {predicted_class}")

                        if predicted_class == expected_class and expected_output == 'true':
                            if sequence == sequence_progress[room][camera_name] + 1:
                                print(f" [{camera_name}] Sequence {sequence} validated in order")
                                outputEmit(sequence, predicted_class, message, frame, room)  # 🔹 Emit event specific to camera
                                sequence_progress[room][camera_name] = sequence 
                                sequence_validated = True
                                time.sleep(8)
                                break
                            else:
                                print(f" [{camera_name}] Out-of-order action detected. Expected sequence {sequence_progress[room][camera_name] + 1}.")
                        else:
                            error_msg = f"Incorrect action detected for sequence {sequence}. Please try again in order."
                            print(error_msg)
                            outputEmit(sequence, predicted_class, error_msg, frame, room)  # 🔹 Emit error specific to camera

                if not sequence_validated:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Error reading frame from {camera_name} RTSP stream")
                        break

    cap.release()
    print(f"🚪 Closing stream for {camera_name} in {room}")



@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    emit('status', {'msg': f'Joined room: {room}'}, to=room)

@socketio.on('leave')
def handle_leave(data):
    room = data['room']
    leave_room(room)
    print(f"Client {request.sid} left the room {room}")




if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')

# def read_video_stream(room, camera_name, rtsp_url):
#     cap = cv2.VideoCapture(rtsp_url)
#     if not cap.isOpened():
#         print(f"Error opening RTSP stream: {rtsp_url}")
#         return

#     print(f"📡 Processing RTSP stream for {camera_name} in {room}")

#     models = room_inference_engine.get(room, {}).get(camera_name, {})

#     # ✅ Initialize tracking for the camera
#     if room not in sequence_progress:
#         sequence_progress[room] = {}
    
#     if camera_name not in sequence_progress[room]:
#         sequence_progress[room][camera_name] = 0  # Start from sequence 0 (before any step)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Error reading frame from {camera_name} RTSP stream")
#             break

#         for sequence in sorted(models.keys()):  # Ensure sequence order
#             if sequence != sequence_progress[room][camera_name] + 1:
#                 continue  # Skip if it's not the expected next sequence

#             sequence_validated = False

#             while not sequence_validated:
#                 for model in models[sequence]:
#                     classifier = model['classifier']
#                     expected_class = model['model_class']
#                     expected_output = model['expected_output']
#                     message = model['message']

#                     # 🔍 Predict the class
#                     predicted_class = classifier.predict(frame)

#                     if isinstance(predicted_class, (tuple, list)):
#                         predicted_class = predicted_class[0] if predicted_class else None

#                     if predicted_class:
#                         predicted_class = predicted_class.strip().lower()
#                         expected_class = expected_class.strip().lower()
#                         expected_output = expected_output.strip().lower()

#                         print(f"[{camera_name}] Expected: {expected_output}, Predicted: {predicted_class}")

#                         if predicted_class == expected_class and expected_output == 'true':
#                             if sequence == sequence_progress[room][camera_name] + 1:
#                                 print(f"✅ [{camera_name}] Sequence {sequence} validated in order")
#                                 outputEmit(sequence, predicted_class, message, frame, room)
#                                 sequence_progress[room][camera_name] = sequence  # ✅ Update sequence tracking
#                                 sequence_validated = True
#                                 break
#                             else:
#                                 print(f"⚠️ [{camera_name}] Out-of-order action detected. Expected sequence {sequence_progress[room][camera_name] + 1}.")
#                         else:
#                             error_msg = f"Incorrect action detected for sequence {sequence}. Please try again in order."
#                             print(error_msg)
#                             outputEmit(sequence, predicted_class, error_msg, frame, room)

#                 if not sequence_validated:
#                     ret, frame = cap.read()
#                     if not ret:
#                         print(f"Error reading frame from {camera_name} RTSP stream")
#                         break

#     cap.release()
#     print(f"🚪 Closing stream for {camera_name} in {room}")

# @app.route('/preprocess_input', methods=['POST'])
# def preprocess_input():
#     conn = get_db_connection()
#     if not conn:
#         return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500

#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({'status': 'error', 'message': 'No data received'}), 400

#         room = data.get('room')
#         camera_name = data.get('camera_name')  # Identifies each device
#         sop_name = data.get('sop_name')

#         if not room or not camera_name or not sop_name:
#             return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

#         print(f"Received Data -> Room: {room}, Camera: {camera_name}, SOP: {sop_name}")

#         rtsp_url = None
#         models = defaultdict(list)

#         cursor = conn.cursor()

#         # Fetch RTSP URL for the specific device
#         cursor.execute("SELECT rtsp_url FROM camera WHERE device_name = ?", (camera_name,))
#         camera_row = cursor.fetchone()

#         if camera_row:
#             rtsp_url = camera_row[0]
#             print(f"RTSP URL: {rtsp_url}")
#         else:
#             print(f"No RTSP URL found for device: {camera_name}")
#             return jsonify({'status': 'error', 'message': f'No RTSP URL found for device {camera_name}'}), 404

#         # Fetch SOP ID
#         cursor.execute("SELECT sop_id FROM sop WHERE sop_name = ?", (sop_name,))
#         sop_row = cursor.fetchone()

#         if sop_row:
#             sop_id = sop_row[0]
#             print(f"SOP ID: {sop_id}")
#         else:
#             print(f"SOP '{sop_name}' not found.")
#             return jsonify({'status': 'error', 'message': f'SOP "{sop_name}" not found'}), 404

#         # Fetch associated models for the SOP
#         cursor.execute("""
#             SELECT DISTINCT p.model_class, p.output, p.message, p.sequence, 
#                    m.model_path, m.model_class_label_path, m.model_type 
#             FROM procedures p
#             JOIN model m ON p.model_id = m.model_id
#             WHERE p.sop_id = ?
#             ORDER BY p.sequence ASC
#         """, (sop_id,))
        
#         procedure_rows = cursor.fetchall()
#         if procedure_rows:
#             print(procedure_rows)
#         else:
#             print(f"No models found for SOP: {sop_name}")
#             return jsonify({'status': 'error', 'message': f'No models configured for {sop_name}'})

#         for record in procedure_rows:
#             model_class, output, message, sequence, model_path, label_path, model_type = record
#             model_type = model_type.strip().lower()
#             print(model_type)

#             if model_type in model_classes:
#                 classifier = model_classes[model_type](model_path, label_path)
#             else:
#                 print(f"⚠️ Warning: Unrecognized model type '{model_type}'. Skipping entry.")
#                 continue 

#             models[sequence].append({
#                 'classifier': classifier,
#                 'model_class': model_class.strip().lower(),
#                 'expected_output': output.strip().lower(),
#                 'message': message
#             })

#         print(f"Models loaded for {camera_name} in room {room}: {models}")

#         # 🔄 Store models for each device inside the room
#         room_inference_engine.setdefault(room, {})[camera_name] = models

#         room_cameras.setdefault(room, set()).add(camera_name)

#         # 🔄 Ensure each device has its own video processing thread
#         if room not in video_stream_ready:
#             video_stream_ready[room] = {}

#         if camera_name not in video_stream_ready[room]:
#             video_stream_ready[room][camera_name] = True
#             thread = Thread(target=read_video_stream, args=(room, camera_name, rtsp_url), daemon=True)
#             thread.start()

#         return jsonify({'status': 'success'})

#     except Exception as e:
#         print(f"Database error: {e}")
#         return jsonify({'status': 'error', 'message': 'Database error'}), 500

#     finally:
#         conn.close()


# def read_video_stream(room, camera_name, rtsp_url):
#     cap = cv2.VideoCapture(rtsp_url)
#     if not cap.isOpened():
#         print(f"Error opening RTSP stream: {rtsp_url}")
#         return

#     print(f"📡 Processing RTSP stream for {camera_name} in {room}")

#     models = room_inference_engine.get(room, {}).get(camera_name, {})

#     # ✅ Initialize tracking for the camera
#     if room not in sequence_progress:
#         sequence_progress[room] = {}
    
#     if camera_name not in sequence_progress[room]:
#         sequence_progress[room][camera_name] = 0  # Start from sequence 0 (before any step)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Error reading frame from {camera_name} RTSP stream")
#             break

#         for sequence in sorted(models.keys()):  # Ensure sequence order
#             if sequence != sequence_progress[room][camera_name] + 1:
#                 continue  # Skip if it's not the expected next sequence

#             sequence_validated = False

#             while not sequence_validated:
#                 for model in models[sequence]:
#                     classifier = model['classifier']
#                     expected_class = model['model_class']
#                     expected_output = model['expected_output']
#                     message = model['message']

#                     # 🔍 Predict the class
#                     predicted_class = classifier.predict(frame)

#                     if isinstance(predicted_class, (tuple, list)):
#                         predicted_class = predicted_class[0] if predicted_class else None

#                     if predicted_class:
#                         predicted_class = predicted_class.strip().lower()
#                         expected_class = expected_class.strip().lower()
#                         expected_output = expected_output.strip().lower()

#                         print(f"[{camera_name}] Expected: {expected_output}, Predicted: {predicted_class}")

#                         if predicted_class == expected_class and expected_output == 'true':
#                             if sequence == sequence_progress[room][camera_name] + 1:
#                                 print(f"✅ [{camera_name}] Sequence {sequence} validated in order")
#                                 outputEmit(sequence, predicted_class, message, frame, room)
#                                 sequence_progress[room][camera_name] = sequence  # ✅ Update sequence tracking
#                                 sequence_validated = True
#                                 break
#                             else:
#                                 print(f"⚠️ [{camera_name}] Out-of-order action detected. Expected sequence {sequence_progress[room][camera_name] + 1}.")
#                         else:
#                             error_msg = f"Incorrect action detected for sequence {sequence}. Please try again in order."
#                             print(error_msg)
#                             outputEmit(sequence, predicted_class, error_msg, frame, room)

#                 if not sequence_validated:
#                     ret, frame = cap.read()
#                     if not ret:
#                         print(f"Error reading frame from {camera_name} RTSP stream")
#                         break

#     cap.release()
#     print(f"🚪 Closing stream for {camera_name} in {room}")


