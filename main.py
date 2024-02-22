import panel as pn
import cv2
from ultralytics import YOLO
import numpy as np
import PIL.Image
import time
from collections import deque
import threading
import datetime
import os
import json
import requests
import PIL
from PIL import Image
import json
import requests
import panel as pn
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
import datetime

if getattr(sys, 'frozen', False):
    # Running as a bundled executable
    application_path = sys._MEIPASS
else:
    # Running as a script
    application_path = os.path.dirname(os.path.abspath(__file__))

pn.extension(theme='dark')

    # Ensure the clear_directory function is defined to actually delete the files
def clear_directory(directory='video'):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): 
                    shutil.rmtree(file_path)  
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"Directory {directory} does not exist.")



class CameraTab:
    def __init__(self, camera_id,gallery_instance=None):
        self.camera_id = camera_id
        self.monitoring = False
        self.cap = None
        self.live_view_pane = pn.pane.JPG(sizing_mode='stretch_width')
        self.gallery_instance = gallery_instance
        self.frame_buffer_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.tabexecutor = ThreadPoolExecutor()
        self.detection_time = None
        self.post_detection_frames = 30  # Number of frames to collect after detection
        self.consecutive_detections = 0
        self.previous_frame = None
        self.future = None
        self.stop_loop = threading.Event()# A shared flag to control the loop execution
        self.camera_url_tooltip = pn.widgets.TooltipIcon(value='Input 0 for webcam or RTSP/HTTP URL')
        self.alert = pn.pane.Alert('''Input 0 for webcam or RTSP/HTTP URL\n''')
        self.layout = self.create_layout()


    def create_layout(self):

        self.header = pn.pane.Markdown(f"## CCTV Monitor - Camera {self.camera_id}")
        self.camera_input = pn.widgets.TextInput(name=f"Camera {self.camera_id} URL", value="0")
        self.confidence_threshold_slider = pn.widgets.FloatSlider(name='Confidence Threshold', start=0.0, end=1.0, step=0.01, value=0.5)
        self.start_button = pn.widgets.Button(name=f"Start Monitoring Camera {self.camera_id}")
        self.stop_button = pn.widgets.Button(name=f"Stop Monitoring Camera {self.camera_id}")
        self.start_stop_switch = pn.widgets.Toggle(name='⏯️',button_type='success',button_style='outline',margin=15)
        self.start_stop_switch.param.watch(self.toggle_monitoring, 'value')
       
        return pn.Column(pn.Row(self.header, self.camera_input,self.camera_url_tooltip,self.start_stop_switch,self.confidence_threshold_slider),self.alert,pn.layout.Divider() ,self.live_view_pane)
    

    
    def load_settings(self):

        settings_file_path = 'settings.json'
        try:
            with open(settings_file_path, 'r') as file:
                settings_data = json.load(file)
                self.telegram_alert_checkbox = settings_data.get("telegram_alerts_enabled", False)
                self.telegram_api_key_input = settings_data.get("telegram_api_key", "")
                self.telegram_chat_id = settings_data.get("chat_id", "")

        except FileNotFoundError:
            print("Settings file not found. setting file path is at {settings_file_path}")


    def toggle_monitoring(self, event):

        if event.new:  # If the switch is turned on
            print('Started....')
            self.start_monitoring()
        else:  # If the switch is turned off
            print('Stopped...')
            self.stop_monitoring()
        
    def start_monitoring(self, event=None):

        try:
            model_path = os.path.join(application_path, 'yolov8n.pt')
            self.model = YOLO(model_path)
            self.frame_buffer = deque(maxlen=300)

            if self.stop_loop is None:
                self.stop_loop = threading.Event()
            else:
                # Clear the stop event in case it's being restarted
                self.stop_loop.clear()
            
            camera_input = self.camera_input.value.strip()

            # Determine the camera source based on the input or checkbox
            if camera_input == '0':
                self.camera_source = 0  # Default webcam
            else:
                self.camera_source = camera_input  # URL or other camera source
            
            # Initialize the camera
            self.cap = cv2.VideoCapture(self.camera_source)

            if not self.cap.isOpened():
                print(f"Error accessing camera {self.camera_id}")
                self.alert.object = '''Error accessing camera, Use 0 for webcam or use HTTP or RTSP URL to start monitoring.'''
                return
            
            self.future = self.tabexecutor.submit(self.update_frame)
            
        except Exception as e:
            print('start monitoring failed ',e)
    

    def stop_monitoring(self, event=None):
        if hasattr(self, 'stop_loop') and self.stop_loop is not None:
            self.stop_loop.set()
            
        if hasattr(self, 'future') and self.future is not None:
            try:
                # Optionally, add a timeout to avoid indefinite blocking
                self.future.result(timeout=10)
            except Exception as e:
                # Handle potential exceptions, e.g., TimeoutError or others
                print(f"Error waiting for the monitoring task to complete: {e}")
        
        if self.cap:
            try:
                self.cap.release()
                print('cap released....')
                self.alert.object = '''Monitoring Stopped...'''
            except Exception as e:
                print(f"Error releasing the camera: {e}")
            finally:
                self.cap = None
        # Reset the stop_loop event for potential future use
        self.stop_loop = None
        del(self.model)
        del(self.frame_buffer)

        
    def process_frame(self, frame):
        # Convert color from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, classes=[0])
        person_detected = any(confidence > self.confidence_threshold_slider.value for result in results for confidence in result.boxes.conf)
        
        thumbnail = results[0].plot()

        if person_detected:
            self.consecutive_detections += 1
            # Check if this is the first time reaching three consecutive detections
            if self.consecutive_detections >= 3 and self.detection_time is None:
                self.detection_time = time.time()

            processed_frame = results[0].plot()
            thumbnail = results[0].plot()
        else:
            self.consecutive_detections = 0
            # Show the regular frame if no detection above the threshold
            processed_frame = frame_rgb

        if self.detection_time and time.time() - self.detection_time > self.post_detection_frames:

            self.executor.submit(self.send_video,thumbnail=thumbnail,detection_time=self.detection_time)
            self.detection_time = None
            self.consecutive_detections = 0
          

        # Always add the frame to the buffer
        self.frame_buffer.append(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

        return processed_frame  # Return the frame (either with plotted detections or regular)
    
    def display_frame(self, frame):
        if isinstance(frame, np.ndarray):
 
             # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(frame)
            # Update the frame pane
            self.live_view_pane.object = image
            self.alert.object = '''Monitoring Started...'''
        else:
            print("Frame is not a valid NumPy array.")

    def update_frame(self):
        retry_attempts = 0
        max_retry_attempts = 5  # Maximum number of retry attempts before reinitializing the camera

        while not self.stop_loop.is_set():
            try:
                if self.cap:
                    success, frame = self.cap.read()
                    if success:
                        processed_frame = self.process_frame(frame)
                        self.display_frame(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))  # Convert back to BGR for display
                        retry_attempts = 0  # Reset retry counter after a successful read
                    else:
                        print("Warning: Frame read unsuccessful.")
                        self.alert.object = '''Warning: Frame read unsuccessful.'''
                        
                        retry_attempts += 1
                        if retry_attempts >= max_retry_attempts:
                            print("Attempting to reinitialize camera...")
                            self.reinitialize_camera()
                            retry_attempts = 0  # Reset retry counter after reinitialization attempt
                else:
                    raise ValueError("Camera capture object is not initialized.")
                
            except Exception as e:

                print(f"Error during frame processing or display: {e}")

    def reinitialize_camera(self):
        """Attempts to reinitialize the camera stream."""
        if self.cap:
            self.cap.release()  # Release the current capture object

        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            print(f"Failed to reinitialize camera {self.camera_id}")
        else:
            print(f"Camera {self.camera_id} reinitialized successfully.")

    def send_video(self,thumbnail,detection_time):


        def append_event_to_metadata_file(metadata_file_path, new_event_metadata):
                # Check if the metadata file exists
                if os.path.exists(metadata_file_path):
                    # If it exists, read the current contents
                    with open(metadata_file_path, 'r') as file:
                        try:
                            data = json.load(file)
                        except json.JSONDecodeError:
                            # If the file is empty or corrupted, start with an empty list
                            data = []
                else:
                    # If the file doesn't exist, start with an empty list
                    data = []

                # Append the new event metadata
                data.append(new_event_metadata)

                # Write the updated data back to the file
                with open(metadata_file_path, 'w') as file:
                    json.dump(data, file, indent=4)



        try:

            

            print('Attempeting to create a video and add it.......')
            # Check if the deque is empty before trying to access its elements
            if not self.frame_buffer:  # This will be True if the deque is empty 
                print("Frame buffer is empty. No frames to process.")
                return
            
            with self.frame_buffer_lock:  # Use the lock to safely make a copy
                buffer_copy = list(self.frame_buffer)
                self.frame_buffer.clear()

            # Now that we've checked the deque isn't empty, we can safely access its elements
            height, width, layers = buffer_copy[0].shape
            size = (width, height)


            # Check if the directory exists, if not, create it
            if not os.path.exists('video'):
                os.makedirs('video')

            if not os.path.exists('thumbnail'):
                os.makedirs('thumbnail')

            video_filename = f'video/camera_{self.camera_id}_video_{detection_time}.mp4'
            thumbnail_filename = f'thumbnail/camera_{self.camera_id}_thumbnail_{detection_time}.jpeg'
        
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(video_filename, fourcc, 10,size)
            for frame in buffer_copy:
                out.write(frame)
            out.release()

            thumbnail_image = Image.fromarray(thumbnail)
            thumbnail_image.save(thumbnail_filename)

            
            metadata = {
                    'video': video_filename,  # Adjust based on actual saving logic
                    'thumbnail': thumbnail_filename,
                    'detection_time': detection_time  # Already formatted as string
                }
            

            append_event_to_metadata_file('metadata.json',new_event_metadata=metadata)


            if self.gallery_instance:

                self.gallery_instance.add_to_flex_box(video_filename,thumbnail_filename,detection_time)

            self.load_settings()

            if self.telegram_alert_checkbox:
    
                self.send_telegram_video(bot_token=self.telegram_api_key_input, chat_id=self.telegram_chat_id, video_path=video_filename)



            del(buffer_copy)
            del(out)

        except Exception as e:

            print('failurrr in video making/sending function....',e)


    def send_telegram_video(self, bot_token, chat_id, video_path, caption=""):

        url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file} 
            data = {
                'chat_id': chat_id,
                'caption': caption,

            }
            response = requests.post(url, files=files, data=data)
        return response.json()

    def send_telegram_alert(self,bot_token):

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        payload = {
            'chat_id': self.telegram_chat_id,
            'text': 'HEY!'
        }
        response = requests.post(url, data=payload)
        return response.json()

    def view(self):
        return self.layout


class SettingsPage:

    def __init__(self):
        self.settings_file = "settings.json"
        self.content = pn.Column("# Settings", "Enter your Telegram Bot API key to receive real time updates directly on your mobile device.")
        self.telegram_alert_checkbox = pn.widgets.Checkbox(name="Enable Telegram Alerts", value=False)
        self.telegram_api_key_input = pn.widgets.TextInput(name="Telegram API Key")
        self.chat_id_display = pn.widgets.StaticText(name="Chat ID", value="")
        self.get_chat_id_button = pn.widgets.Button(name="Send Message")
    
        self.get_chat_id_button.on_click(self.get_chat_id)

        self.content.extend([
            self.telegram_alert_checkbox, 
            self.telegram_api_key_input, 
            self.get_chat_id_button,
            self.chat_id_display,
        ])
        
        self.load_settings()

    def save_settings(self):
        settings_data = {
        "telegram_alerts_enabled": self.telegram_alert_checkbox.value,
        "telegram_api_key": self.telegram_api_key_input.value,
        "chat_id": self.chat_id_display.value
        }
        with open(self.settings_file, 'w') as file:
            json.dump(settings_data, file, indent=4)

    def load_settings(self):
        settings_file_path = os.path.join(application_path, 'settings.json')
        print(settings_file_path)

        try:
            with open(settings_file_path, 'r') as file:
                settings_data = json.load(file)
                self.telegram_alert_checkbox.value = settings_data.get("telegram_alerts_enabled", False)
                self.telegram_api_key_input.value = settings_data.get("telegram_api_key", "")
                self.chat_id_display.value = settings_data.get("chat_id", "")
                print('settings file  found',settings_data)
        except FileNotFoundError:
            print("Settings file not found. setting file path is at {settings_file_path}")


    def get_chat_id(self, event):
        try:
            response_json = self.getUpdates(self.telegram_api_key_input.value)
            print(response_json)
            chat_id = response_json['result'][0]['my_chat_member']['chat']['id']
            self.chat_id_display.value = chat_id
            self.sendMessage(self.telegram_api_key_input.value,self.chat_id_display.value,'Telegram Integration Successsfull!')
            self.save_settings()
           
        except Exception as e:
            self.chat_id_display.value = f"Error: Please delete your chatbot from telegram and start a new conversation and try again!"

    def getUpdates(self, bot_token):
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        response = requests.post(url)
        print(response)
        return response.json()
    
    def sendMessage(self, bot_token, chat_id, message_text):
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message_text
        }
        response = requests.post(url, json=payload)
        return response.json()
    

    def view(self):
        return pn.Column(self.content,name='🛠️',sizing_mode='stretch_both')

class CameraPage:
    def __init__(self,chat_instance):
        self.add_camera_button = pn.widgets.Button(name="Add Camera",sizing_mode='stretch_width')
        self.camera_tabs_area = pn.Tabs(tabs_location='above')
        self.add_camera_button.on_click(self.add_camera)
        self.layout = self.camera_tabs_area
        self.chat_instance = chat_instance
        self.add_camera()

    def add_camera(self,event=None):
        new_camera_id = len(self.camera_tabs_area) + 1
        self.camera_tab = CameraTab(new_camera_id,self.chat_instance)
        self.camera_tabs_area.append((f"Live Camera View", self.camera_tab.view()))

    def view(self):
        return pn.Column(self.layout,name='📷')


class Gallery:

    def __init__(self):
        self.header = pn.pane.Markdown(f"## Gallery - Person Detection Video Clips")
        self.all_video_messages = [] 
        self.updatable_components = []
        self.flex_box = pn.FlexBox()
        self.video_box = pn.pane.Video(autoplay=True,sizing_mode='stretch_width',loop=True)
        self.delete_all_button = pn.widgets.Button(name='Delete All', button_type='danger',sizing_mode='stretch_width')
        self.delete_all_button.on_click(self.delete_all_videos)
        self.ensure_directories_exist()
        self.load_videos_from_metadata('metadata.json')
        self.start_periodic_updates()


    def delete_all_videos(self, event):
        # Clear the internal list of video messages
        self.all_video_messages.clear()
        self.updatable_components.clear()
        # Clear the display in the frontend
        self.flex_box.objects = []
        # Delete all files in the 'video' and 'thumbnail' directories
        clear_directory('video')
        clear_directory('thumbnail')
        # Clear the metadata file
        with open('metadata.json', 'w') as file:
            json.dump([], file)
        print("All videos and thumbnails have been deleted.")



    def ensure_directories_exist(self):
        required_directories = ['video', 'thumbnail']

        for directory in required_directories:
            if not os.path.exists(directory):
                print(f"Directory {directory} does not exist. Creating it.")
                os.makedirs(directory)
            else:
                print(f"Directory {directory} exists.")
        

    def time_since(self,detection_time):
        now = datetime.datetime.now()
       
        diff = now -  datetime.datetime.fromtimestamp(detection_time)
        seconds = diff.total_seconds()
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            return f"{int(seconds // 60)} mins ago"
        elif seconds < 86400:
            return f"{int(seconds // 3600)} hours ago"
        else:
            return f"{int(seconds // 86400)} days ago"
        
    def update_all_relative_times(self):
        for component, detection_time in self.updatable_components:
            component.object = self.time_since(detection_time)

    def start_periodic_updates(self):
        pn.state.add_periodic_callback(self.update_all_relative_times, period=60000)  # every minute

    def create_flex_box_item(self, video_path, thumbnail_path, detection_time):

        thumbnail_pane = pn.pane.Image(thumbnail_path, alt_text='The Panel Logo',width=200, height=150, embed=True)
        detection_datetime = datetime.datetime.fromtimestamp(detection_time)
        formatted_time = detection_datetime.strftime("%Y-%m-%d %H:%M")
        detection_time_pane = pn.pane.Markdown(f"{formatted_time}",align='center')
        relative_time_pane = pn.pane.Markdown(self.time_since(detection_time),align='center')
        self.updatable_components.append((relative_time_pane, detection_time))
        play_button = pn.widgets.Button(name='Play', button_type='primary',align='center')


        def play_video(event):
            self.video_box.object = video_path

        play_button.on_click(play_video)

        flex_box_item = pn.Column(thumbnail_pane,detection_time_pane,relative_time_pane,play_button,margin=(2, 2, 2, 2),styles={'background': '#000000'})

        
        return flex_box_item
    
    def show_video_in_box(self,video_pane,download_button):

        self.video_box.object = video_pane



    def add_to_flex_box(self, video_path, thumbnail_path, detection_time):
        flex_box_item = self.create_flex_box_item(video_path, thumbnail_path, detection_time)
        self.all_video_messages.append(flex_box_item)
        self.flex_box.objects = self.all_video_messages[::-1]

    
    
    def load_videos_from_metadata(self, metadata_file_path):
        if not os.path.exists(metadata_file_path):
            print(f"Metadata file {metadata_file_path} not found. Creating a new one.")
            with open(metadata_file_path, 'w') as file:
                json.dump([], file)  
            videos_metadata = []  
        else:
            with open(metadata_file_path, 'r') as file:
                videos_metadata = json.load(file)

        self.all_video_messages.clear()
        self.flex_box.objects = []  

        # Process each video entry in the metadata
        for entry in videos_metadata:
            video_path = entry['video']
            thumbnail_path = entry['thumbnail']
            detection_time = entry.get('detection_time', 'Unknown time')
            
            # For local development, ensure the thumbnail path is valid
            thumbnail_abs_path = os.path.abspath(thumbnail_path)
            if not os.path.exists(thumbnail_abs_path):
                print(f"Thumbnail {thumbnail_abs_path} not found.")
                continue
            
            # Create a flex box item for this video entry
            flex_box_item = self.create_flex_box_item(video_path, thumbnail_path, detection_time)

            self.all_video_messages.append(flex_box_item)
        
        self.flex_box.objects = self.all_video_messages[::-1]

    

    def update_flex_box_based_on_inputs(self, event=None):
        start_index = self.start_index_input.value
        end_index = self.end_index_input.value
        self.flex_box.objects = self.all_video_messages[start_index:end_index]

    def view(self):
        return pn.Column(self.header ,self.video_box,  self.flex_box,self.delete_all_button,pn.layout.Divider(sizing_mode='stretch_width'),name="🖼️")
        
def main():

    #logo_path = os.path.join(application_path, 'eaglevisionlogo.png')
    gallery_instance = Gallery()
    setting_page = SettingsPage()
    camera_page = CameraPage(gallery_instance)
    main_view_with_tabs = pn.Tabs(tabs_location='left',sizing_mode='stretch_both')
    main_view_with_tabs.append(camera_page.view())
    main_view_with_tabs.append(gallery_instance.view())
    #footer = pn.Row(pn.pane.Markdown("© 2024 eagle-vision.ai . All rights reserved."),align="end")
    main_view_with_tabs.append(setting_page.view())
    banner = pn.Column(pn.Spacer(height=20),pn.layout.Divider(sizing_mode='stretch_width'))
    main_area = pn.Column(banner,main_view_with_tabs)
    template = pn.template.MaterialTemplate(title='Eagle Vision CCTV',main=[main_view_with_tabs])

    #template.footer.append(footer)

    return template

main().show(port=8388,open=True)


