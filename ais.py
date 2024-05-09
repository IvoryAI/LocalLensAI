import cv2
import time
import os
from moondream import VisionEncoder, TextModel
from PIL import Image
from huggingface_hub import snapshot_download
import openai
from openai import OpenAI
import json
import requests
import re

def save_file(filepath, content):
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)
        
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
        
mailgun_api_key = open_file('mgapikey.txt')          
        

# ANSI escape code for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

def mistral7b(user_input):
    streamed_completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "You are a expert at writing security logs"},
            {"role": "user", "content": user_input}
        ],
        stream=True  # Enable streaming
    )

    full_response = ""
    line_buffer = ""

    for chunk in streamed_completion:
        delta_content = chunk.choices[0].delta.content

        if delta_content is not None:
            line_buffer += delta_content

            if '\n' in line_buffer:
                lines = line_buffer.split('\n')
                for line in lines[:-1]:
                    print(NEON_GREEN + line + RESET_COLOR)
                    full_response += line + '\n'
                line_buffer = lines[-1]

    if line_buffer:
        print(NEON_GREEN + line_buffer + RESET_COLOR)
        full_response += line_buffer

    return full_response

def process_image(images_dir):
    # Find all images in the directory
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')]
    
    if not images:
        print("No images found in the directory.")
        return None, None

    # Sort images by modification time (or you can sort by name if they are named with a timestamp)
    latest_image = max(images, key=os.path.getmtime)

    # Remove all other images
    for image in images:
        if image != latest_image:
            os.remove(image)
            print(f"Removed {image}")

    # Now, process the latest image
    model_path = snapshot_download("vikhyatk/moondream1")
    vision_encoder = VisionEncoder(model_path)
    text_model = TextModel(model_path)

    image = Image.open(latest_image)
    image_embeds = vision_encoder(image)
    return text_model, image_embeds

def start_video_capture(stream_url, output_dir='captured_video', capture_duration=5):
    """
    Captures video from the stream for a specified duration and saves it as .mp4 using H.264 codec.
    Returns the path to the saved video file.

    :param stream_url: URL of the video stream.
    :param output_dir: Directory where the captured video will be saved.
    :param capture_duration: Duration in seconds for which to capture the video.
    :return: Path to the saved video file.
    """
    ensure_dir_exists(output_dir)  # Ensure the output directory exists

    cap = cv2.VideoCapture(stream_url)
    
    # Attempt to set the capture to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1270)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Cannot open stream")
        return None

    # Get actual video frame size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.
    # Attempt to use H.264 codec ('X264' or 'avc1')
    # Note: You might need to change this based on your system
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    
    output_path = os.path.join(output_dir, f'captured_{int(time.time())}.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
    
    start_time = time.time()
    print("Starting video capture...")

    while int(time.time() - start_time) < capture_duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video captured and saved as {output_path}")
    
    return output_path  # Return the path where the video was saved

def ensure_dir_exists(directory):
    """Ensure the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_frame(stream_url, images_dir='images'):
    """Capture a single frame from a stream and save it to a directory."""
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Cannot open stream")
        return

    ensure_dir_exists(images_dir)

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
    else:
        filename = os.path.join(images_dir, f'frame_{int(time.time())}.png')
        cv2.imwrite(filename, frame)
        print(f'Saved {filename}')

    cap.release()

# def send_email(recipient, subject, body, attachment=None):
#     data = {
#         "from": "YOU <yourmail@mail.com>",
#         "to": recipient,
#         "subject": subject,
#         "text": body,
#     }

#     if attachment:
#         with open(attachment, 'rb') as f:
#             files = {'attachment': (os.path.basename(attachment), f)}
#             response = requests.post(
#                 "https://api.eu.mailgun.net/v3/allabtai.com/messages",
#                 auth=("api", mailgun_api_key),
#                 data=data,
#                 files=files
#             )
#     else:
#         response = requests.post(
#             "https://api.eu.mailgun.net/v3/allabtai.com/messages",
#             auth=("api", mailgun_api_key),
#             data=data
#         )

#     if response.status_code != 200:
#         raise Exception("Failed to send email: " + str(response.text))

#     print("Email sent successfully.")
    
def main():
    stream_url = ''
    images_dir = "C:/Users/jason/AI Projects/vstream/Moondream/images"
    video_dir = "C:/Users/jason/AI Projects/vstream/captured_video"

    # Initialize a boolean variable
    person_detected = False

    while True:
        capture_frame(stream_url, images_dir=images_dir)
        text_model, image_embeds = process_image(images_dir)
        prompt = f"{NEON_GREEN}Is there a PERSON in the image?{RESET_COLOR} (ONLY ANSWER WITH YES OR NO)"

        print(">", prompt)
        
        # Ensure text_model and image_embeds are valid before proceeding
        if text_model is not None and image_embeds is not None:
            answer = text_model.answer_question(image_embeds, prompt).strip().upper()
        else:
            print("Could not process image properly.")
            continue  # Skip to the next iteration of the loop
        
        print(CYAN + answer + RESET_COLOR)

        # Set the boolean value based on the answer
        if answer == "YES":
            person_detected = True
            # Start video capture if a person is detected
            print(f"{NEON_GREEN}Person detected. Starting video capture.{RESET_COLOR}")
            video_path = start_video_capture(stream_url, output_dir=video_dir, capture_duration=5)
                    # Send an email with the video clip as an attachment
            # print("Sending email...")
            # recipient = "YOUR MAIL ADDRESS"
            # subject = "Video Clip"
            # body = "Here is the 5-second video clip."
            # send_email(recipient, subject, body, attachment=video_path)
            text_model, image_embeds = process_image(images_dir)
            prompt2 = "Describe the image in detail:"
            answer2 = text_model.answer_question(image_embeds, prompt2)
            log = f"Image Description: {answer2} \n From the description write a security log:"
            log2 = mistral7b(log)
            save_file("Security_log.txt", log2)
            # You might want to break or continue here depending on your application
            # break
        elif answer == "NO":
            person_detected = False
        else:
            print("Invalid answer. Please respond with YES or NO.")
            continue  # Skip the rest of the loop and ask the question again


        # Output the status of person detection
        print(f"Person detected: {person_detected}")

        # If you want to break the loop after getting a valid response, uncomment the following line
        # break


if __name__ == "__main__":
    main()    
