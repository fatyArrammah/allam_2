from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from google.cloud import speech, texttospeech
import os
from werkzeug.utils import secure_filename
import tempfile
from anthropic import Anthropic
from dotenv import load_dotenv
import json
import requests
# At the top level of your app.py, add this dictionary
TOPIC_OPTIONS = {
    "1": "موقع ١",
    "2": "موقع ٢",
    "3": "موقع ٣",
    "4": "موقع ٤"
}


# Load environment variables
load_dotenv()

app = Flask(__name__)
# Simple CORS setup
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


# Configure upload folder
UPLOAD_FOLDER = 'temp_audio'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm', 'ogg', 'opus'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize clients
anthropic = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))

# Configure Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_CLOUD_STT_CREDENTIALS_PATH')
speech_client = speech.SpeechClient()

# Configure Text-to-Speech
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_CLOUD_TTS_CREDENTIALS_PATH')
tts_client = texttospeech.TextToSpeechClient()

def transcribe_audio(audio_path):
    """Convert speech to text using Google Cloud Speech-to-Text"""
    print(f"Starting transcription for file: {audio_path}")
    
    try:
        with open(audio_path, 'rb') as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        
        # Updated config with correct sample rate for WebM OPUS
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            # Remove sample_rate_hertz as it's included in the WebM header
            language_code="ar-SA",  # Primary language Arabic
            alternative_language_codes=["en-US"],  # Also recognize English
            model="latest_long",
            enable_automatic_punctuation=True,
            audio_channel_count=1,  # Specify mono audio
        )

        print("Sending request to Google Speech-to-Text")
        response = speech_client.recognize(config=config, audio=audio)
        
        if not response.results:
            raise Exception("No transcription results returned")
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "
        
        print(f"Transcription result: {transcript.strip()}")
        return transcript.strip()
        
    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")
        raise Exception(f"Transcription error: {str(e)}")



def get_claude_response(text):
    """Get response from Claude API in Arabic"""
    print(f"Sending request to Claude API with text: {text}")
    
    message = anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.7,
        system="""You are a helpful tour guide assistant for Diriyah Historical District in Riyadh City. 
        Always respond in Arabic only. Provide accurate, concise information about these locations only. 
        If asked about other places, politely remind the user in Arabic that you can only assist with Diriyah and Boulevard City.
        Make your responses natural and conversational in Modern Standard Arabic.""",
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )
  # Load environment variables
load_dotenv()

# Function to get IBM access token
def get_access_token(api_key):
    token_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    
    try:
        response = requests.post(token_url, headers=headers, data=data, timeout=60)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.RequestException as e:
        print(f"Error obtaining access token: {str(e)}")
        raise Exception(f"IBM API Error: {str(e)}")

def get_allam_response(text):
    """Get response from ALLAM model"""
    try:
        print(f"Sending request to ALLAM API with text: {text}")
        
        # Get access token
        api_key = os.getenv('IBM_API_KEY')
        if not api_key:
            raise Exception("IBM_API_KEY not found in environment variables")
            
        access_token = get_access_token(api_key)
        
        url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
        
        # Prepare the prompt
        prompt = f"""Please act as a tour guide for Diriyah area in Riyadh city. 
        The user asks: {text}
        Please provide information only about these locations. If asked about other places, 
        politely remind that you can only assist with Diriyah area.
        Please respond in Arabic."""

        body = {
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 200,
                "repetition_penalty": 1
            },
            "model_id": "sdaia/allam-1-13b-instruct",
            "project_id": "95e3936c-2b13-4aa4-a02a-2bf9da54e20b",
            "input": prompt
        }
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }

        response = requests.post(url, headers=headers, json=body)
        
        if response.status_code != 200:
            raise Exception(f"ALLAM API Error: {response.text}")

        data = response.json()
        generated_text = data.get('results', [{}])[0].get('generated_text', "عذراً، لم أستطع معالجة طلبك.")
        
        print(f"ALLAM response received: {generated_text}")
        return generated_text

    except Exception as e:
        print(f"Error in get_allam_response: {str(e)}")
        raise Exception(f"ALLAM API Error: {str(e)}")

'''Sometimes, browsers send a preflight request (an OPTIONS request) before the actual POST to check CORS permissions. Ensure that your Flask app can handle this by adding support for OPTIONS if necessary:'''
@app.route('/api/process-audio', methods=['OPTIONS'])
def options():
    return jsonify({'status': 'ok'}), 200

# Update your process_audio route
@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    try:
        print("Received audio processing request")
        
        # Get and store the selected option
        selected_option = request.form.get('selectedOption', '1')
        selected_topic = TOPIC_OPTIONS.get(selected_option)
        print(f"Selected option: {selected_option}, Topic: {selected_topic}")
        
        if 'audio' not in request.files:
            print("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(audio_file.filename):
            print(f"Invalid file type: {audio_file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400

        try:
            # Create temporary files for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
                print(f"Saving audio to temporary file: {temp_audio.name}")
                audio_file.save(temp_audio.name)
                
                # Process the audio
                print("Starting speech-to-text conversion")
                transcript = transcribe_audio(temp_audio.name)
                print(f"Transcript: {transcript}")
                
                # For now, we're just storing the selected option but not using it
                print("Getting ALLAM response")
                allam_response = get_allam_response(transcript)  # Keep the original function call
                print(f"ALLAM response: {allam_response}")
                
                print("Converting text to speech")
                audio_response = text_to_speech(allam_response)
                print("Text-to-speech conversion complete")

                # Save the response audio
                response_filename = f"response_{secure_filename(audio_file.filename)}.mp3"
                response_path = os.path.join(app.config['UPLOAD_FOLDER'], response_filename)
                
                print(f"Saving response audio to: {response_path}")
                with open(response_path, 'wb') as f:
                    f.write(audio_response)

                return jsonify({
                    'success': True,
                    'transcript': transcript,
                    'response': allam_response,
                    'audio_url': f'/audio/{response_filename}'
                })

        except Exception as e:
            print(f"Processing error: {str(e)}")
            return jsonify({'error': str(e)}), 500

        finally:
            if 'temp_audio' in locals():
                try:
                    os.unlink(temp_audio.name)
                    print(f"Cleaned up temporary file: {temp_audio.name}")
                except Exception as e:
                    print(f"Error cleaning up temporary file: {str(e)}")

    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': str(e)}), 500

      
def get_allam_response(text):
    """Get response from Allam API in Arabic"""
    print(f"Sending request to Allam API with text: {text}")
    
    message = anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.7,
        system="""You are a helpful tour guide assistant for Diriyah Historical District and Boulevard Riyadh City. 
        Always respond in Arabic only. Provide accurate, concise information about these locations only. 
        If asked about other places, politely remind the user in Arabic that you can only assist with Diriyah and Boulevard City.
        Make your responses natural and conversational in Modern Standard Arabic.""",
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )
    # Extract the text content from the message
    response_text = message.content[0].text if message.content else "عذراً، لم أستطع معالجة طلبك."
    
    print(f"Claude response received: {response_text}")
    return response_text

def text_to_speech(text):
    """Convert text to speech using Google Cloud Text-to-Speech with enhanced Arabic voice"""
    print("Starting text-to-speech conversion")
    
    try:
        if not isinstance(text, str):
            print(f"Converting text type {type(text)} to string")
            text = str(text)

        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Using WaveNet voice for higher quality
        voice = texttospeech.VoiceSelectionParams(
            language_code="ar-XA",
            name="ar-XA-Wavenet-B",  # Male voice - most natural sounding
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )

        # Optimized audio configuration for Arabic
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # Slightly slower for better clarity
            pitch=0,  # Natural pitch
            volume_gain_db=0.0,  # Normal volume
            effects_profile_id=["telephony-class-application"],  # Better audio quality
            sample_rate_hertz=24000  # Higher sample rate for better quality
        )

        print("Sending request to Google Text-to-Speech")
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        print("Text-to-speech conversion completed")
        return response.audio_content
        
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        print(f"Input text type: {type(text)}")
        print(f"Input text: {text}")
        raise Exception(f"Text-to-speech conversion failed: {str(e)}")

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

