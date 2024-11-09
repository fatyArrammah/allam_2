from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from google.cloud import speech, texttospeech
import os
from werkzeug.utils import secure_filename
import tempfile
from anthropic import Anthropic
from dotenv import load_dotenv
import json

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
    
    with open(audio_path, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        language_code="en-US",
        model="latest_long",
    )

    print("Sending request to Google Speech-to-Text")
    response = speech_client.recognize(config=config, audio=audio)
    
    if not response.results:
        raise Exception("No transcription results returned")
    
    transcript = response.results[0].alternatives[0].transcript
    print(f"Transcription result: {transcript}")
    return transcript

def get_claude_response(text):
    """Get response from Claude API"""
    print(f"Sending request to Claude API with text: {text}")
    
    message = anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.7,
        system="You are a helpful tour guide assistant for Diriyah Historical District and Boulevard Riyadh City. Provide accurate, concise information about these locations only. If asked about other places, politely remind the user that you can only assist with Diriyah and Boulevard City.",
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )
    
    # Extract the text content from the message
    response_text = message.content[0].text if message.content else "I apologize, but I couldn't process your request."
    
    print(f"Claude response received: {response_text}")
    return response_text

def text_to_speech(text):
    """Convert text to speech using Google Cloud Text-to-Speech"""
    print("Starting text-to-speech conversion")
    
    try:
        # Ensure we have a string
        if not isinstance(text, str):
            print(f"Converting text type {type(text)} to string")
            text = str(text)

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-F",  # Changed to a female voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE  # Changed from NEUTRAL to FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
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


'''Sometimes, browsers send a preflight request (an OPTIONS request) before the actual POST to check CORS permissions. Ensure that your Flask app can handle this by adding support for OPTIONS if necessary:'''
@app.route('/api/process-audio', methods=['OPTIONS'])
def options():
    return jsonify({'status': 'ok'}), 200
@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    try:
        print("Received audio processing request")
        
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
                
                print("Getting Claude response")
                claude_response = get_claude_response(transcript)
                print(f"Claude response: {claude_response}")
                
                print("Converting text to speech")
                audio_response = text_to_speech(claude_response)
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
                    'response': claude_response,
                    'audio_url': f'/audio/{response_filename}'
                })

        except Exception as e:
            print(f"Processing error: {str(e)}")
            return jsonify({'error': str(e)}), 500

        finally:
            # Cleanup temporary files
            if 'temp_audio' in locals():
                try:
                    os.unlink(temp_audio.name)
                    print(f"Cleaned up temporary file: {temp_audio.name}")
                except Exception as e:
                    print(f"Error cleaning up temporary file: {str(e)}")

    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

