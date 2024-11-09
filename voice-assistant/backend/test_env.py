from dotenv import load_dotenv
import os
import json
from google.cloud import speech, texttospeech
from anthropic import Anthropic

def test_env_variables():
    print("\n1. Testing Environment Variables:")
    print("--------------------------------")
    load_dotenv()
    
    # Test Claude API Key
    claude_key = os.getenv('CLAUDE_API_KEY')
    print(f"Claude API Key exists: {bool(claude_key)}")
    print(f"Claude API Key starts with: {claude_key[:14]}...")

    # Test Google Cloud paths
    stt_path = os.getenv('GOOGLE_CLOUD_STT_CREDENTIALS_PATH')
    tts_path = os.getenv('GOOGLE_CLOUD_TTS_CREDENTIALS_PATH')
    
    print(f"\nSpeech-to-Text path: {stt_path}")
    print(f"Text-to-Speech path: {tts_path}")

def test_google_credentials():
    print("\n2. Testing Google Cloud Credentials:")
    print("--------------------------------")
    
    # Test Speech-to-Text credentials
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_CLOUD_STT_CREDENTIALS_PATH')
        speech_client = speech.SpeechClient()
        print("✅ Speech-to-Text credentials loaded successfully")
    except Exception as e:
        print("❌ Speech-to-Text credentials error:", str(e))

    # Test Text-to-Speech credentials
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_CLOUD_TTS_CREDENTIALS_PATH')
        tts_client = texttospeech.TextToSpeechClient()
        print("✅ Text-to-Speech credentials loaded successfully")
    except Exception as e:
        print("❌ Text-to-Speech credentials error:", str(e))

def test_claude_api():
    print("\n3. Testing Claude API Connection:")
    print("--------------------------------")
    try:
        anthropic = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        message = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Connection successful' in 5 words or less"
                }
            ]
        )
        print("✅ Claude API connection successful")
        print(f"Test response: {message.content}")
    except Exception as e:
        print("❌ Claude API error:", str(e))

if __name__ == "__main__":
    test_env_variables()
    test_google_credentials()
    test_claude_api()
