# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:23:38 2025

@author: Administrator/Ambrish
"""

from flask import Flask, request, jsonify
import json
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
import requests
import os
import time
import logging
from google.api_core import exceptions
from google.api_core.retry import Retry
from google.api_core.exceptions import FailedPrecondition, InternalServerError
from google.cloud import aiplatform_v1

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Allow Unicode characters in JSON responses

# Initialize Vertex AI
try:
    vertexai.init(project="patientsafe", location="us-central1")
    logger.info("Vertex AI initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI: {str(e)}")

def handle_safety_block(response):
    """Handle cases where content is blocked by safety filters."""
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if candidate.finish_reason == "SAFETY":
            # Log safety ratings for analysis
            safety_ratings = candidate.safety_ratings
            logger.warning(f"Content blocked by safety filters. Ratings: {safety_ratings}")
            
            # Return a specific error message for safety blocks
            return "Content blocked by safety filters. Please review the audio content."
    return None

def translate_text_to_english(text):
    """Translate text from any language to English using Gemini 2.5 Pro."""
    try:
        logger.info("Initializing model for translation")
        model = GenerativeModel("gemini-2.5-pro")
        
        prompt = f"""
        Please translate the following text to English. If the text is already in English, just return it as is.
        Guidelines:
        - Maintain professional and medical terminology accuracy
        - Preserve the original meaning and context
        - Use clear and natural English
        - If there are any technical terms, keep them accurate
        
        Text to translate:
        {text}
        """
        
        logger.info("Generating translation")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
        )
        
        # Check for safety blocks
        safety_message = handle_safety_block(response)
        if safety_message:
            return safety_message
            
        result = response.text if hasattr(response, 'text') else ""
        
        if not result:
            return "Unable to translate the text. Please check the input content."
            
        return result
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise

def generate(audio_file_url, direct_to_english=True):
    try:
        logger.info("Initializing Gemini 2.5 Pro model")
        model = GenerativeModel("gemini-2.5-pro")
        
        logger.info("Downloading audio file")
        response = requests.get(audio_file_url, verify=False, timeout=30)
        response.raise_for_status()
        
        audio_data = response.content
        logger.info(f"Audio file downloaded, size: {len(audio_data)} bytes")
        
        # Determine MIME type based on file extension
        if audio_file_url.lower().endswith('.m4a'):
            mime_type = "audio/mp4"
        elif audio_file_url.lower().endswith('.ogg'):
            mime_type = "audio/ogg"
        elif audio_file_url.lower().endswith('.mp3'):
            mime_type = "audio/mpeg"
        elif audio_file_url.lower().endswith('.wav'):
            mime_type = "audio/wav"
        elif audio_file_url.lower().endswith('.aac'):
            mime_type = "audio/aac"
        elif audio_file_url.lower().endswith('.flac'):
            mime_type = "audio/flac"
        else:
            mime_type = "audio/mpeg"  # Default fallback
        
        logger.info(f"Creating audio part with MIME type: {mime_type}")
        audio_part = Part.from_data(
            mime_type=mime_type,
            data=base64.b64encode(audio_data).decode('utf-8')
        )
        
        logger.info("Generating content")
        if direct_to_english:   
            prompt = """
            highly accurate transcription assistant. Please listen to this audio and transcribe it DIRECTLY TO ENGLISH.
            Even if the original audio is in a different language, provide only the English translation. DO NOT transcribe the original words phonetically.
            
            This is MEDICAL CONTENT with specific requirements:
            
            Guidelines:
            - Translate the content directly to clear, fluent English
            - Preserve all medical terms and drug names exactly as spoken
            - For drug names: maintain exact pronunciation (e.g., "Bimatoprost" not "Bimat")
            - For medical symptoms: translate precisely, especially for pain descriptions
            - Use medical terminology appropriate for patient descriptions, experiences
            - Indicate speaker changes if multiple speakers are present
            - Preserve the patient's description of their experience in natural English
            - Ignore background noise or interruptions, Provide ONLY the patient's words without any introductory text, analysis, or commentary.
            - Format the text clearly with proper punctuation
            
            Focus on:
            - Accurate translation of medical content and drug names
            - Natural, fluent English that preserves the meaning
            - Precise translation of symptoms and medical experiences
            - Clear formatting with proper punctuation
            """
        else:
            prompt = """
            Please transcribe this audio in its original language without translating it to English.
            Guidelines:
            - Maintain the original language as spoken
            - Ignore background noise or interruptions
            - Format the text clearly with proper punctuation
            - If multiple speakers are present, indicate speaker changes
            """
        try:
            # First attempt with streaming
            responses = model.generate_content(
                [prompt, audio_part],
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 4096,
                },
                stream=True
            )
            
            result = ""
            for response in responses:
                # Check for safety blocks
                safety_message = handle_safety_block(response)
                if safety_message:
                    return safety_message
                    
                if hasattr(response, 'text'):
                    result += response.text
            
            if not result:
                # If streaming failed, try without streaming
                logger.info("Retrying without streaming")
                response = model.generate_content(
                    [prompt, audio_part],
                    generation_config={
                        "temperature": 0.2,
                        "top_p": 0.8,
                        "top_k": 40,
                        "max_output_tokens": 4096,
                    }
                )
                
                safety_message = handle_safety_block(response)
                if safety_message:
                    return safety_message
                    
                result = response.text if hasattr(response, 'text') else ""
            
            if not result:
                return "Unable to transcribe the audio. Please check the audio quality and content."
                
            return result
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}", exc_info=True)
            # If we get a safety filter error, try with a more neutral prompt
            if "safety" in str(e).lower():
                logger.info("Retrying with neutral prompt")
                response = model.generate_content(
                    [
                        "Please listen to this audio and provide an English translation of the content.", 
                        audio_part
                    ],
                    generation_config={
                        "temperature": 0.1,
                        "top_p": 0.7,
                        "top_k": 30,
                        "max_output_tokens": 4096,
                    }
                )
                return response.text if hasattr(response, 'text') else "Transcription failed"
            raise
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading audio file: {str(e)}", exc_info=True)
        raise Exception(f"Failed to download audio file: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in generate function: {str(e)}", exc_info=True)
        raise

# Root route for testing
@app.route('/', methods=['GET'])
def home():
    return json.dumps({
        "message": "Audio Transcription API is running with Gemini 2.5 Pro",
            "model": "gemini-2.5-pro",
        "available_routes": [
            "/generate - Transcribe audio directly to English"
        ]
    }, ensure_ascii=False, indent=2), 200, {'Content-Type': 'application/json; charset=utf-8'}

# Route listing for debugging
@app.route('/routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "rule": str(rule)
        })
    return jsonify({"routes": routes})

@app.route('/generate', methods=['GET','POST'])
def generate_content():
    try:
        audio_file_url = request.args.get('text')
        logger.info(f"Received request for URL: {audio_file_url}")
        print("RECEIVED AUDIO FILE URL", audio_file_url)
        
        if not audio_file_url:
            return json.dumps({"error": "No audio file URL provided"}, ensure_ascii=False), 400, {'Content-Type': 'application/json; charset=utf-8'}
        
        result = generate(audio_file_url, direct_to_english=True)
        logger.info("Generation completed successfully")
        print("TRANSCRIPTION RESULT:", result)
        sys.stdout.flush()  # Ensures it gets written immediately




        return json.dumps({"result": result}, ensure_ascii=False, indent=2), 200, {'Content-Type': 'application/json; charset=utf-8'}
    
    except Exception as e:
        logger.error(f"Error in generate_content: {str(e)}", exc_info=True)
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "error_details": {
                "message": str(e),
                "type": str(type(e))
            }
        }, ensure_ascii=False, indent=2), 500, {'Content-Type': 'application/json; charset=utf-8'}
        
@app.route('/transcribe_original', methods=['GET'])
def transcribe_original_language():
    try:
        audio_file_url = request.args.get('text')
        logger.info(f"Received request for URL: {audio_file_url}")
        
        if not audio_file_url:
            return json.dumps({"error": "No audio file URL provided"}, ensure_ascii=False), 400, {'Content-Type': 'application/json; charset=utf-8'}
        
        result = generate(audio_file_url, direct_to_english=False)
        logger.info("Generation completed successfully")
        return json.dumps({"result": result}, ensure_ascii=False, indent=2), 200, {'Content-Type': 'application/json; charset=utf-8'}
    
    except Exception as e:
        logger.error(f"Error in transcribe_original_language: {str(e)}", exc_info=True)
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "error_details": {
                "message": str(e),
                "type": str(type(e))
            }
        }, ensure_ascii=False, indent=2), 500, {'Content-Type': 'application/json; charset=utf-8'}

@app.route('/translate_to_english', methods=['GET', 'POST'])
def translate_to_english():
    try:
        # Support both GET and POST methods
        if request.method == 'GET':
            text_to_translate = request.args.get('text')
        else:  # POST
            if request.is_json:
                data = request.get_json()
                text_to_translate = data.get('text') if data else None
            else:
                text_to_translate = request.form.get('text')
        
        logger.info(f"Received translation request for text: {text_to_translate[:100] if text_to_translate else 'None'}...")
        
        if not text_to_translate:
            return json.dumps({"error": "No text provided for translation"}, ensure_ascii=False), 400, {'Content-Type': 'application/json; charset=utf-8'}
        
        result = translate_text_to_english(text_to_translate)
        logger.info("Translation completed successfully")
        return json.dumps({"result": result}, ensure_ascii=False, indent=2), 200, {'Content-Type': 'application/json; charset=utf-8'}
    
    except Exception as e:
        logger.error(f"Error in translate_to_english: {str(e)}", exc_info=True)
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "error_details": {
                "message": str(e),
                "type": str(type(e))
            }
        }, ensure_ascii=False, indent=2), 500, {'Content-Type': 'application/json; charset=utf-8'}

@app.route('/transcribe_and_translate', methods=['GET', 'POST'])
def transcribe_and_translate():
    """Combined endpoint: transcribe audio in original language, then translate to English"""
    try:
        audio_file_url = request.args.get('text') if request.method == 'GET' else request.form.get('text')
        logger.info(f"Received request for transcribe and translate: {audio_file_url}")
        
        if not audio_file_url:
            return json.dumps({"error": "No audio file URL provided"}, ensure_ascii=False), 400, {'Content-Type': 'application/json; charset=utf-8'}
        
        # Step 1: Transcribe in original language
        logger.info("Step 1: Transcribing in original language")
        original_transcription = generate(audio_file_url, direct_to_english=False)
        
        # Step 2: Translate to English
        logger.info("Step 2: Translating to English")
        english_translation = translate_text_to_english(original_transcription)
        
        logger.info("Transcription and translation completed successfully")
        return json.dumps({
            "original_transcription": original_transcription,
            "english_translation": english_translation
        }, ensure_ascii=False, indent=2), 200, {'Content-Type': 'application/json; charset=utf-8'}
    
    except Exception as e:
        logger.error(f"Error in transcribe_and_translate: {str(e)}", exc_info=True)
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "error_details": {
                "message": str(e),
                "type": str(type(e))
            }
        }, ensure_ascii=False, indent=2), 500, {'Content-Type': 'application/json; charset=utf-8'}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)



