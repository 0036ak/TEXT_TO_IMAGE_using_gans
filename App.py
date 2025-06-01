# app.py
import json
import os
import queue
import threading
import time
import uuid

import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from flask import (Flask, Response, jsonify, render_template, request,
                   send_from_directory, stream_with_context)
from PIL import Image

# Import your GAN's Generator and DataLoader (ensure these are correctly set up)
from models.generator import Generator
from utils.data_loader import DataLoader

# --- Configuration for GAN Generation ---
GAN_LATENT_DIM = 100
GAN_EMBEDDING_DIM = 256
GAN_CAPTION_ENCODER_HIDDEN_SIZE = 256
GAN_IMAGE_SHAPE = (3, 64, 64)
GAN_MODEL_PATH = './saved_models/generator_final.pth'

GAN_IMAGE_DIR_FOR_VOCAB = './data/Images'
GAN_CAPTION_FILE_FOR_VOCAB = './data/captions/captions.txt'
GAN_MAX_CAPTION_LENGTH = 50

# --- Configuration for Stable Diffusion Refinement ---
REFINED_OUTPUT_DIR = 'refined_images'
os.makedirs(REFINED_OUTPUT_DIR, exist_ok=True)

SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

IMG2IMG_STRENGTH = 0.55
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 8.0
NEGATIVE_PROMPT = "blurry, grainy, low quality, deformed, ugly, distorted, noise, jpeg artifacts, text, signature, watermark"
OUTPUT_HEIGHT = 512
OUTPUT_WIDTH = 512

# --- Device Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Global storage for progress queues ---
progress_queues = {}

# --- Estimated Time Tracking ---
# Simple file to store historical times
TIME_LOG_FILE = 'generation_times.json'
MAX_HISTORY_TIMES = 20 # Keep a history of the last 20 generation times

def load_historical_times():
    if os.path.exists(TIME_LOG_FILE):
        with open(TIME_LOG_FILE, 'r') as f:
            return json.load(f)
    return []

def save_historical_times(times):
    with open(TIME_LOG_FILE, 'w') as f:
        json.dump(times, f)

def add_generation_time(new_time):
    times = load_historical_times()
    times.append(new_time)
    # Keep only the latest MAX_HISTORY_TIMES
    times = times[-MAX_HISTORY_TIMES:]
    save_historical_times(times)

def get_estimated_time():
    times = load_historical_times()
    if not times:
        return None # No history yet
    return round(sum(times) / len(times), 2)

# --- Initialize GAN DataLoader (for vocabulary) ---
print("Initializing GAN DataLoader for vocabulary...")
try:
    gan_data_loader = DataLoader(
        image_dir=GAN_IMAGE_DIR_FOR_VOCAB,
        caption_file=GAN_CAPTION_FILE_FOR_VOCAB,
        image_size=GAN_IMAGE_SHAPE[1:],
        min_word_freq=5
    )
    GAN_VOCAB_SIZE = gan_data_loader.vocab_size
    print(f"GAN DataLoader initialized. Vocab size: {GAN_VOCAB_SIZE}")
except Exception as e:
    print(f"Error initializing GAN DataLoader: {e}")
    print("Please ensure GAN_IMAGE_DIR_FOR_VOCAB and GAN_CAPTION_FILE_FOR_VOCAB are correct.")
    # Exit or handle gracefully if critical
    # For now, we'll let it proceed for testing other parts, but this will fail later
    GAN_VOCAB_SIZE = 100 # Fallback if DataLoader init fails for testing
    gan_data_loader = None # Ensure it's None if init failed

# --- Initialize GAN Generator Model ---
print("Initializing GAN Generator model...")
try:
    gan_generator = Generator(
        noise_dim=GAN_LATENT_DIM,
        vocab_size=GAN_VOCAB_SIZE,
        embedding_dim=GAN_EMBEDDING_DIM,
        caption_encoder_hidden_size=GAN_CAPTION_ENCODER_HIDDEN_SIZE,
        output_channels=GAN_IMAGE_SHAPE[0]
    ).to(DEVICE)

    if os.path.exists(GAN_MODEL_PATH):
        gan_generator.load_state_dict(torch.load(GAN_MODEL_PATH, map_location=DEVICE))
        gan_generator.eval() # Set to evaluation mode
        print(f"Successfully loaded GAN Generator from {GAN_MODEL_PATH}")
    else:
        print(f"Error: GAN Generator model not found at {GAN_MODEL_PATH}. Please check the path.")
        # Exit or handle gracefully if critical
        gan_generator = None # Ensure it's None if init failed
except Exception as e:
    print(f"Error initializing GAN Generator: {e}")
    gan_generator = None


# --- Initialize Stable Diffusion Image-to-Image Pipeline ---
print(f"Loading Stable Diffusion Image-to-Image model: {SD_MODEL_ID}...")
try:
    if "xl" in SD_MODEL_ID:
        sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16)
    else:
        sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL_ID)
    sd_pipeline.to(DEVICE)
    print("Stable Diffusion model loaded successfully for Flask app.")
except Exception as e:
    print(f"Error loading Stable Diffusion model for Flask app: {e}")
    print("Ensure you have enough VRAM or consider running on CPU (will be very slow).")
    sd_pipeline = None # Ensure it's None if init failed


# --- Flask App Setup ---
app = Flask(__name__)

# --- Prompt Enhancement Function ---
def enhance_prompt(original_caption):
    enhanced_prompt = (
        f"{original_caption.strip()}, "
        "detailed, high quality, sharp focus, vibrant colors, clear, "
        "natural lighting, good composition."
    )
    return enhanced_prompt

# --- Background generation task ---
def generation_task(job_id, caption):
    q = progress_queues[job_id]
    start_time = time.time()

    try:
        # Check if models are loaded. If not, send error and exit.
        if gan_data_loader is None or gan_generator is None or sd_pipeline is None:
            raise RuntimeError("One or more AI models failed to load. Please check server logs.")

        # Stage 1: Generate initial low-res image with GAN
        q.put({"status": "GAN generation started", "progress": 5})
        with torch.no_grad():
            z = torch.randn(1, GAN_LATENT_DIM, device=DEVICE)
            encoded_caption_np = gan_data_loader.tokenize_and_encode_caption(caption, max_len=GAN_MAX_CAPTION_LENGTH)
            encoded_caption_tensor = torch.tensor(encoded_caption_np, dtype=torch.long).unsqueeze(0).to(DEVICE)

            gan_generated_img_tensor = gan_generator(z, encoded_caption_tensor)

            gan_generated_img_np = (gan_generated_img_tensor.squeeze(0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
            gan_generated_img_pil = Image.fromarray(np.transpose(gan_generated_img_np, (1, 2, 0)))
        q.put({"status": "GAN generation complete", "progress": 10})

        # Stage 2: Refine with Stable Diffusion Image-to-Image
        enhanced_caption = enhance_prompt(caption)
        init_image_for_sd = gan_generated_img_pil.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)

        def sd_progress_callback(step, timestep, latents):
            current_progress = 10 + int((step / NUM_INFERENCE_STEPS) * 80)
            q.put({"status": f"Refining image (Step {step}/{NUM_INFERENCE_STEPS})", "progress": current_progress})

        refined_image_pil = sd_pipeline(
            prompt=enhanced_caption,
            negative_prompt=NEGATIVE_PROMPT,
            image=init_image_for_sd,
            strength=IMG2IMG_STRENGTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=OUTPUT_HEIGHT,
            width=OUTPUT_WIDTH,
            callback=sd_progress_callback,
            callback_steps=1
        ).images[0]

        image_id = f"refined_final_{uuid.uuid4().hex}"
        img_filename = f'{image_id}.png'
        img_filepath = os.path.join(REFINED_OUTPUT_DIR, img_filename)
        refined_image_pil.save(img_filepath)

        elapsed_time = time.time() - start_time
        print(f"Job {job_id}: Total elapsed time calculated: {elapsed_time:.2f} seconds")
        add_generation_time(elapsed_time) # Add to historical data

        q.put({
            "status": "Image generation and refinement complete!",
            "progress": 100,
            "image_url": f'/refined_images/{img_filename}',
            "original_caption": caption,
            "time_taken": round(elapsed_time, 2),
            "done": True
        })

    except Exception as e:
        print(f"Error in generation_task for job {job_id}: {e}")
        q.put({"status": f"Error: {e}", "progress": -1, "done": True})
    finally:
        q.put(None)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_and_refine_image', methods=['POST'])
def generate_and_refine_image_request():
    caption = request.json.get('caption')
    if not caption:
        return jsonify({'error': 'Caption is required'}), 400

    job_id = str(uuid.uuid4())
    progress_queues[job_id] = queue.Queue()

    # Get estimated time before starting the thread
    estimated_time = get_estimated_time()

    thread = threading.Thread(target=generation_task, args=(job_id, caption))
    thread.start()

    return jsonify({'job_id': job_id, 'estimated_time': estimated_time})

@app.route('/progress/<job_id>')
def progress(job_id):
    if job_id not in progress_queues:
        return Response(json.dumps({"status": "Error: Invalid job ID", "progress": -1}), mimetype='application/json'), 404

    @stream_with_context
    def generate_events():
        q = progress_queues[job_id]
        while True:
            try:
                message = q.get(timeout=60)
                if message is None:
                    break
                yield f"data: {json.dumps(message)}\n\n"
                if message.get("done"):
                    break
            except queue.Empty:
                yield ": keep-alive\n\n"
            except Exception as e:
                print(f"Error in SSE stream for job {job_id}: {e}")
                break
        if job_id in progress_queues:
            del progress_queues[job_id]

    return Response(generate_events(), mimetype='text/event-stream')

@app.route('/refined_images/<filename>')
def serve_refined_image(filename):
    return send_from_directory(REFINED_OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)