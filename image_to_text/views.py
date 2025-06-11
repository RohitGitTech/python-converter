# image_to_text/views.py
from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageUploadForm, TextToImageForm
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from transformers import pipeline
from diffusers import  StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler, StableDiffusionPipeline
import torch
import os
import imageio
from transformers import BlipProcessor, BlipForConditionalGeneration
from django.shortcuts import render
import shutil
from huggingface_hub import snapshot_download


def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']

            # Ensure the media directory exists
            media_dir = 'media'
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)

            # Save the uploaded image temporarily
            input_image_path = os.path.join(media_dir, 'uploaded_image.jpg')
            with open(input_image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            img = Image.open(input_image_path).convert('RGB')

            # Load the BLIP model for image captioning
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to('cpu')

            # Generate a description
            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)

            return render(request, 'image_to_text/result_description.html', {'description': description})
    else:
        form = ImageUploadForm()
    return render(request, 'image_to_text/upload.html', {'form': form})

def generate_image(request):
    if request.method == 'POST':
        form = TextToImageForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']

            # Load the Stable Diffusion model
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
            pipe = pipe.to('cpu')  # Use 'cuda' if you have a GPU

            # Generate the image
            result = pipe(text).images[0]

            # Ensure the media directory exists
            media_dir = 'media'
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)

            # Save the image
            image_path = os.path.join(media_dir, "generated_image.png")

            result.save(image_path)

            return render(request, 'image_to_text/result_image.html', {'image_path': image_path})
    else:
        form = TextToImageForm()
    return render(request, 'image_to_text/generate_image.html', {'form': form})




def clear_huggingface_cache():
    # Define the cache directory path
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    # Remove the cache directory
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


# device = torch.device("cpu")
#
#
# model_id = "stabilityai/stable-diffusion-2-1"
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     model_id, scheduler=scheduler, torch_dtype=torch.float32
# ).to(device)

def animate_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']

            # Ensure the media directory exists
            media_dir = 'media'
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)

            # Save the uploaded image temporarily
            input_image_path = os.path.join(media_dir, 'uploaded_image.jpg')
            with open(input_image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            try:
                # Clear the Hugging Face cache
                clear_huggingface_cache()

                # Initialize the model and scheduler within the function
                model_id = "stabilityai/stable-diffusion-2-1"
                scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id, scheduler=scheduler, torch_dtype=torch.float32
                ).to("cpu")

                # Open the uploaded image and ensure it's in the correct format and size
                init_image = Image.open(input_image_path).convert("RGB")
                init_image = init_image.resize((768, 768))

                # Generate multiple frames
                frames = []
                num_frames = 2
                for i in range(num_frames):
                    prompt = f"A high-quality frame in the animation: {i}"
                    negative_prompt = 'disfigured, bad anatomy, low quality, tiling, poorly drawn hands, out of frame'
                    strength = 0.8
                    guidance_scale = 9

                    image = img2imgprompt(pipe, prompt, n=1, style=None, path=media_dir, negative_prompt=negative_prompt,
                                          init_images=[input_image_path], strength=strength, guidance_scale=guidance_scale)

                    frame_path = os.path.join(media_dir, f'frame_{i}.png')
                    image.save(frame_path)
                    frames.append(imageio.imread(frame_path))

                # Create an animated GIF
                output_animation_path = os.path.join(media_dir, 'output_animation.gif')
                imageio.mimsave(output_animation_path, frames, duration=0.3)

                return render(request, 'image_to_text/result_animation.html', {'output_animation_path': f'/media/output_animation.gif'})

            except Exception as e:
                # Handle exceptions, such as connection errors or model not found
                return render(request, 'image_to_text/error.html', {'error': str(e)})

    else:
        form = ImageUploadForm()
    return render(request, 'image_to_text/animate.html', {'form': form})

def img2imgprompt(pipe, prompt, n=1, style=None, path='.', negative_prompt=None,
                  init_images=None, strength=0.8, guidance_scale=9, seed=None):
    if style is not None:
        prompt += ' by %s' % style
    init_images = [Image.open(image).convert("RGB").resize((768, 768)) for image in init_images]
    if negative_prompt is None:
        negative_prompt = 'disfigured, bad anatomy, low quality, tiling, poorly drawn hands, out of frame'
    for c in range(n):
        if seed is None:
            currseed = torch.randint(0, 10000, (1,)).item()
        else:
            currseed = seed
        print(prompt, strength, currseed)
        generator = torch.Generator(device="cpu").manual_seed(currseed)
        image = pipe(prompt, negative_prompt=negative_prompt, image=init_images, num_inference_steps=50,
                     guidance_scale=guidance_scale, generator=generator, strength=strength).images[0]
        if not os.path.exists(path):
            os.makedirs(path)
        i = 1
        imgfile = os.path.join(path, prompt[:100] + '_%02d_%d.png' % (i, currseed))
        while os.path.exists(imgfile):
            i += 1
            imgfile = os.path.join(path, prompt[:100] + '_%02d_%d.png' % (i, currseed))
        image.save(imgfile, 'png')
    return image

def home(request):

    return render(request, 'image_to_text/home.html')