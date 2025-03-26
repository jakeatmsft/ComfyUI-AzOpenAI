import io
import os
import openai
import base64
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo"
]

class OpenAICaptionImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in" : ("IMAGE", {}),
                "model": (MODELS, ),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant."}),
                "caption_prompt": ("STRING", {"default": "What's in this image?"}),
                "max_tokens": ("INT", {"default": 300}),
                "temperature": ("FLOAT", {"default": 0.5}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    CATEGORY = "openai"
    FUNCTION = "caption"

    def caption(self, image_in, model, system_prompt, caption_prompt, max_tokens, temperature):
        # image to base64, image is bwhc tensor

        # Convert tensor to PIL Image
        pil_image = Image.fromarray(np.clip(255. * image_in.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Set up OpenAI client
        client = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version = "2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            )

        # Make API call to OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response.choices[0].message.content is None:
            raise ValueError("No content in response")

        # Extract and return the caption
        caption = response.choices[0].message.content.strip()
        return (caption,)
