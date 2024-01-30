# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import time
import torch
from PIL import Image

from weights_downloader import WeightsDownloader
from honeybee.pipeline.interface import get_model

MODEL_NAME = "kakaobrain/honeybee_7B-C-Abs-M256"
MODEL_CACHE = "model-cache"
MODEL_URL = "https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-C-Abs-M256.tar.gz"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        WeightsDownloader.download_if_not_exists(MODEL_URL, MODEL_CACHE)

        ckpt_path = "checkpoints/7B-C-Abs-M256/last"

        start = time.time()
        print("Loading pipeline...")
        self.model, self.tokenizer, self.processor = get_model(ckpt_path, use_bf16=True)
        self.model.cuda()
        print("setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt"),
        max_length: int = Input(
            description="Maximum number of tokens to generate", default=512,
        ),
        top_k: int = Input(
            description="top k for sampling", default=5,
        ),
        do_sample: bool = Input(
            description="Whether you do sampling or not",
            default=True,
        ),
        agree_to_research_only: bool = Input(
            description="You must agree to use this model only for research. It is not for commercial use.",
            default=True,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        if not agree_to_research_only:
            raise Exception(
                "You must agree to use this model for research-only, you cannot use this model comercially."
            )
        prompt_list = [prompt_template(prompt)]
        image_list = [Image.open(image)]
        
        inputs = self.processor(text=prompt_list, images=image_list)
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        generate_kwargs = {
            'do_sample': do_sample,
            'top_k': top_k,
            'max_length': max_length
        }
        
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        output = self.tokenizer.batch_decode(res, skip_special_tokens=True)[0]

        return output
    
def prompt_template(prompt):
    SYSTEM_MESSAGE = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    IMAGE_TOKEN = "Human: <image>\n" #<image> denotes an image placehold.
    USER_PROMPT = f"Human: {prompt}\n"

    return SYSTEM_MESSAGE + IMAGE_TOKEN + USER_PROMPT + "AI: "