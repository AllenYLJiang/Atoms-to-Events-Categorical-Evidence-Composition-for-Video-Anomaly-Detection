"""
Optional Qwen2.5-VL-3B captioner.
If you pass --dry_caption, the pipeline will skip calling this file and will
mock captions for speed. Otherwise, we show how to use HF Transformers.
"""
from typing import List
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch

PROMPT = (
"Describe the frame **concisely** focusing on any signs of anomalies. "
"List atoms if present in two lines:\n"
"1) CONFIRMING atoms among [explosion, fireball, smoke plume, blast flash, fighting, punch, kick, wrestling, crowd brawl, shooting, gun muzzle flash, rifle aiming, pistol pointed, abuse, beating, forced restraint, choke, riot, throwing objects, looting, vandalism, crowd clash, car accident, crash, collision, deformed car, skid marks].\n"
"2) RESEMBLING atoms that do not confirm anomalies (e.g., people holding guns without flash, smoke without fire, sparks, police lights).\n"
"Answer as: 'CONFIRM: ...' and 'RESEMBLE: ...'. If none, write 'none'.\n"
)

class QwenCaptioner:
    def __init__(self, device=None, model_id='Qwen/Qwen2.5-VL-3B-Instruct'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def describe(self, image_paths: List[str]) -> List[str]:
        outs = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            inputs = self.processor(images=img, text=PROMPT, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=128)
            text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            outs.append(text)
        return outs
