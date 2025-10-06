import torch, numpy as np
from typing import List, Dict
# import open_clip
# from open_clip import tokenizer
from ..atoms.ontology import ALL_ATOMS, text_templates_for_atom
import torch
from PIL import Image
import clip # OpenAI CLIP
from typing import List
from ..atoms.ontology import ALL_ATOMS, text_templates_for_atom
import math

class CLIPBackend:
    """
    Standard OpenAI CLIP backend.
    - model_name examples: 'ViT-B/32', 'ViT-L/14'
    """
    def __init__(self, model_name: str = 'ViT-B/32', device: str | None = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # clip.load auto-downloads weights if missing
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        self.model.eval()

        # Pre-encode text prompts per atom (average across templates), L2-normalized
        with torch.no_grad():
            text_embs = []
            owners = []
            for atom in ALL_ATOMS:
                templates = text_templates_for_atom(atom) # e.g., ["a photo of X", ...]
                tokens = clip.tokenize(templates).to(self.device)
                txt = self.model.encode_text(tokens).float()
                txt = txt / txt.norm(dim=-1, keepdim=True) # (#templates x D)
                txt_mean = txt.mean(dim=0, keepdim=True) # (1 x D)
                text_embs.append(txt_mean)
                owners.append(atom)
            self.text_embs = torch.cat(text_embs, dim=0) # (C x D)
            self.atom_list = owners

    @torch.no_grad()
    def encode_image_paths(self, image_paths: List[str], batch_size: int = 8192,
                           show_progress: bool = False) -> torch.Tensor:
        """
        Returns (T x D) image embeddings for all paths, encoded in batches.
        - L2-normalized per row.
        - Preserves the order of `image_paths`.
        """
        if len(image_paths) == 0:
            return torch.empty(0, self.text_embs.shape[1], device=self.device)  # D inferred from text_embs

        self.model.eval()

        ranges = range(0, len(image_paths), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                ranges = tqdm(ranges, total=math.ceil(len(image_paths) / batch_size), desc="CLIP encode")
            except Exception:
                pass  # fallback to plain range if tqdm isn't available

        batches = []
        for start in ranges:
            end = min(start + batch_size, len(image_paths))
            # preprocess on CPU; move to device in one go
            imgs = [self.preprocess(Image.open(p).convert("RGB")) for p in image_paths[start:end]]
            ims = torch.stack(imgs, dim=0).to(self.device, non_blocking=True)

            emb = self.model.encode_image(ims).float()  # (B x D)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            batches.append(emb)

            # optional: free memory
            del ims, emb
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return torch.cat(batches, dim=0)  # (T x D)
    # def encode_image_paths(self, image_paths: List[str]) -> torch.Tensor:
    #     imgs = [self.preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    #     ims = torch.stack(imgs, dim=0).to(self.device)
    #     img = self.model.encode_image(ims).float()
    #     img = img / img.norm(dim=-1, keepdim=True) # (T x D), unit norm
    #     return img

    def presence_scores(self, img_embs: torch.Tensor) -> torch.Tensor:
        """
        p_c(t) = cosine(img_emb_t, text_emb_c) mapped to [0, 1] via (cos+1)/2
        """
        sim = img_embs @ self.text_embs.t() # (T x C)
        return (sim + 1.0) / 2.0

# class CLIPBackend:
#     def __init__(self, model_name='ViT-B-32-quickgelu', pretrained='laion400m_e32', device=None):
#         self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
#         self.tokenizer = open_clip.get_tokenizer(model_name)
#
#         # Pre-encode text prompts per atom (average of templates)
#         with torch.no_grad():
#             texts = []
#             owners = []
#             for atom in ALL_ATOMS:
#                 tps = text_templates_for_atom(atom)
#                 toks = self.tokenizer(tps)
#                 txt = self.model.encode_text(toks.to(self.device))
#                 txt = txt / txt.norm(dim=-1, keepdim=True)
#                 txt_mean = txt.mean(dim=0, keepdim=True)  # 1 x D
#                 texts.append(txt_mean)
#                 owners.append(atom)
#             self.text_embs = torch.cat(texts, dim=0)  # C x D
#             self.atom_list = owners
#
#     @torch.no_grad()
#     def encode_image_paths(self, image_paths: List[str]) -> torch.Tensor:
#         from PIL import Image
#         ims = []
#         for p in image_paths:
#             im = Image.open(p).convert("RGB")
#             ims.append(self.preprocess(im))
#         ims = torch.stack(ims, dim=0).to(self.device)
#         img = self.model.encode_image(ims)
#         img = img / img.norm(dim=-1, keepdim=True)   # T x D
#         return img
#
#     def presence_scores(self, img_embs: torch.Tensor) -> torch.Tensor:
#         # p_c(t) = < E_V(v_t)/||.||, t_c > mapped to [0,1]
#         sim = img_embs @ self.text_embs.t()  # T x C (cosine, as both normalized)
#         return (sim + 1.0) / 2.0             # T x C in [0,1]



