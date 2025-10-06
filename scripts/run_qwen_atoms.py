import argparse, os, json, re
from glob import glob
from PIL import Image
from tqdm import tqdm

# Optional heavy import guarded to allow --dry run
def try_import_qwen():
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch
    return AutoModelForCausalLM, AutoProcessor, torch

PROMPT = (
"Describe the frame concisely focusing on any signs of anomalies. "
"List atoms if present in two lines:\n"
"1) CONFIRMING atoms among [explosion, fireball, smoke plume, blast flash, fighting, punch, kick, wrestling, crowd brawl, shooting, gun muzzle flash, rifle aiming, pistol pointed, abuse, beating, forced restraint, choke, riot, throwing objects, looting, vandalism, crowd clash, car accident, crash, collision, deformed car, skid marks].\n"
"2) RESEMBLING atoms that do not confirm anomalies (e.g., people holding guns without flash, smoke without fire, sparks, police lights).\n"
"Answer as: 'CONFIRM: ...' and 'RESEMBLE: ...'. If none, write 'none'.\n"
)

def parse_atoms(text: str):
    confirm, resemble = [], []
    # simple robust extraction
    m1 = re.search(r'CONFIRM\s*:([^\n\r]*)', text, flags=re.I)
    m2 = re.search(r'RESEMBLE\s*:([^\n\r]*)', text, flags=re.I)
    if m1:
        part = m1.group(1)
        confirm = [x.strip().lower() for x in re.split(r'[,\uFF0C;]', part) if x.strip()]
    if m2:
        part = m2.group(1)
        resemble = [x.strip().lower() for x in re.split(r'[,\uFF0C;]', part) if x.strip()]
    # drop 'none'
    confirm = [x for x in confirm if x!='none']
    resemble = [x for x in resemble if x!='none']
    return confirm, resemble

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dry", action="store_true", help="Write empty atoms (debug)")
    args = ap.parse_args()

    frames = sorted(glob(os.path.join(args.frames_root, "*.jpg")) + glob(os.path.join(args.frames_root, "*.png")))
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    if args.dry:
        with open(args.out_json, "w") as f:
            json.dump({p: {"confirm": [], "resemble": [], "raw": ""} for p in frames}, f, indent=2)
        print("Wrote empty atoms (dry) to", args.out_json)
        return

    AutoModelForCausalLM, AutoProcessor, torch = try_import_qwen()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16 if device=='cuda' else torch.float32, device_map="auto")
    proc = AutoProcessor.from_pretrained(args.model_id)

    out = {}
    for p in tqdm(frames):
        img = Image.open(p).convert("RGB")
        inputs = proc(images=img, text=PROMPT, return_tensors="pt").to(device)
        out_ids = model.generate(**inputs, max_new_tokens=128)
        text = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
        confirm, resemble = parse_atoms(text)
        out[p] = {"confirm": confirm, "resemble": resemble, "raw": text}
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", args.out_json)

if __name__ == "__main__":
    main()
