# 🎬 AI Video Studio

Open-source, modular AI video generation studio: text-to-video, image-to-video,
character consistency across scenes, and a full multi-scene generator that
merges everything into one final MP4 — all behind a professional Gradio UI.

Runs on **Google Colab Free (T4 16GB)**, **Colab Pro (L4/A100)**, **RunPod**,
or any local machine with an NVIDIA GPU.

---

## ✨ Features

- **Text to Video** — prompt, negative prompt, seed, resolution, duration,
  FPS, aspect ratio, guidance scale, steps.
- **Image to Video** — animate a still image with a motion-strength control.
- **Character Consistency** — save a character's face/clothes/hair/palette
  once, reuse it across every scene for a stable identity.
- **Scene Generator** — build a multi-scene storyboard (prompt, duration,
  camera motion, lighting, character, background) and auto-merge into a
  final video, with `scene_1.mp4`, `scene_2.mp4`, ..., `thumbnail.png` and
  `subtitle.srt` exported alongside it.
- **Prompt Optimizer** — expands a short prompt into a detailed cinematic
  prompt automatically.
- **Auto memory management** — detects your GPU (T4 / L4 / A10 / A100 / H100
  / CPU) and automatically enables xFormers, model CPU offload, VAE
  slicing/tiling and attention slicing, and clamps resolution/steps/frames
  to what your hardware can handle.
- **Modular model backbone** — swap between Wan 2.1, CogVideoX, LTX-Video or
  HunyuanVideo with a one-line config change.
- **Google Drive integration** (Colab) — all outputs are saved automatically
  to `MyDrive/AI-Video-Studio/`.

---

## 📁 Project structure

```
AI-Video-Studio/
├── app.py                    # Gradio application (entry point)
├── config.py                 # GPU detection, quality presets, model registry
├── requirements.txt
├── README.md
├── modules/
│   ├── text_to_video.py      # Text-to-Video pipeline wrapper
│   ├── image_to_video.py     # Image-to-Video pipeline wrapper
│   ├── scene_generator.py    # Multi-scene orchestration + merge
│   ├── character.py          # Character consistency manager
│   ├── subtitle.py           # .srt generation
│   ├── voice.py               # Optional TTS voiceover (edge-tts / pyttsx3)
│   ├── merge.py               # Scene merging (ffmpeg concat / moviepy)
│   └── utils.py               # Seeding, timers, thumbnails, history, GPU info
├── assets/
│   └── characters/           # Saved character reference sheets
├── outputs/                  # Generated videos, thumbnails, subtitles
├── models/                   # Local model cache (optional)
└── notebooks/
    └── GoogleColab.ipynb     # One-click Colab launcher
```

---

## 🚀 Installation (local / RunPod)

```bash
git clone https://github.com/your-username/AI-Video-Studio.git
cd AI-Video-Studio
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y ffmpeg   # if not already installed

python app.py
```

The app opens at `http://localhost:7860`. Models are downloaded from
Hugging Face automatically on first generation (not at startup), so the UI
opens instantly.

---

## ☁️ Google Colab

1. Open `notebooks/GoogleColab.ipynb` in Colab.
2. Runtime ▸ Change runtime type ▸ **T4 GPU** (free tier) or **A100/L4** (Pro).
3. Runtime ▸ Run all.
4. Click the `*.gradio.live` public link printed in the last cell.

The notebook automatically mounts Google Drive and symlinks `outputs/` to
`MyDrive/AI-Video-Studio/`, so every video, thumbnail and subtitle file is
saved there permanently.

---

## 🖥️ RunPod

1. Deploy a pod with an NVIDIA GPU (T4/L4/A10/A100) and the
   `runpod/pytorch` base image (CUDA + PyTorch preinstalled).
2. `git clone` this repo inside the pod.
3. `pip install -r requirements.txt` (add `apt-get install -y ffmpeg` if
   missing).
4. `python app.py` — then use RunPod's exposed HTTP port (7860) to access
   the UI, or set `share=True` in `app.py` for a temporary public URL.

---

## ⚙️ GPU configuration & auto-optimization

`config.py` detects your GPU at runtime and picks a matching quality preset:

| GPU tier | Max resolution | Max frames | Max steps | Notes |
|----------|-----------------|------------|-----------|-------|
| CPU      | 256x256         | 17         | 15        | Very slow, for testing only |
| T4 (16GB)| 512x512         | 49         | 30        | Free Colab tier |
| L4 / A10 | 768x768         | 81         | 40        | Colab Pro |
| A100     | 1024x1024       | 121        | 60        | Colab Pro+ / RunPod |
| H100     | 1280x1280       | 161        | 60        | RunPod |

xFormers, model CPU offload, VAE slicing/tiling and attention slicing are
enabled automatically according to the detected tier — no manual flags
needed. Requested settings above the tier's limits are silently clamped so
generation never OOMs.

To force a specific backbone model:

```bash
export AIVS_T2V_MODEL=cogvideox-5b       # see config.MODEL_REGISTRY for all keys
export AIVS_I2V_MODEL=cogvideox-5b-i2v
python app.py
```

---

## 🧠 Prompt optimizer

Turns a short prompt into a detailed cinematic prompt automatically:

```python
from modules.prompt_optimizer import optimize_prompt

optimize_prompt("Une femme danse.")
# → "A cinematic realistic scene showing a woman dances inside a modern
#    bedroom, with soft natural lighting, cozy atmosphere, smooth camera
#    movement, ultra detailed, 4K, HDR, cinematic composition, sharp focus."
```

Set `mode="llm"` and the `AIVS_LLM_ENDPOINT` / `AIVS_LLM_API_KEY` /
`AIVS_LLM_MODEL` environment variables to use an external LLM for richer
rewrites (falls back to the offline rule-based expander on any failure).

---

## 🧑‍🎤 Character consistency example

```python
from modules.character import Character, CharacterManager
from PIL import Image

manager = CharacterManager()
character = Character(
    name="Amina",
    description="young woman, warm smile",
    face="oval face, brown eyes, freckles",
    clothes="beige linen dress",
    hairstyle="long curly black hair",
    color_palette="warm earthy tones",
)
manager.save(character, reference_image=Image.open("amina_ref.png"))
```

Reuse `character_name="Amina"` in any scene of the Scene Generator tab —
the same descriptive fragment is appended to every scene prompt to anchor a
consistent identity.

---

## 🎞️ Exported files

Every Scene Generator run produces, inside `outputs/<job_id>/`:

```
video.mp4          # final merged video
scene_1.mp4
scene_2.mp4
scene_3.mp4
thumbnail.png       # first-frame thumbnail
subtitle.srt         # one cue per scene, timed to scene durations
```

---

## 🩹 Troubleshooting / FAQ

**`CUDA out of memory`**
Lower resolution, duration or steps, or switch to a smaller model
(`wan2.1-t2v-1.3b`, `cogvideox-2b`). The app already clamps to your GPU's
preset automatically — if you still OOM, another process may be holding
VRAM; restart the runtime.

**`xFormers` fails to install / import**
The app catches this and silently falls back to standard attention — it is
an optimization, not a hard requirement.

**Video looks blurry / low quality on a free T4**
This is expected: the T4 preset caps resolution/frames/steps to stay within
16GB VRAM. Use Colab Pro (L4/A100) or RunPod for higher fidelity.

**`ffmpeg: command not found`**
Install it: `apt-get install -y ffmpeg` (already included in the Colab
notebook's setup cell).

**How do I add a new video model?**
Add an entry to `MODEL_REGISTRY` in `config.py` with its Hugging Face
`repo_id`, then point `AIVS_T2V_MODEL` / `AIVS_I2V_MODEL` at its key — no
other code changes required, as long as the model exposes a standard
`diffusers` `DiffusionPipeline` interface.

**Can I use my own fine-tuned model?**
Yes — add a registry entry whose `repo_id` points to your Hugging Face repo
(public or private, with `HF_TOKEN` set in your environment for private
repos).

---

## 💡 Example prompts

- `A cat sitting by a rainy window, warm indoor lighting, cinematic`
- `An astronaut walking on Mars, dust storm in the background, dramatic lighting`
- `A chef plating a dessert in a modern kitchen, close-up, shallow depth of field`
- `Time-lapse of a city skyline at sunset, clouds moving fast, golden hour`

---

## 📜 License

MIT License — see below.

```
MIT License

Copyright (c) 2026 AI Video Studio contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```
