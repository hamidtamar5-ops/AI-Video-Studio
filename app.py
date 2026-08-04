"""
app.py
======
AI Video Studio — Gradio application entry point.

Run locally:
    pip install -r requirements.txt
    python app.py

Run in Google Colab:
    see notebooks/GoogleColab.ipynb

Tabs:
    1. Text to Video          - modules/text_to_video.py
    2. Image to Video         - modules/image_to_video.py
    3. Character Consistency  - modules/character.py
    4. Scene Generator        - modules/scene_generator.py (+ merge/subtitle)

The heavy model pipelines are loaded lazily (on first generate click), so
the UI opens instantly even before any model weights are downloaded.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import gradio as gr

from config import (
    APP_TITLE,
    APP_VERSION,
    DEFAULT_ASPECT_RATIOS,
    DEFAULT_CAMERA_MOTIONS,
    DEFAULT_FPS,
    DEFAULT_LIGHTING,
    OUTPUTS_DIR,
)
from modules.character import Character, CharacterManager
from modules.image_to_video import ImageToVideoGenerator
from modules.prompt_optimizer import build_negative_prompt, optimize_prompt
from modules.scene_generator import Scene, SceneGenerator
from modules.text_to_video import TextToVideoGenerator
from modules.utils import get_history, gpu_summary_string

# --------------------------------------------------------------------------- #
# Lazily-instantiated singletons (models load on first use, not at boot)
# --------------------------------------------------------------------------- #

_t2v_generator: TextToVideoGenerator | None = None
_i2v_generator: ImageToVideoGenerator | None = None
_scene_generator: SceneGenerator | None = None
character_manager = CharacterManager()


def get_t2v() -> TextToVideoGenerator:
    global _t2v_generator
    if _t2v_generator is None:
        _t2v_generator = TextToVideoGenerator()
    return _t2v_generator


def get_i2v() -> ImageToVideoGenerator:
    global _i2v_generator
    if _i2v_generator is None:
        _i2v_generator = ImageToVideoGenerator()
    return _i2v_generator


def get_scene_generator() -> SceneGenerator:
    global _scene_generator
    if _scene_generator is None:
        _scene_generator = SceneGenerator(generator=get_t2v())
    return _scene_generator


def aspect_ratio_to_size(aspect_ratio: str, base: int = 512) -> tuple[int, int]:
    ratios = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3)}
    rw, rh = ratios.get(aspect_ratio, (1, 1))
    if rw >= rh:
        w = base
        h = int(base * rh / rw)
    else:
        h = base
        w = int(base * rw / rh)
    # round to multiple of 8 for VAE compatibility
    w, h = (max(64, (w // 8) * 8), max(64, (h // 8) * 8))
    return w, h


def safe_call(fn, *args, **kwargs):
    """Wrap generation calls so exceptions surface as readable Gradio errors
    instead of an opaque traceback / silent failure.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {e}")


# --------------------------------------------------------------------------- #
# Tab 1: Text to Video
# --------------------------------------------------------------------------- #

def run_text_to_video(prompt, negative_prompt, seed, aspect_ratio, duration,
                       fps, guidance_scale, steps, use_optimizer,
                       progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    width, height = aspect_ratio_to_size(aspect_ratio)
    final_prompt = optimize_prompt(prompt) if use_optimizer else prompt
    final_negative = build_negative_prompt(negative_prompt)

    def cb(frac, msg):
        progress(frac, desc=msg)

    generator = get_t2v()
    output_path = safe_call(
        generator.generate,
        prompt=final_prompt,
        negative_prompt=final_negative,
        seed=int(seed),
        width=width,
        height=height,
        num_frames=max(9, int(duration * fps)),
        fps=int(fps),
        guidance_scale=guidance_scale,
        num_inference_steps=int(steps),
        progress_cb=cb,
    )
    return str(output_path), final_prompt


# --------------------------------------------------------------------------- #
# Tab 2: Image to Video
# --------------------------------------------------------------------------- #

def run_image_to_video(image, prompt, motion_strength, duration, fps,
                        progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload an image.")

    def cb(frac, msg):
        progress(frac, desc=msg)

    generator = get_i2v()
    output_path = safe_call(
        generator.generate,
        image=image,
        prompt=prompt,
        motion_strength=motion_strength,
        num_frames=max(9, int(duration * fps)),
        fps=int(fps),
        progress_cb=cb,
    )
    return str(output_path)


# --------------------------------------------------------------------------- #
# Tab 3: Character Consistency
# --------------------------------------------------------------------------- #

def save_character(name, image, description, face, clothes, hairstyle,
                    background, color_palette):
    if not name or not name.strip():
        raise gr.Error("Please provide a character name.")
    character = Character(
        name=name.strip(), description=description or "", face=face or "",
        clothes=clothes or "", hairstyle=hairstyle or "",
        background=background or "", color_palette=color_palette or "",
    )
    safe_call(character_manager.save, character, image)
    return (
        f"✅ Character '{name}' saved.",
        gr.update(choices=character_manager.list_characters()),
        gr.update(choices=character_manager.list_characters()),
    )


def load_character_preview(name):
    if not name:
        return None, ""
    character = character_manager.load(name)
    if not character:
        return None, "Character not found."
    return character.reference_image_path, character.to_prompt_fragment()


# --------------------------------------------------------------------------- #
# Tab 4: Scene Generator
# --------------------------------------------------------------------------- #

def run_scene_generation(scenes_state, project_name, add_subtitles,
                          burn_subtitles, progress=gr.Progress()):
    if not scenes_state:
        raise gr.Error("Add at least one scene first.")

    scenes = [
        Scene(
            prompt=s["prompt"], duration=s["duration"],
            camera_motion=s["camera_motion"], lighting=s["lighting"],
            character_name=s.get("character") or None,
            background=s.get("background") or None,
        )
        for s in scenes_state
    ]

    def cb(frac, msg):
        progress(frac, desc=msg)

    result = safe_call(
        get_scene_generator().generate_project,
        scenes=scenes,
        project_name=project_name or "project",
        add_subtitles=add_subtitles,
        burn_subtitles=burn_subtitles,
        progress_cb=cb,
    )

    scene_video_paths = [str(p) for p in result["scene_videos"]]
    return (
        str(result["final_video"]),
        scene_video_paths,
        str(result["thumbnail"]) if result["thumbnail"] else None,
        str(result["subtitle"]) if result["subtitle"] else None,
    )


def add_scene_to_list(scenes_state, prompt, duration, camera_motion,
                       lighting, character, background):
    if not prompt or not prompt.strip():
        raise gr.Error("Scene prompt cannot be empty.")
    scenes_state = list(scenes_state or [])
    scenes_state.append({
        "prompt": prompt, "duration": duration, "camera_motion": camera_motion,
        "lighting": lighting, "character": character, "background": background,
    })
    return scenes_state, render_scene_table(scenes_state), ""


def render_scene_table(scenes_state):
    if not scenes_state:
        return "_No scenes added yet._"
    rows = ["| # | Prompt | Duration | Camera | Lighting | Character |",
            "|---|--------|----------|--------|----------|-----------|"]
    for i, s in enumerate(scenes_state, start=1):
        rows.append(
            f"| {i} | {s['prompt'][:40]} | {s['duration']}s | "
            f"{s['camera_motion']} | {s['lighting']} | {s.get('character') or '-'} |"
        )
    return "\n".join(rows)


def clear_scenes():
    return [], render_scene_table([])


# --------------------------------------------------------------------------- #
# Gallery / history tab
# --------------------------------------------------------------------------- #

def refresh_gallery():
    videos = sorted(OUTPUTS_DIR.rglob("video.mp4"), key=lambda p: p.stat().st_mtime,
                     reverse=True)
    return [str(v) for v in videos[:30]]


def refresh_history():
    entries = get_history(limit=30)
    if not entries:
        return "_No history yet._"
    lines = []
    for e in entries:
        prompts = "; ".join(e.get("prompts", [e.get("prompt", "")]))[:120]
        lines.append(f"- **{e.get('timestamp', '')}** — {prompts}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# UI layout
# --------------------------------------------------------------------------- #

CUSTOM_CSS = """
.gradio-container {max-width: 1200px !important; margin: auto;}
#gpu_banner {padding: 8px 14px; border-radius: 8px; background: #1f2937; color: #f3f4f6; font-size: 0.9em;}
"""

with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🎬 {APP_TITLE}  \n*v{APP_VERSION} — open-source AI video generation studio*")
    gr.Markdown(gpu_summary_string(), elem_id="gpu_banner")

    with gr.Tabs():
        # ------------------------------------------------------------- #
        with gr.Tab("📝 Text to Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2v_prompt = gr.Textbox(label="Prompt", lines=3,
                                             placeholder="A woman dances in a sunlit room...")
                    t2v_negative = gr.Textbox(label="Negative Prompt", lines=2,
                                               placeholder="Optional extra negative terms")
                    t2v_optimize = gr.Checkbox(label="✨ Auto-optimize prompt (cinematic)", value=True)
                    with gr.Row():
                        t2v_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                        t2v_aspect = gr.Dropdown(DEFAULT_ASPECT_RATIOS, value="16:9", label="Aspect Ratio")
                    with gr.Row():
                        t2v_duration = gr.Slider(1, 10, value=4, step=0.5, label="Duration (s)")
                        t2v_fps = gr.Slider(8, 30, value=DEFAULT_FPS, step=1, label="FPS")
                    with gr.Row():
                        t2v_guidance = gr.Slider(1, 15, value=6.0, step=0.5, label="Guidance Scale")
                        t2v_steps = gr.Slider(10, 60, value=25, step=1, label="Steps")
                    t2v_btn = gr.Button("🎥 Generate", variant="primary")
                with gr.Column(scale=1):
                    t2v_video = gr.Video(label="Preview")
                    t2v_final_prompt = gr.Textbox(label="Final prompt used", interactive=False)

            t2v_btn.click(
                run_text_to_video,
                inputs=[t2v_prompt, t2v_negative, t2v_seed, t2v_aspect, t2v_duration,
                        t2v_fps, t2v_guidance, t2v_steps, t2v_optimize],
                outputs=[t2v_video, t2v_final_prompt],
            )

        # ------------------------------------------------------------- #
        with gr.Tab("🖼️ Image to Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2v_image = gr.Image(label="Upload Image", type="pil")
                    i2v_prompt = gr.Textbox(label="Prompt (motion description)", lines=2)
                    i2v_motion = gr.Slider(0, 1, value=0.6, step=0.05, label="Motion Strength")
                    with gr.Row():
                        i2v_duration = gr.Slider(1, 10, value=4, step=0.5, label="Duration (s)")
                        i2v_fps = gr.Slider(8, 30, value=DEFAULT_FPS, step=1, label="FPS")
                    i2v_btn = gr.Button("🎥 Generate", variant="primary")
                with gr.Column(scale=1):
                    i2v_video = gr.Video(label="Preview")

            i2v_btn.click(
                run_image_to_video,
                inputs=[i2v_image, i2v_prompt, i2v_motion, i2v_duration, i2v_fps],
                outputs=[i2v_video],
            )

        # ------------------------------------------------------------- #
        with gr.Tab("🧑‍🎤 Character Consistency"):
            with gr.Row():
                with gr.Column(scale=1):
                    char_image = gr.Image(label="Character Reference Image", type="pil")
                    char_name = gr.Textbox(label="Character Name")
                    char_description = gr.Textbox(label="General description", lines=2)
                    char_face = gr.Textbox(label="Face details")
                    char_clothes = gr.Textbox(label="Clothes")
                    char_hair = gr.Textbox(label="Hairstyle")
                    char_bg = gr.Textbox(label="Default background")
                    char_palette = gr.Textbox(label="Color palette")
                    char_save_btn = gr.Button("💾 Save Character", variant="primary")
                    char_status = gr.Markdown()
                with gr.Column(scale=1):
                    char_reuse_dropdown = gr.Dropdown(
                        character_manager.list_characters(), label="Reuse Character")
                    char_preview_img = gr.Image(label="Reference preview", interactive=False)
                    char_preview_prompt = gr.Textbox(label="Consistency prompt fragment",
                                                      interactive=False, lines=3)

            char_save_btn.click(
                save_character,
                inputs=[char_name, char_image, char_description, char_face,
                        char_clothes, char_hair, char_bg, char_palette],
                outputs=[char_status, char_reuse_dropdown],
            )
            char_reuse_dropdown.change(
                load_character_preview,
                inputs=[char_reuse_dropdown],
                outputs=[char_preview_img, char_preview_prompt],
            )

        # ------------------------------------------------------------- #
        with gr.Tab("🎬 Scene Generator"):
            scenes_state = gr.State([])
            gr.Markdown("Add multiple scenes, then generate them all — they are "
                        "automatically merged into one final video.")
            with gr.Row():
                with gr.Column(scale=1):
                    sc_prompt = gr.Textbox(label="Scene Prompt", lines=2)
                    sc_duration = gr.Slider(1, 10, value=4, step=0.5, label="Duration (s)")
                    sc_camera = gr.Dropdown(DEFAULT_CAMERA_MOTIONS, value="static", label="Camera Motion")
                    sc_lighting = gr.Dropdown(DEFAULT_LIGHTING, value=DEFAULT_LIGHTING[0], label="Lighting")
                    sc_character = gr.Dropdown(character_manager.list_characters(),
                                                label="Character (optional)", allow_custom_value=True)
                    sc_background = gr.Textbox(label="Background (optional)")
                    with gr.Row():
                        sc_add_btn = gr.Button("➕ Add Scene")
                        sc_clear_btn = gr.Button("🗑️ Clear Scenes")
                with gr.Column(scale=1):
                    sc_table = gr.Markdown("_No scenes added yet._")
                    sc_project_name = gr.Textbox(label="Project Name", value="my_video")
                    sc_add_subs = gr.Checkbox(label="Generate subtitle.srt", value=True)
                    sc_burn_subs = gr.Checkbox(label="Burn subtitles into video", value=False)
                    sc_generate_btn = gr.Button("🚀 Generate & Merge All Scenes", variant="primary")

            with gr.Row():
                sc_final_video = gr.Video(label="Final merged video")
                sc_thumbnail = gr.Image(label="Thumbnail", interactive=False)
            sc_scene_gallery = gr.Gallery(label="Individual scene clips", columns=4)
            sc_subtitle_file = gr.File(label="subtitle.srt")

            sc_add_btn.click(
                add_scene_to_list,
                inputs=[scenes_state, sc_prompt, sc_duration, sc_camera, sc_lighting,
                        sc_character, sc_background],
                outputs=[scenes_state, sc_table, sc_prompt],
            )
            sc_clear_btn.click(clear_scenes, outputs=[scenes_state, sc_table])
            sc_generate_btn.click(
                run_scene_generation,
                inputs=[scenes_state, sc_project_name, sc_add_subs, sc_burn_subs],
                outputs=[sc_final_video, sc_scene_gallery, sc_thumbnail, sc_subtitle_file],
            )

        # ------------------------------------------------------------- #
        with gr.Tab("🗂️ Gallery & History"):
            with gr.Row():
                gallery_refresh_btn = gr.Button("🔄 Refresh Gallery")
                history_refresh_btn = gr.Button("🔄 Refresh History")
            gallery = gr.Gallery(label="Generated videos", columns=4)
            history_md = gr.Markdown("_No history yet._")

            gallery_refresh_btn.click(refresh_gallery, outputs=[gallery])
            history_refresh_btn.click(refresh_history, outputs=[history_md])

    demo.load(refresh_gallery, outputs=[gallery])
    demo.load(refresh_history, outputs=[history_md])


if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True,
    )
