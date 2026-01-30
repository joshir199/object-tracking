import torch
import os
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import math
import cv2     # assuming you still need it downstream
from sam2.build_sam import build_sam2_video_predictor


# global predictor initialisation
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


# single object tracking (SOT)
def run_tracking(video_frames_path, selected_points_np):
    # SAM2 / tracking code here
    #initialisation of sam predictor
    inference_state = predictor.init_state(video_path=video_frames_path)
    predictor.reset_state(inference_state)

    # set selected object for first frame
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # for labels, `1` means positive click and `0` means negative click
    labels = np.ones(len(selected_points_np), dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=selected_points_np,
        labels=labels,
    )

    output_video_path = "output_segmentation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    # Get dimensions from the first frame image
    sample_img = cv2.imread(os.path.join(video_frames_path, "00000.jpg"))
    height, width, _ = sample_img.shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Starting propagation and video writing...")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # Load original frame
        frame_path = os.path.join(video_frames_path, f"{out_frame_idx:05d}.jpg")
        frame = cv2.imread(frame_path)
        overlay = frame.copy()

        for i, out_obj_id in enumerate(out_obj_ids):
            # Convert logits to binary mask and squeeze to ensure 2D shape
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
            color = [250, 200, 0]
            # Apply color to the mask area on the overlay
            overlay[mask > 0] = color

            alpha = 0.4
            combined_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            video_writer.write(combined_frame)

    # Finalize the video file
    video_writer.release()

    return output_video_path



# ─── Star drawing helper ────────────────────────────────────────

def draw_star(draw, cx, cy, outer_r=14, inner_r=5,
              fill=(255, 255, 0, 180), outline=(200, 0, 0), width=2):
    """Draw a 5-pointed star (semi-transparent yellow + red border)"""
    pts = []
    for i in range(10):
        angle = math.pi * 2 * i / 10 - math.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        pts.append((x, y))
    draw.polygon(pts, fill=fill, outline=outline, width=width)


# ─── Core interaction functions ─────────────────────────────────

def on_video_upload(video_path):
    """Extract all frames from video, save them to disk folder, show first frame"""
    if video_path is None:
        return None, None, [], "No video uploaded."

    # Create persistent folder for frames (relative to your script)
    frames_dir = "./video_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # We'll clear previous frames (optional – comment out if you want to keep history)
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, [], "Cannot open video file."

    frames_pil = []           # list of PIL images (for state / display if needed)
    frame_idx = 0
    saved_paths = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # BGR → RGB → PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Save frame to disk as JPG
        frame_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
        pil_img.save(frame_path, "JPEG", quality=95)


        frames_pil.append(pil_img)
        saved_paths.append(frame_path)
        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        return None, None, [], "Video has no frames."

    first_frame_pil = frames_pil[0]

    status = (
        f"Video processed: {frame_idx} frames saved to '{frames_dir}'\n"
        f"First frame ready – click 2 points to select object."
    )

    # Return: original first PIL, display first PIL, reset points, status
    # You can also return frames_pil list if you want to keep them in state
    return first_frame_pil, first_frame_pil, [], status


def on_select(evt: gr.SelectData, original_pil: Image.Image, current_points):
    """Handle click → keep last 2 points → draw stars"""
    if original_pil is None:
        return None, current_points, "No image loaded."

    points = list(current_points)
    points.append(evt.index)           # (x, y) from click

    if len(points) > 2:
        points = points[-2:]

    # Draw on copy
    display = original_pil.copy()
    draw = ImageDraw.Draw(display, "RGBA")

    for x, y in points:
        draw_star(draw, x, y)

    status = f"Points: {points}"
    if len(points) == 2:
        status += " → ready to track"

    return display, points, status


def clear_points(original_pil):
    if original_pil is None:
        return None, [], "No image to clear."
    return original_pil, [], "Points cleared."


def on_run_tracking(video_path, points_list):
    if video_path is None:
        return None, "No video loaded."
    if len(points_list) < 1:
        return None, "Select at least one point."

    points_np = np.array(points_list, dtype=np.float32)

    try:
        frames_dir = "./video_frames"
        out_path = run_tracking(frames_dir, points_np)
        return out_path, f"Tracking finished → {out_path}"
    except Exception as e:
        return None, f"Tracking error: {str(e)}"



if __name__ == "__main__":

    # ─── Gradio UI ──────────────────────────────────────────────────

    with gr.Blocks(title="Object Tracking Made Easy") as demo:
        gr.Markdown("""
        # ✨ SAM2-based Object Tracking 
        1. Upload a video
        2. Click **two points** on the **first frame** to mark the object
        3. Press **Run Tracking**
        """)

        with gr.Row():
            video_input = gr.Video(
                label="Upload video (MP4)",
                sources=["upload"],
                format="mp4",
                height=360
            )

        status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            first_frame_display = gr.Image(
                label="First frame – click 2 points to select object",
                type="pil",
                interactive=True,
                height=520
            )

        with gr.Row():
            clear_btn = gr.Button("Clear points", variant="secondary")
            run_btn = gr.Button("Run Tracking", variant="primary")

        output_video = gr.Video(label="Tracking Result")

        # ─── States ─────────────────────────────────────────────────────

        original_first_pil = gr.State(None)  # clean first frame (PIL)
        clicked_points = gr.State([])  # list of (x,y) tuples
        all_frames_state = gr.State(None)  # optional – if you extract all frames

        # ─── Event bindings ─────────────────────────────────────────────

        # Upload video → show first frame + reset
        video_input.upload(
            fn=on_video_upload,
            inputs=video_input,
            outputs=[original_first_pil, first_frame_display, clicked_points, status_text]
        )

        # Click on image → update points & visualization
        first_frame_display.select(
            fn=on_select,
            inputs=[original_first_pil, clicked_points],
            outputs=[first_frame_display, clicked_points, status_text]
        )

        # Clear button
        clear_btn.click(
            fn=clear_points,
            inputs=original_first_pil,
            outputs=[first_frame_display, clicked_points, status_text]
        )

        # Run tracking
        run_btn.click(
            fn=on_run_tracking,
            inputs=[video_input, clicked_points],
            outputs=[output_video, status_text]
        )

    demo.launch(share=True)
