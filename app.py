import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define object categories for furniture
OBJECT_NAMES = {56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "table"}

# Function to detect furniture in an image
def detect_objects(image_path):
    results = model(image_path)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id in OBJECT_NAMES:
                detected_objects.append({
                    "class_id": cls_id,
                    "name": OBJECT_NAMES[cls_id],
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                })

    return detected_objects

# Function to compare detected objects and suggest adjustments
def generate_suggestions(room_objects, reference_objects):
    suggestions = []
    room_positions = {obj["name"]: obj["center"] for obj in room_objects}
    reference_positions = {obj["name"]: obj["center"] for obj in reference_objects}

    for obj_name, ref_center in reference_positions.items():
        if obj_name in room_positions:
            room_center = room_positions[obj_name]
            dx, dy = ref_center[0] - room_center[0], ref_center[1] - room_center[1]
            movement = []

            if abs(dx) > 30:
                movement.append("right" if dx > 0 else "left")
            if abs(dy) > 30:
                movement.append("down" if dy > 0 else "up")

            if movement:
                suggestions.append(f"➡️ {obj_name}: Move {' and '.join(movement)}")

    if not suggestions:
        suggestions.append("✅ Your furniture layout is already optimal!")

    return suggestions

# Function to annotate and visualize movement suggestions
def annotate_image(image_path, detected_objects, suggestions, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = obj["center"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), obj["name"], fill="red")

        for suggestion in suggestions:
            if obj["name"] in suggestion:
                if "right" in suggestion:
                    draw.line([(cx, cy), (cx + 50, cy)], fill="blue", width=3)
                if "left" in suggestion:
                    draw.line([(cx, cy), (cx - 50, cy)], fill="blue", width=3)
                if "up" in suggestion:
                    draw.line([(cx, cy), (cx, cy - 50)], fill="blue", width=3)
                if "down" in suggestion:
                    draw.line([(cx, cy), (cx, cy + 50)], fill="blue", width=3)

    image.save(output_path)
    return output_path

# Streamlit UI
st.title("🏡 AI-Powered Virtual Interior Designer")
st.write("Upload your *room image* and a *reference image* to get *furniture placement suggestions*.")

# Upload images
room_image_file = st.file_uploader("Upload your *room image*", type=["jpg", "png", "jpeg"])
reference_image_file = st.file_uploader("Upload a *reference image*", type=["jpg", "png", "jpeg"])

if room_image_file and reference_image_file:
    # Save uploaded images
    room_image_path = "room_image.jpg"
    reference_image_path = "reference_image.jpg"

    Image.open(room_image_file).save(room_image_path)
    Image.open(reference_image_file).save(reference_image_path)

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.image(room_image_path, caption="📸 Uploaded Room Image", use_container_width=True)
    with col2:
        st.image(reference_image_path, caption="🎨 Reference Image", use_container_width=True)

    # Detect objects
    room_objects = detect_objects(room_image_path)
    reference_objects = detect_objects(reference_image_path)

    # Generate suggestions
    suggestions = generate_suggestions(room_objects, reference_objects)

    # Annotate and display suggestions
    detected_image_path = "suggested_layout.jpg"
    annotated_image_path = annotate_image(room_image_path, room_objects, suggestions, detected_image_path)
    
    st.image(annotated_image_path, caption="📍 Detected Objects & Suggested Changes", use_container_width=True)
    
    # Show placement suggestions
    st.subheader("📋 Suggested Furniture Placements")
    for suggestion in suggestions:
        st.write(suggestion)

    # Provide download option for annotated image
    with open(annotated_image_path, "rb") as file:
        st.download_button("📥 Download Processed Image", file, file_name="suggested_layout.jpg")
