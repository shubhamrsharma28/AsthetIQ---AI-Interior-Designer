import os
os.environ["PYTHONOPTIMIZE"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(page_title="AsthetIQ", layout="wide")

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

model = load_model()


OBJECT_NAMES = {
    56: "chair",
    57: "couch",         
    58: "potted plant",
    59: "bed",
    60: "table",        
    61: "lamp",
    62: "carpet"
}

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

def generate_suggestions(room_objects, reference_objects):
    suggestions = []
    room_positions = {obj["name"]: obj["center"] for obj in room_objects}
    reference_positions = {obj["name"]: obj["center"] for obj in reference_objects}

    for obj_name, ref_center in reference_positions.items():
        if obj_name in room_positions:
            room_center = room_positions[obj_name]
            dx = ref_center[0] - room_center[0]
            dy = ref_center[1] - room_center[1]
            movement = []

            threshold = 30

            if abs(dx) > threshold:
                movement.append("right" if dx > 0 else "left")
            if abs(dy) > threshold:
                movement.append("down" if dy > 0 else "up")

            if movement:
                verb = {
                    "chair": "Adjust",
                    "couch": "Shift",
                    "sofa": "Shift",
                    "table": "Reposition",
                    "center table": "Reposition",
                    "potted plant": "Move",
                    "bed": "Reposition",
                    "lamp": "Adjust",
                    "carpet": "Reposition"
                }.get(obj_name, "Move")

                emoji = {
                    "chair": "🪑",
                    "couch": "🛋️",
                    "sofa": "🛋️",
                    "table": "🧺",
                    "center table": "🧺",
                    "potted plant": "🌼",
                    "bed": "🛏️",
                    "lamp": "💡",
                    "carpet": "🧶"
                }.get(obj_name, "➡️")

                suggestion = f"{emoji} {obj_name}: {verb} {' and '.join(movement)} for better placement"
                suggestions.append(suggestion)

    if not suggestions:
        suggestions.append("✅ Your furniture layout is already optimal!")

    return suggestions

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

# from here, the main UI code of Streamlit starts :
st.title("🏡 AsthetIQ – AI-Powered Virtual Interior Designer")
st.write("Upload your *room image* and a *reference image* to get *furniture placement suggestions*.")

room_image_file = st.file_uploader("Upload your *room image*", type=["jpg", "png", "jpeg"])
reference_image_file = st.file_uploader("Upload a *reference image*", type=["jpg", "png", "jpeg"])

if room_image_file and reference_image_file:
    with st.spinner("Analyzing layout..."):
        room_image_path = "room_image.jpg"
        reference_image_path = "reference_image.jpg"

        Image.open(room_image_file).convert("RGB").save(room_image_path, format="JPEG")
        Image.open(reference_image_file).convert("RGB").save(reference_image_path, format="JPEG")

        col1, col2 = st.columns(2)
        with col1:
            st.image(room_image_path, caption="📸 Uploaded Room Image", use_column_width=True)
        with col2:
            st.image(reference_image_path, caption="🎨 Reference Image", use_column_width=True)

        room_objects = detect_objects(room_image_path)
        reference_objects = detect_objects(reference_image_path)
        suggestions = generate_suggestions(room_objects, reference_objects)

        detected_image_path = "suggested_layout.jpg"
        annotated_image_path = annotate_image(room_image_path, room_objects, suggestions, detected_image_path)

        st.image(annotated_image_path, caption="📍 Detected Objects & Suggested Changes", use_column_width=True)

        st.subheader("📋 Suggested Furniture Placements")
        for suggestion in suggestions:
            st.write(suggestion)

        with open(annotated_image_path, "rb") as file:
            st.download_button("📥 Download Processed Image", file, file_name="suggested_layout.jpg")

else:
    st.info("Please upload both images to begin.")


#Triggering StartUp
