import streamlit as st
import cv2
import numpy as np
import zipfile
import io

st.title("My first project - Image processing App")

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg","png","jpeg"],
    accept_multiple_files=True
)

if uploaded_files:

    option = st.selectbox("Choose Operation", ["Resize","Grayscale","Blur","Edge Detection"])

    # Inputs OUTSIDE loop (important)
    if option == "Resize":
        scale = st.slider("Resize (%)", 10, 200, 100)

    elif option == "Grayscale":
        intensity = st.slider("Gray Intensity", 0.5, 2.0, 1.0)

    elif option == "Blur":
        k = st.slider("Blur Level", 1, 25, 5, step=2)

    elif option == "Edge Detection":
        mode = st.selectbox("Edge Sensitivity", ["Low","Medium","High"])

    results = []
    progress = st.progress(0)

    for i, file in enumerate(uploaded_files):

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Apply operation
        if option == "Resize":
            width = int(img.shape[1] * scale / 100)
            height = int(img.shape[0] * scale / 100)
            output = cv2.resize(img, (width, height))

        elif option == "Grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            output = cv2.convertScaleAbs(gray, alpha=intensity, beta=0)

        elif option == "Blur":
            output = cv2.GaussianBlur(img, (k,k), 0)

        elif option == "Edge Detection":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)

            if mode == "Low":
                low, high = 100, 200
            elif mode == "Medium":
                low, high = 50, 150
            else:
                low, high = 20, 80

            output = cv2.Canny(blur, low, high)

        # Fix color for display
        if len(output.shape) == 3:
            display_img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        else:
            display_img = output

        # Show only first 3 images (avoid lag)
        if i < 3:
            st.image(display_img, caption=f"Preview {file.name}")

        # Encode and store
        _, buffer = cv2.imencode(".png", output)
        results.append((file.name, buffer))

        progress.progress((i + 1) / len(uploaded_files))

    # Create ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for name, buffer in results:
            zf.writestr(name, buffer.tobytes())

    st.download_button(
        "Download All Images",
        zip_buffer.getvalue(),
        file_name="processed_images.zip",
        mime="application/zip"
    )