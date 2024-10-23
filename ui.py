# import streamlit as st
# from ultralytics import YOLO
# import os
# import tempfile

# # Load the YOLO model
# @st.cache_resource
# def load_model():
#     return YOLO('./best.pt')

# model = load_model()

# st.title('YOLO Thermal Object Detection')

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","mp4"])

# if uploaded_file is not None:
#     # Determine if the file is an image or video
#     file_type = uploaded_file.type.split('/')[0]
    
#     if file_type == "image":
#         # Display the uploaded image
#         st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
#         # Save the uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             temp_file_path = tmp_file.name
        
#         if st.button('Detect Objects'):
#             if model is None:
#                 st.warning("Please load the model first by clicking 'Load Model'")
#             else:
#                 # Perform inference
#                 results = model.predict(source=temp_file_path, save=True)
                
#                 # Find the latest 'exp' folder
#                 detect_folder = './runs/detect/'
#                 exp_folders = [f for f in os.listdir(detect_folder) if f.startswith('predict')]
#                 latest_exp = max(exp_folders, key=lambda f: os.path.getctime(os.path.join(detect_folder, f)))
                
#                 # Get the path of the output image
#                 output_image_path = os.path.join(detect_folder, latest_exp, os.path.basename(temp_file_path))
                
#                 # Display the output image
#                 if os.path.exists(output_image_path):
#                     st.image(output_image_path, caption='Detected Objects', use_column_width=True)
#                 else:
#                     st.error("Output image not found. There might be an issue with the detection process.")
        
#         # Clean up the temporary file
#         os.remove(temp_file_path)
    
#     elif file_type == "video":
#         # Save the uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.avi') as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             temp_file_path = tmp_file.name
        
#         if st.button('Process Video'):
#             if model is None:
#                 st.warning("Please load the model first by clicking 'Load Model'")
#             else:
#                 # Perform inference on video
#                 results = model.predict(source=temp_file_path, save=True)
                
#                 # Find the latest 'exp' folder
#                 detect_folder = './runs/detect/'
#                 exp_folders = [f for f in os.listdir(detect_folder) if f.startswith('predict')]
#                 latest_exp = max(exp_folders, key=lambda f: os.path.getctime(os.path.join(detect_folder, f)))
                
#                 # Get the path of the output video
#                 output_video_path = os.path.join(detect_folder, latest_exp, os.path.basename(temp_file_path))
                
#                 # Display the output video
#                 if os.path.exists(output_video_path):
#                      with open(output_video_path, 'rb') as f:
#                         st.download_button(
#                             label="Download Processed Video",
#                             data=f,
#                             file_name=os.path.basename(output_video_path),
#                             mime="video/avi"
#                         )
#                 else:
#                     st.error("Output video not found. There might be an issue with the detection process.")
        
#         # Clean up the temporary file
#         os.remove(temp_file_path)

# st.info('Upload an image and click "Detect Objects" to see the results.')

# new code -->

# import streamlit as st
# from ultralytics import YOLO
# import os
# import tempfile
# import glob
# from PIL import Image
# import cv2
# import numpy as np

# # Function to get available model files
# def get_available_models():
#     # Look for .pt files in the current directory
#     model_files = glob.glob('./*.pt')
#     # Extract just the filenames
#     model_names = [os.path.basename(f) for f in model_files]
#     return model_names

# # Load the YOLO model
# @st.cache_resource
# def load_model(model_path):
#     return YOLO(model_path)

# # Function to process image without saving
# def process_image(model, image):
#     # Convert PIL Image to numpy array
#     if isinstance(image, Image.Image):
#         image_array = np.array(image)
#     else:
#         image_array = image
        
#     # Run detection
#     results = model.predict(source=image_array, save=False)
    
#     # Get the plotted image directly from results
#     plotted_image = results[0].plot()
    
#     # Convert BGR to RGB
#     if plotted_image.shape[-1] == 3:  # If image is colored
#         plotted_image = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
        
#     return plotted_image

# st.title('YOLO Thermal Object Detection')

# # Model selection
# available_models = get_available_models()
# if not available_models:
#     st.error("No model files (.pt) found in the current directory!")
# else:
#     selected_model = st.selectbox(
#         'Select Model',
#         available_models,
#         index=available_models.index('best.pt') if 'best.pt' in available_models else 0
#     )
    
#     model = load_model(f'./{selected_model}')

#     # Display selected model information
#     with st.expander("Model Information"):
#         st.write(f"Selected model: {selected_model}")
#         st.write("Task: Object Detection")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "mp4"])

# if uploaded_file is not None:
#     # Determine if the file is an image or video
#     file_type = uploaded_file.type.split('/')[0]
    
#     if file_type == "image":
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
        
#         if st.button('Detect Objects'):
#             if model is None:
#                 st.warning("Please ensure a model is selected")
#             else:
#                 with st.spinner('Detecting objects...'):
#                     # Process the image without saving
#                     processed_image = process_image(model, image)
                    
#                     # Display the processed image
#                     st.success('Detection completed!')
#                     st.image(processed_image, caption='Detected Objects', use_column_width=True)
                    
    
#     elif file_type == "video":
#         # Save the uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.avi') as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             temp_file_path = tmp_file.name
        
#         if st.button('Process Video'):
#             if model is None:
#                 st.warning("Please ensure a model is selected")
#             else:
#                 with st.spinner('Processing video...'):
#                     # Create a temporary output path for the video
#                     output_path = tempfile.mktemp(suffix='.avi')
                    
#                     # Process video with YOLO
#                     results = model.predict(source=temp_file_path, save=True, project=os.path.dirname(output_path), name=os.path.basename(output_path))
                    
#                     # Check if video was processed successfully
#                     if os.path.exists(output_path):
#                         st.success('Video processing completed!')
#                         # Read the processed video for download
#                         with open(output_path, 'rb') as f:
#                             st.download_button(
#                                 label="Download Processed Video",
#                                 data=f,
#                                 file_name=f"detected_{uploaded_file.name}",
#                                 mime="video/avi"
#                             )
#                         # Clean up the output video
#                         os.remove(output_path)
#                     else:
#                         st.error("Error processing video.")
                        
#                 # Clean up the temporary input file
#                 os.remove(temp_file_path)

# st.info('Upload an image or video and click the appropriate button to see the results.')

import streamlit as st
from ultralytics import YOLO
import os
import tempfile
import glob

# Function to get available model files
def get_available_models():
    # Look for .pt files in the current directory
    model_files = glob.glob('./*.pt')
    # Extract just the filenames
    model_names = [os.path.basename(f) for f in model_files]
    return model_names

# Load the YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

st.title('YOLO Thermal Object Detection')

# Model selection
available_models = get_available_models()
if not available_models:
    st.error("No model files (.pt) found in the current directory!")
else:
    selected_model = st.selectbox(
        'Select Model',
        available_models,
        index=available_models.index('best.pt') if 'best.pt' in available_models else 0
    )
    
    model = load_model(f'./{selected_model}')

    # Display selected model information
    with st.expander("Model Information"):
        st.write(f"Selected model: {selected_model}")
        st.write("Task: Object Detection")
        # You can add more model information here if needed

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # Determine if the file is an image or video
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == "image":
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        if st.button('Detect Objects'):
            if model is None:
                st.warning("Please ensure a model is selected")
            else:
                with st.spinner('Detecting objects...'):
                    # Perform inference
                    results = model.predict(source=temp_file_path, save=True)
                    
                    # Find the latest 'exp' folder
                    detect_folder = './runs/detect/'
                    exp_folders = [f for f in os.listdir(detect_folder) if f.startswith('predict')]
                    latest_exp = max(exp_folders, key=lambda f: os.path.getctime(os.path.join(detect_folder, f)))
                    
                    # Get the path of the output image
                    output_image_path = os.path.join(detect_folder, latest_exp, os.path.basename(temp_file_path))
                    
                    # Display the output image
                    if os.path.exists(output_image_path):
                        st.success('Detection completed!')
                        st.image(output_image_path, caption='Detected Objects', use_column_width=True)
                    else:
                        st.error("Output image not found. There might be an issue with the detection process.")
        
        # Clean up the temporary file
        os.remove(temp_file_path)
    
    elif file_type == "video":
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.avi') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        if st.button('Process Video'):
            if model is None:
                st.warning("Please ensure a model is selected")
            else:
                with st.spinner('Processing video...'):
                    # Perform inference on video
                    results = model.predict(source=temp_file_path, save=True)
                    
                    # Find the latest 'exp' folder
                    detect_folder = './runs/detect/'
                    exp_folders = [f for f in os.listdir(detect_folder) if f.startswith('predict')]
                    latest_exp = max(exp_folders, key=lambda f: os.path.getctime(os.path.join(detect_folder, f)))
                    
                    # Get the path of the output video
                    output_video_path = os.path.join(detect_folder, latest_exp, os.path.basename(temp_file_path))
                    
                    # Display the output video
                    if os.path.exists(output_video_path):
                        st.success('Video processing completed!')
                        with open(output_video_path, 'rb') as f:
                            st.download_button(
                                label="Download Processed Video",
                                data=f,
                                file_name=os.path.basename(output_video_path),
                                mime="video/avi"
                            )
                    else:
                        st.error("Output video not found. There might be an issue with the processing.")
        
        # Clean up the temporary file
        os.remove(temp_file_path)

st.info('Upload an image or video and click the appropriate button to see the results.')