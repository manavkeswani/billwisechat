import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
import pandas as pd

# Configure Google GenerativeAI
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize GenerativeModel
model = genai.GenerativeModel('gemini-pro-vision')

# Initialize Streamlit app
st.set_page_config(page_title="Invoice Extractor")
st.header("Invoice Extractor")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Function to get Gemini response
def get_gemini_response(input, image, prompt):
    response = model.generate_content([input, image, prompt])
    return response.text

# Function to setup image input
def input_image_setup(uploaded_files):
    images = []
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.getvalue()
            image = {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
            images.append(image)
    return images

# Input prompt
input_prompt = st.text_input("Input Prompt: ", key="input")
uploaded_files = st.file_uploader("Choose images of the invoice...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Display uploaded images
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me about the invoice")

input_prompt_template = """
You are an expert in understanding invoices.
You will receive input images as invoices and
you will have to answer questions based on the input image.
"""

# If ask button is clicked
if submit:
    image_data = input_image_setup(uploaded_files[0])  # Take the first image only for now
    if image_data:
        response = get_gemini_response(input_prompt_template, image_data, input_prompt)
        st.subheader("The Response is")
        st.write(response)
        # Append the conversation to the history
        st.session_state.conversation_history.append({"prompt": input_prompt, "response": response})
    else:
        st.warning("Please upload an image.")

# Function to extract data from image and store it in CSV
def extract_data_from_image(image, system_prompt, user_prompt):
    extracted_data = []
    output = get_gemini_response(system_prompt, image, user_prompt)
    extracted_data.append(output)
    return extracted_data

# If extract data button is clicked
extract_data_button = st.button("Extract Data")
if extract_data_button and uploaded_files:
    system_prompt = """
               You are a specialist in comprehending receipts.
               Input images in the form of receipts will be provided to you,
               and your task is to respond to questions based on the content of the input image.
               """
    user_prompt = "Convert Invoice data into json format with appropriate json tags as required for the data in image "
    image = input_image_setup(uploaded_files[0])  # Take the first image only for now
    extracted_data = extract_data_from_image(image, system_prompt, user_prompt)
    
    # Convert extracted data to CSV and offer download
    df = pd.DataFrame({"Extracted Data": extracted_data})
    csv = df.to_csv(index=False)
    st.download_button(label="Download Extracted Data", data=csv, file_name="extracted_data.csv", mime="text/csv")

# Display conversation history
st.subheader("Conversation History")
for i, conversation in enumerate(st.session_state.conversation_history):
    st.write(f"Prompt {i+1}: {conversation['prompt']}")
    st.write(f"Response {i+1}: {conversation['response']}")
