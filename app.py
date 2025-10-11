#import all the required libraries
import requests
import re
import base64
import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "supersecretkey"

# IBM Watsonx setup
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    # api_key="<YOUR_API_KEY>"
)
client = APIClient(credentials)
model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
project_id = "skills-network"
params = TextChatParameters()

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)

#converting the image to base64 strings to send it through text llms
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        encoded_image = base64.b64encode(bytes_data).decode("utf-8")
        return encoded_image
    else:
        raise FileNotFoundError("No file uploaded")

#using regex to format the LLMs response and make it more user friendly
def format_response(response_text):
    response_text = re.sub(r"\*\*(.*?)\*\*", r"<p><strong>\1</strong></p>", response_text)
    response_text = re.sub(r"(?m)^\s*\*\s(.*)", r"<li>\1</li>", response_text)
    response_text = re.sub(r"(<li>.*?</li>)+", lambda match: f"<ul>{match.group(0)}</ul>", response_text, flags=re.DOTALL)
    response_text = re.sub(r"</p>(?=<p>)", r"</p><br>", response_text)
    response_text = re.sub(r"(\n|\\n)+", r"<br>", response_text)
    return response_text

#defining the payload in the models format to let it know the kinds of input it will receive
def generate_model_response(encoded_image, user_query, assistant_prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": assistant_prompt + "\n\n" + user_query},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
            ]
        }
    ]
    #feeding the inputs to the model and receiving a response
    try:
        response = model.chat(messages=messages)
        raw_response = response['choices'][0]['message']['content']
        formatted_response = format_response(raw_response)
        return formatted_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "<p>An error occurred while generating the response.</p>"

#basic flask app to test it out
@app.route("/", methods=["GET", "POST"])
def index():
    response_ask = None
    response_cal = None
    uploaded_image = None
    user_query = ""

    # to upload the image
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        action = request.form.get("action")
        user_query = request.form.get("user_query", "")
        # image upload error handling
        if not uploaded_file:
            flash("Please upload an image file.", "danger")
            return redirect(url_for("index"))
        

        encoded_image = input_image_setup(uploaded_file)
        uploaded_image = encoded_image 
        #since we have two seperate buttons, we ensure that the right inputs and prompt templates
        #are used for both
        if action == "ask_anything":
            assistant_prompt = "You are an expert nutritionist. Answer the user's question about the food image."
            response_ask = generate_model_response(encoded_image, user_query, assistant_prompt)

        elif action == "total_calories":
            assistant_prompt = """
            You are an expert nutritionist. Your task is to analyze the food items in the image and provide a detailed nutritional assessment:

            1. Identification: List each identified food item.
            2. Portion Size & Calorie Estimation: Use bullet points.
            3. Total Calories
            4. Nutrient Breakdown: Protein, Carbs, Fats, Vitamins, Minerals.
            5. Health Evaluation
            6. Disclaimer: The nutritional information is approximate.
            """
            response_cal = generate_model_response(encoded_image, "", assistant_prompt)

    # render the html page
    return render_template(
        "index.html",
        user_query=user_query,
        uploaded_image=uploaded_image,
        response_ask=response_ask,
        response_cal=response_cal
    )

if __name__ == "__main__":
    app.run(debug=True)
