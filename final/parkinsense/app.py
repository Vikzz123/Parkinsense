import os
import numpy as np
import pickle
import requests
from flask import Flask, request, redirect, render_template, url_for,session
from werkzeug.utils import secure_filename
from process_image import process_image
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)
model = pickle.load(open('model_knn.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))
app.secret_key = os.urandom(24)

app.config["IMAGE_UPLOADS"] = "./static/img"


@app.route("/", methods=["GET", "POST"])
def upload_image():
   if request.method == "POST":
      if request.files:
         image = request.files["image"]

         image_url = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)

         image.save(image_url)

         print("Image saved")

         result = process_image(image_url)
         serializable_result = result.tolist()  # Convert NumPy array to list, or custom object to dictionary
         session['uploaded_image_result'] = serializable_result
         image_path = "static/img/" + image.filename
         return render_template("result.html", result=result, image_path=image_path)
   
   return render_template("index.html")

@app.route("/about", methods=["GET"])
def about():
   return render_template("about.html")


@app.route('/predict',methods=['POST'])
def predict():
    uploaded_image_result = session['uploaded_image_result']
    print(uploaded_image_result)
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)  
    output = prediction  
   #  if output == [0]:
   #      output = "Parkinsons Disease Not Detected"
   #  elif output == [1]:
   #      output = "Parkinsons Disease Detected"
        
    if uploaded_image_result == [1] and output == [1]:
        pred = "You have Parkinson's Disease. Please consult a specialist."
        color = "red"
    elif uploaded_image_result == [1]:
        pred = "Your spiral drawing indicates you may have parkinsons."
        color = "orange"
    elif uploaded_image_result == [0] and output == [1]:
      pred = "Your voice data indicates you may have parkinsons."
      color = "orange"
    else:
      pred = "You are Healthy"
      color = "green"


    return render_template ("final_result.html", prediction =
              pred , color = color )
    






DOWNLOAD_DIRECTORY = "./static/img"
@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():

   resp = MessagingResponse()
   
   if request.values['NumMedia'] != '0':
      
      # Use the message SID as a filename.
      filename = request.values['MessageSid'] + '.png'
      with open('{}/{}'.format(DOWNLOAD_DIRECTORY, filename), 'wb') as f:
         image_url = request.values['MediaUrl0']
         f.write(requests.get(image_url).content)

         result = process_image('{}/{}'.format(DOWNLOAD_DIRECTORY, filename))

         print(result)
         
      if (result[0] == 0):
         resp.message("Image successfully processed. The prediction came back negative. It is unlikely that you have Parkinson's. Please note that this is not a diagnosis. If you have any questions or concerns, please consult a medical professional.")
      else:
         resp.message("Image successfully processed. The prediction came back positive. The drawing was similar to other patients that have Parkinson's disease. Please note that this is not a diagnosis. If you have any questions or concerns, please consult a medical professional.")
   else:
      resp.message("Try sending a picture message.")

   return str(resp)


if __name__ == '__main__':
    app.run(debug=True)