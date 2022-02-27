from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS




app = Flask(__name__)
CORS(app)


classifier = joblib.load('./model/pipeline.pkl')
# .\venv\Scripts\activate

# make predict route a post route
# add route to fetch newsmedia ranking
# add route to get related news articles

@app.route("/<name>")
def home(name):
     return f'{name}'


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.get_json()
    print("here")  

    pred= classifier.predict([[json_['industry'],json_['skill']]])  

    print(pred[0])

    return jsonify({'pred': int(pred[0])})

           
        

    

if __name__ == "__main__":
    classifier = joblib.load('./model/pipeline.pkl')
    app.run(debug=True)

