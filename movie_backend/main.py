from flask import Flask, json, jsonify, request
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

regressor = pickle.load(open('regressor.pkl', 'rb'))
clf = pickle.load(open('DT.pkl', 'rb'))
rfc = pickle.load(open('RFC.pkl', 'rb'))
rfc2 = pickle.load(open('RFC2.pkl', 'rb'))


@app.route("/evaluate", methods=['POST'])
def evaluate():
    request_data = json.loads(request.data)
    print(request_data)
    runtime = request_data["runtime"]
    rating = request_data["rating"]
    votes = request_data["votes"]
    revenue = request_data["revenue"]
    metascore= request_data["metascore"]
    
    
    #######################################
    ##########CODE HERE####################
    genres=['Action', 'Adventure', 'Horror', 'Animation', 'Comedy',
       'Biography', 'Drama', 'Crime', 'Mystery', 'Sci-Fi', 'Fantasy',
       'Romance', 'Thriller']
    print("The predicted movie could be made in:",genres[int(np.round_((regressor.predict([[runtime,rating,votes,revenue,metascore]]))))],
      "\nThe best director for the movie is predicted as:",clf.predict([[runtime,rating,votes,revenue,metascore,int(np.round_((regressor.predict([[runtime,rating,votes,revenue,metascore]]))))]])[0],
      "\nThe actor preferred for the film would be:",rfc.predict([[runtime,rating,votes,revenue,metascore,int(np.round_((regressor.predict([[runtime,rating,votes,revenue,metascore]]))))]])[0],
      "\nThe actor preferred to work along with",rfc.predict([[runtime,rating,votes,revenue,metascore,int(np.round_((regressor.predict([[runtime,rating,votes,revenue,metascore]]))))]])[0],"for the film would be:",rfc2.predict([[runtime,rating,votes,revenue,metascore,int(np.round_((regressor.predict([[runtime,rating,votes,revenue,metascore]]))))]])[0])
    output = {"predictedgen": genres[int(np.round_((regressor.predict([[runtime,rating,votes,revenue,metascore]]))))],
                "predicteddir":clf.predict([[runtime,rating,votes,revenue,metascore,int(np.round_((regressor.predict([[runtime,rating,votes,revenue,metascore]]))))]])[0],
                "predictedac1":rfc.predict([[runtime,rating,votes,revenue,metascore,int(np.round_((regressor.predict([[runtime,rating,votes,revenue,metascore]]))))]])[0],
                "predictedac2":rfc2.predict([[runtime,rating,votes,revenue,metascore,int(np.round_((regressor.predict([[runtime,rating,votes,revenue,metascore]]))))]])[0]}
    return output


if __name__ == "__main__":
    app.run(debug=True)
