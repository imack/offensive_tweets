# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import flask
from flask import json
from scoring_service import ScoringService

predictions = ScoringService.predict(["this is just a warmup inference, warmup the inference engine"])
print(predictions[0])
app = flask.Flask(__name__)

@app.route('/invocations', methods=['POST'])
def invocation():

    params = flask.request.json
    data = params.get("instances")

    predictions = ScoringService.predict(data)

    return app.response_class(
        response = json.dumps({'predictions': predictions.tolist()}),
        status = 200,
        mimetype = 'application/json'
    )

    return flask.Response(response=result, status=200, mimetype='text/json')

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))

    # start the flask app, allow remote connections
    app.run(host='0.0.0.0')