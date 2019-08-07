## Offensive Tweet Finder

The purpose of this project is to make a Machine Learning service that will take tweets as input and give back score of 
0.0 to 1.0 based on how offensive our algorithm finds the tweet to be.

Using the python notebook to build our model in [Keras](https://keras.io/), we serve the model inferences using a Flask app.


## Requirements

### Packages
* Python 3
* Keras
* Tensorflow
* Flask 


### Data (put in /data dir)
* [Hate Speech Identification from Crowdflower](https://data.world/crowdflower/hate-speech-identification)
* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
    * Pull the Twitter pre-trained vectors and put them all in /data
    
## Code

* `offensive_tweets.ipynb` Jupyter notebook to run our training algorithm on the hate speech dataset
* `offensive_server/scoring_service` Python singleton class to load our ML model in `./models` and provide a programmatic inference service
* `offensive_server/predictor.py` Flask app which proxies the `ScoringService` to give an HTTP interface to the service

## Example Request

```
POST /invocations
{
    "instances":[
        "this is a test post"
    ]
}
```

response:
```
{
    "predictions": [
        [
            0.7499632239341736, //confidence inoffensive
            0.2500367760658264 //confidence offensive
        ]
    ]
}
```


### Making Ready For Sagemaker

You might want to better read: [Use Your Own Algorithms or Models with Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html)

1. Add a `GET /ping` endpoint to the flask app 
2. Have the model be picked up from `/opt/ml` instead of `./models`. Sagemaker will dump the model files in this directory.
3. Make the Flask app run with WSGI, behind Nginx, and be initialized from a script named `server`. For a similar example, check out: [SciKit bring your own](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own/container/decision_trees)
4. Add a Dockerfile for setup and push to [AWS ECR](https://aws.amazon.com/ecr/)