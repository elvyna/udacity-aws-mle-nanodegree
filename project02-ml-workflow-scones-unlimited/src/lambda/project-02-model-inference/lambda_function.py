import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "" ## TODO: fill in

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["image_data"]) ## TODO: fill in

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(
        ENDPOINT,
        sagemaker_session=sagemaker.Session()
    ) ## TODO: fill in

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    predictor.serializer = IdentitySerializer("image/png")
    with open(f"./{event['s3_key']}", "rb") as f:
        payload = f.read()

    # Make a prediction:
    inferences = predictor.predict(payload) ## TODO: fill in

    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }