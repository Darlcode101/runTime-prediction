
ok so for this project

1. Find only athletes with at least 3 races, these can be from all times, however 1 needs to be a 5k

2. experiment with different ML techniques to figure out what factors predict the best, figure out why 
   some methods are better than others 

   Linear Regression — baseline ML model
    Random Forest
    XGBoost
    potentially Gradient Boosting / Extra Trees 

3. compare this to traditional riegel formula 
            T2​=T1​(D1​D2​​)1.06

4. find the difference between my findings and normal to calculate increased accuracy 

5. figure out how im gonna get this on AWS

   Deployed as a Lambda function (container image, arm64) behind a public
   Function URL, via the AWS Lambda Web Adapter so the same Flask app runs
   locally and on Lambda unmodified. See deploy.sh for the redeploy flow.
   Function URL: https://pufcam4qxmpucbls56xfork4fm0bxurt.lambda-url.eu-central-1.on.aws/
   (currently 403 - AWS restricts public Function URLs on new accounts until
   verified, should clear on its own within ~24h)

