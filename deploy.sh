#!/usr/bin/env bash
# Rebuilds the container image and pushes a new version to the already-provisioned
# Lambda function. Run this after retraining the model or changing app.py/predict.py.
#
# One-time setup (already done, kept here for reference/rebuilding from scratch):
#   aws ecr create-repository --repository-name run-predictor --region eu-central-1
#   aws iam create-role --role-name run-predictor-lambda-role \
#       --assume-role-policy-document file://lambda-trust-policy.json
#   aws iam attach-role-policy --role-name run-predictor-lambda-role \
#       --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
#   aws lambda create-function --function-name run-predictor --package-type Image \
#       --code ImageUri=$ECR_URI:latest --role <role-arn> --architectures arm64 \
#       --memory-size 1024 --timeout 30 --region eu-central-1
#   aws lambda create-function-url-config --function-name run-predictor --auth-type NONE
#   aws lambda add-permission --function-name run-predictor \
#       --statement-id FunctionURLAllowPublicAccess --action lambda:InvokeFunctionUrl \
#       --principal "*" --function-url-auth-type NONE

set -euo pipefail

REGION="eu-central-1"
ACCOUNT_ID="357021374993"
REPO="run-predictor"
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO"

echo "Building image..."
# --provenance=false/--sbom=false: Lambda rejects the extra attestation
# manifests Docker Desktop adds by default.
docker build --provenance=false --sbom=false -t "$REPO:latest" .

echo "Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URI"

echo "Pushing image..."
docker tag "$REPO:latest" "$ECR_URI:latest"
docker push "$ECR_URI:latest"

echo "Updating Lambda function code..."
aws lambda update-function-code \
  --function-name "$REPO" \
  --image-uri "$ECR_URI:latest" \
  --region "$REGION" > /dev/null

aws lambda wait function-updated --function-name "$REPO" --region "$REGION"

URL=$(aws lambda get-function-url-config --function-name "$REPO" --region "$REGION" --query FunctionUrl --output text)
echo "Deployed. Function URL: $URL"
