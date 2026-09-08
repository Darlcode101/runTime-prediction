FROM python:3.13-slim

# Lets this container run as a Lambda function (behind a Function URL) with
# zero code changes: it proxies Lambda invoke events to HTTP calls against
# the app's normal web server. See https://github.com/awslabs/aws-lambda-web-adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_READINESS_CHECK_PATH=/health

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py predict.py features.py train_final_model.py ./
COPY templates/ templates/
COPY model/ model/

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "app:app"]
