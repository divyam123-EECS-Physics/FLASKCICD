FROM public.ecr.aws/lambda/python:3.11

# RUN sudo apt-get update && sudo apt-get install -y libgomp1

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN yum install libgomp -y
RUN pip install -r requirements.txt
# RUN yum install libgomp1
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]