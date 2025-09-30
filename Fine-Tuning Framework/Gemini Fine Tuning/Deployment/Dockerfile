# ==============================================================================
# Dockerfile for MineLLM Inference API
# ==============================================================================
# This Dockerfile packages the FastAPI or Ray Serve application for deployment.
# It uses an NVIDIA PyTorch image as a base to ensure all CUDA drivers are included.

# Start from an official NVIDIA PyTorch image to get CUDA drivers
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your chosen API script into the container.
# Make sure your script is named 'api.py' in your project folder.
COPY api.py .

# Expose the port the server will run on (e.g., 8000)
EXPOSE 8000

CMD ["serve", "run", "api:deployment", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# DEPLOYMENT INSTRUCTIONS (For reference)
# ==============================================================================

# ------------------------------------------------------------------------------
# Push to AWS Elastic Container Registry (ECR)
# (Requires 'aws-cli' configured)
# ------------------------------------------------------------------------------

# 1. Create an ECR repository:
#
#    aws ecr create-repository --repository-name minecraft-llm-api
#
# 2. Log in, build, tag, and push your Docker image:
#    (Replace ACCOUNT_ID and REGION with your AWS details)
#
#    # Log in Docker to your AWS ECR
#    aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
#
#    # Build the image from this Dockerfile
#    docker build -t minecraft-llm-api .
#
#    # Tag the image for ECR
#    docker tag minecraft-llm-api:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/minecraft-llm-api:latest
#
#    # Push the image to ECR
#    docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/minecraft-llm-api:latest
#

# ------------------------------------------------------------------------------
# Deploy on AWS Elastic Kubernetes Service (EKS)
# (Requires 'eksctl' and 'kubectl' configured)
# ------------------------------------------------------------------------------

# 1. Create an EKS cluster with GPU nodes:
#    (This command will create a small cluster and can take 15-20 minutes.)
#
#    eksctl create cluster --name minecraft-llm-cluster --nodegroup-name gpu-nodes --node-type g4dn.xlarge --nodes 1 --gpu-support
#
# 2. Apply the Kubernetes manifest (deployment.yaml):
#
#    kubectl apply -f deployment.yaml
#
# 3. Check the status and get the public URL:
#    (Wait a few minutes for the LoadBalancer to get an external IP)
#
#    kubectl get service minecraft-llm-service --watch

# Delete the EKS cluster and all its resources
# eksctl delete cluster --name minecraft-llm-cluster --region <REGION>

# Delete the ECR repository
# aws ecr delete-repository --repository-name minecraft-llm-api --force --region <REGION>