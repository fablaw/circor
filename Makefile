reinstall_package:
	@pip uninstall -y circor || :
	@pip install -e .

run_preprocess:
	python -c 'from circor.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from circor.interface.main import train; train()'

run_pred:
	python -c 'from circor.interface.main import pred; pred()'

run_evaluate:
	python -c 'from circor.interface.main import evaluate; evaluate()'

run_all: run_train run_evaluate run_pred

streamlit:
	-@streamlit run app/app.py

run_api:
	uvicorn circor.api.api:app --reload

GCP_PROJECT_ID=wagon-bootcamp-1002
DOCKER_IMAGE_NAME=circor
GCR_MULTI_REGION=eu.gcr.io
GCR_REGION=europe-west1

docker_params:
	@echo "project id: ${GCP_PROJECT_ID}"
	@echo "image name: ${DOCKER_IMAGE_NAME}"
	@echo "multi region: ${GCR_MULTI_REGION}"
	@echo "region: ${GCR_REGION}"

docker_build:
	docker build -t ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} .

docker_run:
	docker run -e PORT=8000 -p 8000:8000 ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}

# 🚨 additional step for apple silicon only (you will not be able to run this new image locally but it will work on production)
docker_build_m1:
	docker buildx build --platform linux/amd64 -t ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} --load .

docker_push:
	docker push ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}

docker_deploy:
	gcloud run deploy --image ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region ${GCR_REGION}
