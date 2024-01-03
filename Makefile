download_models:
	sh utils/download_models.sh

lint:
	black src/
	black notebooks/