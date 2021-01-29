MODEL_NAME="clase_20210129"
INPUT_DATA_FILE="data/instances.json"
VERSION_NAME="versionclase20210129"
REGION="europe-west1"

gcloud ai-platform predict --model $MODEL_NAME \
--version $VERSION_NAME \
--json-instances $INPUT_DATA_FILE \
--region $REGION