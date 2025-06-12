include .env
export

deploy:
	gcloud run deploy ${SERVICE_NAME} \
		--source . \
		--region ${GOOGLE_CLOUD_LOCATION} \
		--project ${GOOGLE_CLOUD_PROJECT} \
		--allow-unauthenticated \
		--memory 2Gi \
		--cpu 2 \
		--concurrency 80 \
		--timeout 3600 \
		--set-env-vars="GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT},GOOGLE_CLOUD_LOCATION=${GOOGLE_CLOUD_LOCATION},SNOWFLAKE_ACCOUNT=${SNOWFLAKE_ACCOUNT},SNOWFLAKE_USER=${SNOWFLAKE_USER},SNOWFLAKE_PASSWORD=${SNOWFLAKE_PASSWORD},SNOWFLAKE_DATABASE=${SNOWFLAKE_DATABASE},SNOWFLAKE_SCHEMA=${SNOWFLAKE_SCHEMA}"

delete:
	gcloud run services delete ${SERVICE_NAME} --region ${GOOGLE_CLOUD_LOCATION} --project ${GOOGLE_CLOUD_PROJECT}

logs:
	gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}" --limit 50 --format "table(timestamp,jsonPayload.message)" --project ${GOOGLE_CLOUD_PROJECT}

status:
	gcloud run services describe ${SERVICE_NAME} --region ${GOOGLE_CLOUD_LOCATION} --project ${GOOGLE_CLOUD_PROJECT}

.PHONY: deploy delete logs status 