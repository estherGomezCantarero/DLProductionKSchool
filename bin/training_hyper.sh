#!/bin/sh

BUCKET=esther_gomez_cantarero_20210123_kschool

gcloud ai-platform jobs submit training mnist_int_ht_`date +"%s"` \
    --python-version 3.7 \
    --runtime-version 2.3 \
    --scale-tier BASIC \
    --package-path ./trainer \
    --module-name trainer.task \
    --region europe-west1 \
    --job-dir gs://$BUCKET/tmp \
    --config ./bin/hyper.yaml \
    -- \
    