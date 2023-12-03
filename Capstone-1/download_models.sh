#!/bin/bash
mkdir models
cd models
wget --no-check-certificate --no-proxy "https://lemonbucketzoomcamp.s3.eu-central-1.amazonaws.com/models/model.info"
wget --no-check-certificate --no-proxy "https://lemonbucketzoomcamp.s3.eu-central-1.amazonaws.com/models/lemon.tflite"
wget --no-check-certificate --no-proxy "https://lemonbucketzoomcamp.s3.eu-central-1.amazonaws.com/models/lemon_rich_enrichments_13_0.998.h5"
mkdir lemon_saved && cd lemond_saved
wget --no-check-certificate --no-proxy "https://lemonbucketzoomcamp.s3.eu-central-1.amazonaws.com/models/lemon_saved/fingerprint.pb"
wget --no-check-certificate --no-proxy "https://lemonbucketzoomcamp.s3.eu-central-1.amazonaws.com/models/lemon_saved/saved_model.pb"
mkdir variables && cd variables
wget --no-check-certificate --no-proxy "https://lemonbucketzoomcamp.s3.eu-central-1.amazonaws.com/models/lemon_saved/variables/variables.index"
wget --no-check-certificate --no-proxy "https://lemonbucketzoomcamp.s3.eu-central-1.amazonaws.com/models/lemon_saved/variables/variables.data-00000-of-00001"
