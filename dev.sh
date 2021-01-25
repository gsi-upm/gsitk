#!/bin/bash

docker build -t gsitk:dev -f Dockerfile.dev .

docker run --rm \
  --name gsitk_dev \
	-e DATA_PATH="/tmp" -e RESOURCES_PATH="/usr/src/app/gsitk/resources/" \
	-ti \
	-p 8888:8888 \
	-v $(pwd):/usr/src/app \
	gsitk:dev
