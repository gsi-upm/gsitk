#!/bin/bash

docker build -t gsitk:debug -f Dockerfile.test .

docker run --rm -e DATA_PATH="/usr/src/app/tests/data" -ti gsitk:debug pytest --pdb
