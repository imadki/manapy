#!/bin/bash

cd ../../
docker build -t manapy:0.4 -f ./manapy_v2/Docker/Dockerfile .
#sudo docker run --name manapy -d init manapy:0.4
#sudo docker exec -it manapy /bin/bash
