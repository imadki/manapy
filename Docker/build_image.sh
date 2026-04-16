#!/bin/bash

cd ../../
docker build -t manapy:0.4 -f ./manapy/Docker/Dockerfile .
# docker run -it --rm -v "$PWD:/workspace/manapy" manapy:0.4
#sudo docker run --name manapy -d init manapy:0.4
#sudo docker exec -it manapy /bin/bash
