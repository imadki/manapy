#!/bin/bash

cd ../../
docker build -t manapy:0.4 -f ./manapy/Docker/Dockerfile .
# docker run -it --rm -v "$PWD:/workspace/manapy" manapy:0.4
#sudo docker run --name manapy -d init manapy:0.4
#sudo docker exec -it manapy /bin/bash

# docker login
# docker tag myapp yourusername/myapp:latest
# docker push yourusername/myapp:latest

# Inside an empty directory
#git clone -b manapy-1.0 https://github.com/imadki/manapy .
#docker run --name manapy_container -it --rm -v "$PWD:/workspace/manapy" ayoub8899/manapy:1.1
## Commands inside docker
#python3 -m pip install -e .
#cd meshes/big
#wget -O data.zip https://foxer19.hopto.org/nextcloud/public.php/dav/files/6J5mjSEjfgijmHk/big/var/?accept=zip
#unzip data.zip