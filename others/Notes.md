# Project notes

### update requirement.txt
`pip3 freeze > requirements.txt`<br>
`pip install -r requirements.txt` <br>

### docker cmd
docker build -f Dockerfile -t unetr-project:latest .
docker run --gpus all -itd [docker_image]

#### docker bind mount
docker run --gpus all --shm-size 8G -itd -v D:/Capstone/dataset:/workspace/unetr-project/datasets unetr-project
docker run --gpus all -itd -v D:/Capstone/dataset:/workspace/unetr-project/datasets unetr-project

docker exec -it [docker_image] bash
cd mnt/c/users/kjiak/nus-iss/Project2/unetr-project/

### docker copy
docker cp container_id:/foo.txt foo.txt

### dump into a Bash session instead of inside Python 3.9 REPL
CMD ["/bin/bash"] replace with docker exec

### pip
`pip install monai matplotlib nibabel tqdm einops"`

### tensorboard
`python -m tensorboard.main --logdir='D:\ITSS\unetr-project\lightning_logs\version_0'`

### empty annotations
`https://blog.roboflow.com/missing-and-null-image-annotations/`

### debugging
`python3 -m torch.utils.collect_env`
