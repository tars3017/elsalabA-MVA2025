docker build -t elsalab_a -f docker/Dockerfile .
docker run -it --rm \
    -v $(pwd):/Co-DETR \
    --gpus all \
    elsalab_a
