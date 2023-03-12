# equivariant-muzero


Start docker container as:

```bash
docker run -it \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /home/ando/Projects/equivariant-muzero:/root/equivariant-muzero rocm/pytorch
```