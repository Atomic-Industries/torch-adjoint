interactive := -i
image-tag := torch-adjoint
host-port := 8888
mounts := -v `pwd`:/home/torch_adjoint
run := docker run --shm-size=3g --rm  --entrypoint="" $(mounts) $(interactive) -t $(image-tag)

.PHONY: build bash test

build:
	@docker build -t $(image-tag) -f ./Dockerfile .

bash:
	@$(run) bash