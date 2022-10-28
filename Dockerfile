FROM firedrakeproject/firedrake-vanilla:latest

WORKDIR /home/torch_firedrake
COPY requirements.txt requirements.txt
RUN bash -c ". /home/firedrake/firedrake/bin/activate && pip install -r requirements.txt"

COPY setup.py .
COPY torch_firedrake .
RUN bash -c ". /home/firedrake/firedrake/bin/activate && pip install -e ."

ENV OMP_NUM_THREADS 1