screen -S tensorboard -dm bash -c "source .venv/bin/activate; tensorboard --bind_all --logdir checkpoints/runs/"
