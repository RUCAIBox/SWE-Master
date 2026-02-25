pip install aria2
bash ./data_preparation/hfd.sh R2E-Gym/SWE-Bench-Verified --dataset  --tool aria2c
bash ./data_preparation/hfd.sh SWE-Gym/SWE-Gym --dataset  --tool aria2c
bash ./data_preparation/hfd.sh R2E-Gym/R2E-Gym-Subset --dataset  --tool aria2c
bash ./data_preparation/hfd.sh nebius/SWE-rebench --dataset  --tool aria2c
bash ./data_preparation/hfd.sh SWE-bench/SWE-smith --dataset  --tool aria2c