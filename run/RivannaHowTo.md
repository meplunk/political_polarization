# How to get set up with Rivanna computing OnDemand:

## One-time set-up:
First, go to https://ood.hpc.virginia.edu and log in with your NetBadge. Go to Files > /scratch/your-computing-id and upload the whole political_polarization folder (including raw data). Then, you need to set up your virtual environment. Go to Clusters > HPC Shell Access and input the following

```bash
cd /scratch/your-computing-id/political_polarization
module load gcc/11.4.0  openmpi/4.1.4
module load python/3.11.4
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If there is an error with installing python, input 
```bash
module spider python
```
and follow the instructions from there.

## To run a job:
Each file has an associated .slurm file in the /run folder that will run it. To do so from the HPC Shell, input the following:
```bash
module purge
module load gcc/11.4.0  openmpi/4.1.4
module load python/3.11.4
cd /scratch/your-computing-id/political_polarization
source venv/bin/activate
sbatch run/file-name.slurm
```

To check your progress while (or after) it is running, input:
```bash
squeue -u your-computing-id
```
The "ST" (status) column will tell you your progress; PD = pending, CF = configuring, R = running, blank = done.

You can also watch the progress by looking at the .out file; input:
```bash
tail -f logs/tokenize_746XXXXX.out
```
(replace with your actual job number). 




