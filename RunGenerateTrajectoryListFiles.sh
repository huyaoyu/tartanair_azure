#!/bin/bash

echo "Generate the trajectory list files."

python GenerateTrajectoryFiles.py \
    /home/yaoyuh/Scripts/tartanair_azure/tartanair \
    ./TrajectoryFiles \
    trajfiles/Easy_*_relative.txt \
    --out-prefix Easy/

python GenerateTrajectoryFiles.py \
    /home/yaoyuh/Scripts/tartanair_azure/tartanair \
    ./TrajectoryFiles \
    trajfiles/Hard_*_relative.txt \
    --out-prefix Hard/

echo "Done with generating trajectory list files."
