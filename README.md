# CLUSTERING MD TRAJECTORIES

This script can be executed if the molecular dynamics trajectories are present in the folder. 
Here we specifically aim to see how the position of the active site (Cu-oxo complex) evolves in the molecular dynamics trajectories.
This script accounts for the permutations of the atoms in the active site, alignment of the whole trajectory and accounting for the reference giving the lowest RMSD displacement. 
With the active site coordinates chosen throughout trajectory, the clustering procedure has been carried out useinf unsupervised machine learning algorithms K-medoids. 
As a result of running this script, the coordinates of the centers of the clusters are generated. 

## Installation

```
git clone https://github.com/ekhramen/clustering_MD_trajectories.git   

```

* [trajectories]: This folder contains the molecular dynamics trajectories. The snapshots from these trajectories were used for the clustering procedure.
* [scripts]: This folder contains the scripts used to run clustering.
* [example]: This folder contains working script with inserted value corresponding to trinuclear Cu-oxo complex in mordenite.
This script was created and executed using Spyder 5.0.5 and Python 3.9.6.
More information on Spyder could be found here: https://www.spyder-ide.org/

## Instructions
To execute the script, the following files have to present:
1. Specify the folder with the data.
2. Intial configuration used as a reference for permutations and alignment of the trajectory.

## Steps in the script
This script contains the following steps:
1. Perfrom permutations of the atoms in the investigated active site.
In this procedure, a xyz is generated where all possible displacements of the atoms of the tracked active site are accounted for.
2. In the nex step, trajectries with targeted active sites are treated.
The snapshots of the trajectories are aligned with the reference structures generated in the step 1 via permutations.
3. The results of the step 3 are the RMSD values obtained from the alignment procedure of the trajectories with the reference structures.
This result is saved in the dataframe.
4. In the next step, the dataframe from step 3 is used for the alignment of the trajecotories with the reference strucutre that gives the best alignment (the lowest RMSD).
5. A dataframe is created where the name of trajectory, the snapshot number, number of the best reference from the alignment, RMSD value.  
6. Perform the unsupervised machine learning clustering procedure K-medoids to choose the best number of clusters.
Choose the number of clusters based on best Silhouette score.
7. Perform clustering procedure using the best number of clusters from step 6. Print these clusters using the t-sne a method.
8. Identify the centres of the clusters and save their corresponding xyz coordinates in the folder. 




