# coding: utf-8
#
#  __authors__ = Elena Khramenkova
#  __institution__ = TU Delft
#
#
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import MDAnalysis as mda 
from MDAnalysis.analysis import align
import seaborn as sns
from sklearn import manifold
from sklearn_extra.cluster import KMedoids
import itertools
from sklearn.metrics import silhouette_score


# Perform the permutations ina  file where 6 intial configrations are saved with already premuted Cu/Al atoms.
# Giving 6 intial configurations in a files, performing permutaitons of 4 oxygen atoms on each conf gives 144 strucutres in total
dic = os.getcwd()
# for Cu atoms: 6 permutations for each O reshuffle
# permutations 
def permutations(path_initial_conf, path_final_conf):
    """
    

    Parameterss
    ----------
    path_initial_conf : str 
         the path to the initial configuration.
    path_final_conf : str
        the path where the set of structures with permutations should be placed.

    Returns
    -------
    None.

    """
    
    with open(dic + path_initial_conf, "r") as f:
        traj_list = f.readlines()
        coordinates_all = []
        m =len(range(0, len(traj_list), 154))
        coordinates_all = np.array_split(traj_list, m)
        oxy_atoms = [147, 148, 150, 152] # these are the oxygens atoms participating in the permutations
        with open(dic + path_final_conf, "w") as file:
            for structure in coordinates_all:
                for l in itertools.permutations(structure[oxy_atoms]) :
                    file.writelines(structure[0:147])
                    file.writelines(l[0:2])
                    file.writelines(structure[149])
                    file.writelines(l[2])
                    file.writelines(structure[151])
                    file.writelines(l[3])
                    file.writelines(structure[153])
        file.close()
    f.close()

## THERE ARE TWO TYPES OF INITIAL CONFIGURATIONS. WE ANALYZE THEM SEPARATELY #######
dic = os.getcwd()
class Align:
    def __init__(self, name, ref_name, directory):
        """
        

        Parameters
        ----------
        name : str
            name of the file/trajectory.
        ref_name : str
            path to the set of reference structures with permutaitons.
        directory : str
            path to the trajectory to be analyzed.

        Returns
        -------
        None.

        """
        
        self.name = name
        self.ref_name = ref_name
        self.directory = directory
        self.rmsd_file = []
        self.min_rmsd_to_df = []
        self.ref = mda.Universe(dic + self.ref_name, all_coordinates=True)
    
    def iterate_traj(self):
        
        print("Iterating through the trajectories of " + self.name)
        for subdir, dirs, files in os.walk(dic + self.directory):
            for file in files:
                if file.endswith("-pos-1.xyz"):
                    universe = mda.Universe(os.path.join(subdir, file))
                    universe.transfer_to_memory(start = 50, step=500, stop=8550) #170 frames which is 8500 in total (4.25 ps)### Aligning the trajectories ###
                    #alignment of each selected structure from the trajectory with the all the perfutations listed in references
                    for ts in universe.trajectory:
                        for ts in self.ref.trajectory:
                            Al_ = align.alignto(universe, self.ref, select = "index 144:151")
                            #rmsd from alignment: name of traj, frame of ref, max rmsd, average rmsd
                            self.rmsd_file.append([file, universe.select_atoms("index 144:151").ts.frame, \
                                              self.ref.select_atoms("index 144:151").ts.frame, Al_[1]])
                    
        #convert the final list into array
        self.rmsd_file = np.array(self.rmsd_file)
                            
    
    
    def make_correct_dataframe(self):
        
        dataframe_after_alignment = pd.DataFrame({"name": self.rmsd_file[:,0], "traj_frame": self.rmsd_file[:,1], \
                      "ref_frame": self.rmsd_file[:,2], "average_rmsd":self.rmsd_file[:,3]})
                                   
        # beginning is 0, step is number of trajectories * number of steps * number of references 
        # (which is the number of rows in the rmsd file)
        # separate the data in 4 different trajectories
        dif_traj = range(0, self.rmsd_file.shape[0], len(self.ref.trajectory))
        for i in dif_traj:
            self.min_rmsd_to_df.append(dataframe_after_alignment.loc[dataframe_after_alignment["average_rmsd"] \
                                == min(dataframe_after_alignment["average_rmsd"][i:i+144])].values)
        # reshape the data 
        self.min_rmsd_to_df = np.array(self.min_rmsd_to_df).reshape(68,4)
                            
              
# Do the alignment with the reference structure that gives the lowest RMSD 
class Align_min:
    def __init__(self, name, ref_name, directory, min_rmsd):
        """
        

        Parameters
        ----------
        name : str
            name of the trajectory/file.
        ref_name : str
            path to the set of reference structures with permutaitons.
        directory : str
            path to the trajectory to be analyzed.
        min_rmsd : array
            contains the data on the lowest RMSD values for each snapshot.

        Returns
        -------
        None.

        """
        
        self.name = name
        self.directory = directory
        self.min_rmsd = min_rmsd
        self.ref_name = ref_name
        self.universe_all_no_al = []
        self.list_rmsd_min = []
        self.traj_atoms = []
        self.traj_all = []
        self.sse = []
        self.ref = mda.Universe(dic + self.ref_name, all_coordinates=True)
        

    
    
    def iterate_traj_min(self):
        print("Iterating through the trajectories of " + self.name)
        for subdir, dirs, files in os.walk(dic + self.directory):
            for file in files:  
                if file.endswith("-pos-1.xyz"):
                    print(os.path.join(subdir, file))
                    universe = mda.Universe(os.path.join(subdir, file))
                    universe.transfer_to_memory(start = 50, step=500, stop=8550) #170 frames which is 8500 in total (4.25 ps)
                    #save the trajectory with NO ALIGNMENT
                    for ts in universe.trajectory:
                        self.universe_all_no_al.append(universe.select_atoms("all").ts.positions)
                        #there are 170 frames in each trajectory nd we have 680 references in the ref_
                    for ts, ts in zip(universe.trajectory, \
                                      self.ref.trajectory[self.min_rmsd[:,2].astype(int)]):
                        Al_ = align.alignto(universe, self.ref, select = "index 144:151")
                        self.list_rmsd_min.append([file, universe.trajectory.ts.frame, \
                                          self.ref.trajectory.ts.frame, Al_[1]])
                        #save the trajectory after alignment
                        self.traj_atoms.append(universe.select_atoms("index 144:151").ts.positions)
                        self.traj_all.append(universe.select_atoms("all").ts.positions)
                        
        #convert our final list with cluster atoms and cluster/framework atoms to array
        self.traj_atoms = np.array(self.traj_atoms).reshape(68, 24)
        self.traj_all = np.array(self.traj_all)
        self.list_rmsd_min = np.array(self.list_rmsd_min)
        self.universe_all_no_al = np.array(self.universe_all_no_al)
    
        
    def create_a_dataframe(self, name_of_the_file, sheet_name):
        """
        

        Parameters
        ----------
        name_of_the_file : str
            the path to the dataframe.
        sheet_name : str
            name of the sheet in the excel file.

        Returns
        -------
        dataframe

        """
        
        df_to_save = pd.DataFrame(self.list_rmsd_min, \
                                  columns=["Name_of_the_trajectory", "Frame_of_the_trajectory", \
                                           "Reference_from_144", "RMSD,A"], index=range(0, len(self.list_rmsd_min)))
         
        writer = pd.ExcelWriter(name_of_the_file, engine="xlsxwriter")
        df_to_save.to_excel(writer, sheet_name = sheet_name)
        writer.save() 
        return df_to_save
        
    
    @staticmethod
    def print_rmsd_tajectories(df):
        """
        This function can print the RMSD change for the trajectories.

        Parameters
        ----------
        df : dataframe
            this dataframe can be generated by using the create_a_dataframe() function above.

        Returns
        -------
        None.

        """
        
        df[['Frame_of_the_trajectory', 'Reference_from_144']] = df[['Frame_of_the_trajectory', 'Reference_from_144']].astype(int)
        df['RMSD,A'] = df['RMSD,A'].astype(float)
        
        # print the RMSD values for trajectories where color is based on the type of trajectory
        sns.lineplot(data=df, x='Frame_of_the_trajectory', y='RMSD,A', hue="Name_of_the_trajectory")
        plt.show()
        # save the figure
        fig.savefig(dic + 'graph.png', bbox_inches='tight', dpi=300)
    

    def choose_number_of_clusters(self):
        # Choose the optimal number of clusters for K-medoids algorithm by applying a Silhouette score
        n_clusters = range(2, 65)
        for i in n_clusters:
            clustering = KMedoids(n_clusters = i, max_iter = 1000, random_state=10).fit(self.traj_atoms)
            silhouette_avg = silhouette_score(self.traj_atoms, clustering.labels_, metric="euclidean")
            self.sse.append([i, clustering.inertia_,silhouette_avg])
        self.sse = np.array(self.sse)
        Sil_max = np.where(self.sse[:,2] == max(self.sse[:,2]))
        number_of_clusters = self.sse[Sil_max,:1]
        plt.plot(self.sse[:,0], self.sse[:,2],markersize=12, color='skyblue', linewidth=4)
        plt.title('Optimal Number of Clusters \n using Silhouette Method \n for all trajectories')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score_')
        fig.savefig(dic + 'Sil_score.png', bbox_inches='tight', dpi=300)
        plt.show()
        
        return number_of_clusters

    def do_clustering(self, number_of_clusters):
        """
        This fucntion employes the number of clusters that was found to be optimal in 
        the previous procedure of testing the range of clusters.

        Parameters
        ----------
        number_of_clusters : int
            number of clusters for the K-medoids clustering procedure.

        Returns
        -------
        array
            medoids of the found clusters.

        """
        
        clustering_= KMedoids(n_clusters = int(number_of_clusters), random_state=10).fit(self.traj_atoms) 
        u_sil_avg = silhouette_score(self.traj_atoms, clustering_.labels_, metric="euclidean")
        
        # Reduce the dimenstionality and project the data on 2D surface using t-sne
        tsne = manifold.TSNE(n_components=2, perplexity=22)
        tsne_ = tsne.fit_transform(self.traj_atoms)
        
        # Plot the data from the dimensionality reduction technique: clusters
        plt.figure(figsize = (5,5))
        sns.scatterplot(tsne_[:,0], tsne_[:,1], 
                hue= clustering_.labels_,markers="o", s=100, palette='deep').set_title('KMeans Clusters Derived from Original Dataset \n PCA DIM. Red. \n for all trajectories', fontsize=10)
        # Plot the data from the dimensionality reduction technique: clusters' centers
        sns.scatterplot(tsne_[clustering_.medoid_indices_][:,0], tsne_[clustering_.medoid_indices_][:,1], marker="o",               zorder=100, linewidth=2, color="black")
        plt.legend(loc=(1.15,0), ncol=7)
        plt.ylabel('t-SNE2')
        plt.xlabel('t-SNE1')
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(dic + 't-SNE.svg', dpi=300)
        plt.show()
        return clustering_.medoid_indices_

    def make_xyz_files(self, path_first_str, path_xyz_clusters, medoid_indices):
        """
        

        Parameters
        ----------
        path_first_str : TYPE
            DESCRIPTION.
        path_xyz_clusters : TYPE
            DESCRIPTION.
        medoid_indices : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        
        # Find the centers of clusters. Print them as xyz files
        # Write xyz file for further analysis
        
        # save the atoms names from the xyz file
        # we use this initila structure as the reference for RMSD calculations and for collecting the atoms     
        with open(dic + path_first_str, "r") as f:
            traj_list = f.readlines()
            atoms_xyz = []
            for i in range(2, len(traj_list)):
                atoms_xyz.append(traj_list[i][0:2])
        f.close()

        ### Write the final trajctory ###
        with open(dic + path_xyz_clusters, "w") as file:
            for i, m in zip(np.array(self.universe_all_no_al[medoid_indices]), medoid_indices):
                file.writelines(str(152) + "\n")
                file.writelines("med_bef_opt, " + str(m) + "\n") 
                for k, l in zip(i, atoms_xyz):
                    file.writelines(str(l) + str(k).replace('[', ' ').replace(']', ' ') + '\n')
        file.close()
                

if __name__ == '__main__':
    ##### Do permutations first #####
    #permutations for MD-1_m3
    permutations('\\MD-1_m3\\initial_configuration_MD-1_m3.xyz', '\\MD-1_m3\\initial_configuration_MD-1_m3_topology_.xyz')
    #permutations for MD-4_m5
    permutations('\\MD-4_m5\\initial_configuration_MD-4_m5.xyz', '\\MD-4_m5\\initial_configuration_MD-4_m5_topology_.xyz')
    
    ##### Do the alignment #####
    #### MD-1_m3 ####
    traj1 = Align('MD-1_m3', '\\MD-1_m3\\initial_configuration_MD-1_m3_topology.xyz',  r'\MD-1_m3')
    traj1.iterate_traj()
    
    # make the dataframe with the lowest RMSD values
    traj1.make_correct_dataframe()

    #### MD-4_m5 ####
    traj2 = Align('MD-4_m5', '\\MD-4_m5\\initial_configuration_MD-4_m5_topology.xyz',   r'\MD-4_m5')
    traj2.iterate_traj()
    
    ####### First generate the resutls for the MD-1_m3 trajectory #######
    # Define the object of the Align_min class
    traj1_min = Align_min('MD-1_m3',  '\\MD-1_m3\\initial_configuration_MD-1_m3_topology.xyz', \
                          r'\MD-1_m3', traj1.min_rmsd_to_df)
    # Separate the framework and clusters; do the alignment of the clusters with the reference structure that gives the lowest RMSD
    traj1_min.iterate_traj_min()
    
    # Generate the dataframe and print the RMSD changes for different types of structures over the time
    #df1 = traj1_min.create_a_dataframe('bla.xlsx', 'md-1_m3')
    #traj1_min.print_rmsd_tajectories(df1)
    
    # choose the best number of clusters nd print the Silhouette score vs number of clusters
    number_cl1 = traj1_min.choose_number_of_clusters()
    
    #use the best number of lusters to do the K-medoids clustering and print the clusters using t-sne technique
    medoids1 = traj1_min.do_clustering(number_cl1)
    
    #write tha xyz trajectory based on the path of the initial confgruation, xyz file you want to create
    # and indices of the medoids obtained in the best clustering procedure
    traj1_min.make_xyz_files('\\MD-1_m3\\initial_configuration_MD-1_m3_first_str.xyz', \
                             '\\MD-1_m3\\Cu2AlO4H_MD_1_m3_med_before_opt_.xyz', medoids1)

    ####### First generate the resutls for the MD-1_m3 trajectory #######
    # Define the object of the Align_min class
    traj2_min = Align_min('MD-4_m5',  '\\MD-4_m5\\initial_configuration_MD-4_m5_topology.xyz', \
                          r'\MD-4_m5', traj2.min_rmsd_to_df)
    # Separate the framework and clusters; do the alignment of the clusters with the reference structure that gives the lowest RMSD
    traj2_min.iterate_traj_min()
    
    # Generate the dataframe and print the RMSD changes for different types of structures over the time
    #df2 = traj1_min.create_a_dataframe('bla.xlsx', 'md-4_m5')
    #traj2_min.print_rmsd_tajectories(df2)
    
    # choose the best number of clusters nd print the Silhouette score vs number of clusters
    number_cl2 = traj2_min.choose_number_of_clusters()
    
    #use the best number of lusters to do the K-medoids clustering and print the clusters using t-sne technique
    medoids2 = traj2_min.do_clustering(number_cl2)
    
    #write tha xyz trajectory based on the path of the initial confgruation, xyz file you want to create
    # and indices of the medoids obtained in the best clustering procedure
    traj2_min.make_xyz_files('\\MD-4_m5\\initial_configuration_MD-4_m5_first_str.xyz', \
                             '\\MD-4_m4\\Cu2AlO4H_MD_4_m5_med_before_opt_.xyz', medoids1)

