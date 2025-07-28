"""
Takes cluster labels of shape (nens,npts) - nemi_pack
Calculates entropy and overlap
"""

import pandas as pd
import numpy as np
import os,sys
import logging
import multiprocessing



class Entropy():

    def __init__(self,ensembles,max_ens=None,verbose=True):

        self.ensembles = ensembles
        self.nc = int(np.max(ensembles)) + 1
        self.ensembles = ensembles
        if max_ens == None:
            print('Using full ensemble')
            self.nens = self.ensembles.shape[0]
        else:
            print(f'Using {max_ens} realisations')
            self.nens = max_ens
            print('Using full ensemble')
        self.npts = self.ensembles.shape[1]

        self.verbose = verbose
        # return        


    def relabel(self,base_id, max_clusters=None, ):

        """
        Written by Maike Sonnewald
        ensembles: nemi_pack - should have dimension (n_ens,npts)
        """
        # print(base_id)
        
        # self.base_id = base_id
        base_labels = self.ensembles[base_id]
        compare_ids = [i for i in range(self.nens)]
        compare_ids.pop(base_id)

        num_clusters = int(np.max(base_labels) + 1)


        # if not pre-set, set max number of clusters to total number of clusters in the base
        if max_clusters is None:
            max_clusters = num_clusters

        sortedOverlap=np.zeros((len(compare_ids)+1, max_clusters, base_labels.shape[0]))*np.nan

        # print(num_clusters, max_clusters)
        summaryStats=np.zeros((num_clusters, max_clusters))

        # compile sorted cluster data
        # TODO: add assert statement to make sure that the clusters have been sorted?


        # dataVector=[nemi.clusters for id, nemi in enumerate(self.nemi_pack) if id != base_id]
        dataVector=[self.ensembles[id] for id, nemi in enumerate(self.ensembles) if id != base_id]

        # loop over ensemble members, not including the base member
        for compare_cnt, compare_id in enumerate(compare_ids):
            # grab clusters of ensemble member
            compare_labels= dataVector[compare_cnt]

            # go through each cluster in the base and assess the percentage overlap
            # for every cluster in the ensemble member (overlap / total coverage area) 
            for c1 in range(max_clusters): 
                # Initialize dummy array to mark location of the cluster for the base member
                data1_M = np.zeros(base_labels.shape, dtype=int)
                # mark where the considered cluster is in the member that is being used as the baseline
                data1_M[np.where(c1==base_labels)] = 1 
                # # Count numer of entries [Why?] 
                summaryStats[0, c1]=np.sum(data1_M) 

                # go through each cluster
                # k = 0
                for c2 in range(num_clusters):
                    # Initialize dummy array to mark where the cluster is in the comparison member
                    data2_M = np.zeros(base_labels.shape, dtype=int) 

                    # mark where the considered cluster is in the member that is being used as the comparison
                    data2_M[np.where(c2==compare_labels)] = 1    

                    # Sum of flags where the two datasets of that cluster are both present
                    num_overlap=np.sum(data1_M*data2_M)       

                    #Sum of where they overlap
                    num_total=np.sum(data1_M | data2_M)       

                    #Collect the number that is largest of k and the num_overlap/num_total
                    # k = max(k, num_overlap / num_total)       
                    summaryStats[c2, c1]=(num_overlap / num_total)*100 # Add percentage of coverage

                #Filled in 'summaryStatistics' matrix results of percentage overlaps

            usedClusters = set() # Used to mak sure clusters don't get selected twice
            #Clusters are already sorted by size
            
            sortedOverlapForOneCluster=np.zeros(base_labels.shape, dtype=int)*np.nan
            # go through clusters from (biggest to smallest since they are sorted)
            for c1 in range(max_clusters):  
                sortedOverlapForOneCluster=np.zeros(base_labels.shape, dtype=int)*np.nan
                #print('cluster number ', c1, summaryStats.shape, summaryStats[1:,c1-1].shape)

                # find biggest cluster in first column, making sure it has not been used
                sortedClusters = np.argsort(summaryStats[:, c1])[::-1]
                biggestCluster = [ele for ele in sortedClusters if ele not in usedClusters][0]

                # record it for later
                usedClusters.add(biggestCluster)

                # Initialize dummy array
                data2_M = np.zeros(base_labels.shape, dtype=int)

                # Select which country is being assessed
                data2_M[np.where(biggestCluster == compare_labels)]=1 # Select cluster being assessed

                sortedOverlapForOneCluster[np.where(data2_M==1)]=1
                sortedOverlap[compare_id, c1, :] = sortedOverlapForOneCluster

        # fill in the base entry in the sorted overlap
        for c1 in range(max_clusters):  
            sortedOverlap[base_id, c1, :] = 1 * (base_labels == c1)

        return sortedOverlap
    
    def _entropy(self,row,i=0):

        """
        Calculates entropy 
        called by get_entropy()
        """

        data = row['counts']
        L = sum(data) # ensemble size
        n = len(data) # number counts

        # print('ensemble size = '+str(L),'nclust = '+str(n))
        
        if n != 1:
            ress = 0
            for i in range(n):
                ress =  ress + (data[i]/L * np.log2(data[i]/L))
                                
        else:
                                
            ress = (data[i]/L * np.log2(data[i]/L))
        
        return ress*(-1)


    def get_entropy(self):

        """
        Realabels ensemble members for different base_ids
        Then calculates entropy; note _entropy() is applied 
        over columns of DataFrames containing counts
        """

        ent_lst = []
        df_count = []
        ov_lst = [] 
        
        for bid in range(self.nens):        
            if self.verbose:
                print('Running for..'+str(bid)+'th member')
            # self._log('Running for..'+str(bid)+'th member')

            ov = self.relabel(base_id=bid) # calculates relabelled clusters for a given base_id
            ov = np.nan_to_num(ov)
            df = pd.DataFrame(np.argmax(ov,axis=1)).T
            df = df.astype('int64')
            df_c = pd.DataFrame(
                df.stack().groupby(level=0).apply(lambda x: np.unique(x, return_inverse=True, return_counts=True)[2])
            )
            df_c.columns = ['counts']
            ent =  df_c.apply(self._entropy, axis=1) # entropy
            EntMax = (-1) * self.nc * (1/self.nc) * np.log2(1/self.nc)
            # print(ent)
            ent_lst.append(np.asanyarray((ent * 100)/EntMax))
            df_count.append(df_c)
            ov_lst.append(ov)
            
        self.ov = np.array(ov_lst)
        
        return np.asanyarray(ent_lst)


# -------------------------------------------------------------------------------------
    """
    Experimental: multiprocessing.pool to speed up?!
    """

    def entropy_for_bid(self,bid):

        """
        Calculate relabelling and entropy for a single base_id
        """
        ov = self.relabel(base_id=bid) # calculates relabelled clusters for a given base_id
        ov = np.nan_to_num(ov)
        df = pd.DataFrame(np.argmax(ov,axis=1)).T
        df = df.astype('int64')
        df_c = pd.DataFrame(
            df.stack().groupby(level=0).apply(lambda x: np.unique(x, return_inverse=True, return_counts=True)[2])
        )
        df_c.columns = ['counts']
        ent =  df_c.apply(self._entropy, axis=1) # entropy
        EntMax = (-1) * self.nc * (1/self.nc) * np.log2(1/self.nc)

        return np.array((ent * 100)/EntMax)
        # self.count = df_c
        # self.ov = ov   
    
    def run_parallel(self):

        """
        Use multiplrocess to apply entropy_for_bid() for different baselabel_ids     
        """
        bids = list(range(self.nens))
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.entropy_for_bid, bids)
        return np.array(results)

# -------------------------------------------------------------------------------------
    
    def get_overlap(self):

        """
        Calculates overlap from relabeled ensemble members
        Output array shape: (nens,nens)
        For each member, nens number of pairs, e.g., for base_id 0 with nens = 3, pairs are - 0 and 1, 0 and 2, 0 and 3.
        Similarly for base_id 1, pairs are - 1 and 0, 1 and 2, 1 and 3.
        overlap with itself 100% but used as nan in the output array
        """

        ovlp = np.zeros((self.nens,self.nens)) * np.nan
        array = self.ov # (ens,ens,nc,npts)
        for bid in range(self.nens):  
            for j in range(self.nens):
                if bid==j: # overlap with itself
                    continue
                    
                count = 0
                for i in range(self.nc):

                    common_ones = np.all(array[bid,[bid,j],i,:] == 1, axis=0) # overlap between 
                    count = count + np.sum(common_ones)
            
                count = count/self.npts
                count *= 100
                ovlp[bid,j] = count
            
        return ovlp

    def export_nemi_clusters(self):
        pass


# --------------------------------------------------------
# Chose which data to work with
# --------------------------------------------------------


# dummy = False # data switch


# if dummy:

#     """
#     Use a dummy cluster label data with nc = 4, nens = 50, npts = 10
#     """
#     np.random.seed(42)  # Set seed for reproducibility
#     arrays = []
#     while len(arrays) < 50:
#         arr = np.random.choice([0, 1, 2, 3], size=10)
#         # print(set(arr))
#         if len(set(arr)) == 4:  # Ensure all labels 0,1,2,3 are present
#             arrays.append(arr)
#     lab = np.array(arrays)

# else:
#     """
#     Use real data
#     """
#     # filename = './nemi_pack_nclust_4_nbr_100_mind_0.0.npy'
#     filename = './test_clusters.npy'
#     if os.path.exists(filename):
#         lab = np.load(filename)
#         print("File loaded successfully.")
#     else:
#         raise FileNotFoundError(f"Error: '{filename}' does not exist. Use dummy=True for dummy data to proceed")    

    
# ent = Entropy(lab)
# print(ent.get_entropy().shape)