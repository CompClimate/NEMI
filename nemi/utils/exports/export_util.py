"""
Exports nemi_pack clusters to .npy files - shape: (n_ens,npts)
"""

import pandas as pd
import numpy as np
import os
import logging
from collections import OrderedDict
import re



class Export():

    def __init__(self,nc,nbr,mind,nens,datadir,verbose=True):

        self.nens = nens
        self.nc = nc
        self.nbr = nbr
        self.mind = mind
        self.datadir = datadir
        self.verbose = verbose
        self.npts = None

    def get_numbers_from_filename(self,filename):
        return re.findall(r'\d+', filename)        

    def create_nemi_pack(self):
        if self.verbose:
            print('Reading clusters..')
            
        cluster = {}
        for root, dirs, files in os.walk(self.datadir):
            for f in files:
                # if self.verbose:
                #     print(os.path.join(root, f))
                ld = np.load(os.path.join(root, f))
                num = self.get_numbers_from_filename(f)
        
                if int(num[0][0])  == 0:
                    ens = int(num[0][1])
                else:
                    ens = int(num[0])

                md = float(num[2]+'.'+num[3])
                knn = int(num[4])
        
                if knn==self.nbr and md==self.mind:
                    # if self.verbose:
                        # print(f'ens = {ens}, nbr = {knn} and md = {md}')
                        # print(f'ens = {ens}')
                    cluster[(ens,knn,md)] = ld   
                    # if self.verbose:
                    #     print(ld.shape)
                    

        self.npts = ld.shape[0]
        cluster_ = OrderedDict(sorted(cluster.items(), key=lambda t: t[0]))
        
        # Empty labels array
        lab = np.zeros((self.nens,self.npts)) * np.nan        
        for k1,k2 in enumerate(range(1,self.nens+1)):
            # print(k1, k2,)
            # print((k2, self.nbr, self.mind))
            lab[k1,:] = cluster_[(k2, self.nbr, self.mind)]

        if self.verbose:
            print(f'Finished creating nemi_pack of shape {self.nens},{self.npts}')
        return lab
        
