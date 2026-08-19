
import os
import gc
import json
import umap
import pickle
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
# import sciris as sc

__all__ = ['NEMI', 'SingleNemi', 'MODES', 'DEFAULT_EMBEDDINGS_PATH', 'EMBEDDINGS']

MODES = ('full', 'embed', 'cluster')
EMBEDDINGS = ('umap', 'tsne')
DEFAULT_EMBEDDINGS_PATH = 'nemi_embeddings.npz'

default_params = dict(
    device="cpu",
    embedding_dict=dict(method="umap", min_dist=0.0, n_components=3,
                        n_neighbors=20),
    clustering_dict=dict(method="agglomerative", linkage="ward",
                         n_clusters=30, n_neighbors=40),
)


def free_device_memory():
    """ Release the GPU buffers left behind by a finished cuML estimator.

    cuML estimators sit in reference cycles, so their device arrays survive
    until the cyclic collector runs -- and that collector is driven by host
    allocations, never by GPU pressure. Without this, each ensemble member
    leaves an embedding's worth of device memory behind until a fit runs out.
    """
    import cupy  # optional dependency, installed with the 'gpu' extra

    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()


def _num_clusters(labels):
    """Number of clusters in a label vector, ignoring NaN noise."""
    finite = labels[~np.isnan(labels)]
    return int(finite.max()) + 1 if finite.size else 0


class SingleNemi():
    """
    A single instance of the NEMI pipeline

    Args:
        params (dict, optional): A dictionary of the embedding and clustering options. Defaults to ``nemi.workflow.default_params``.
    """

    def __init__(self, params=None):

        # pipeline parameters
        # self.params = sc.mergedicts(default_params, params)
        # pipeline parameters
        self.params = copy.deepcopy(default_params)
        self.params.update(params if params is not None else {})

        # set during the run
        self.embedding = None
        self.clusters = None
        self.X = None

        return
    
    def run(self, X, output=None):
        """ Run a single instance of the NEMI pipeline

        The pipeline consists of steps:

        - fitting the embedding
        - predicting the clusters,
        - sorting the clusters by descending size

        Args:
            X (:py:class:`~numpy.ndarray`): The data contained in a sparse matrix of shape (``n_samples``, ``n_features``)
            output (str, optional): if given, write embedding + clusters to this
                ``.npz`` path (keys: ``embedding``, ``clusters``).
        """

        # fit the embedding
        print('Fitting the embedding')
        self.fit_embedding(X)

        # predict the clusters
        print('Predicting the clusters')
        self.clusters = self.predict_clusters()

        # sort the clusters by (descending) size
        print('Sorting clusters')
        self.clusters = self.sort_clusters(self.clusters)

        if output is not None:
            self.save_outputs(output)

    def scale_data(self, X):
        """ Scale the data to have a mean and variance of 1.

        Args:
            X (:py:class:`~numpy.ndarray`): The data to pick seeds for. A sparse matrix of shape (``n_samples``, ``n_features``)
            **kwargs : keyword arguments to embedding function
        """

        # scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X)
        return scaled_data

    def fit_embedding(self, X):
        """ Run the embedding algorithm on the data

        Args
            X (:py:class:`~numpy.ndarray`): The data to pick seeds for. A sparse matrix of shape (``n_samples``, ``n_features``)
            **kwargs : keyword arguments to embedding function
        """

        # initialize data
        self.X = X
        # run embedding on the configured device (cpu: umap-learn, gpu: cuML)
        embedding_fn = self.__embedding_algo(self.params['device'],
                                             **self.params['embedding_dict'])
        self.embedding = embedding_fn(self.X)
        # embedding_fn is a bound method of the estimator, and so the only thing
        # keeping its device arrays alive; drop it before the next member fits
        del embedding_fn
        if self.params['device'] == 'gpu':
            free_device_memory()


    def predict_clusters(self):
        """ Run the clustering algorithm on the embedding

        Method and parameters are set by the ``clustering_dict`` attribute.

        Returns:
            Identified clusters
        """
        device = self.params['device']
        cluster_params = dict(self.params['clustering_dict'])
        print(f"Clustering | device={device} | {cluster_params}")

        if device == "gpu":
            labels = self.__cluster_gpu(**cluster_params)
            free_device_memory()
        else:
            labels = self.__cluster_cpu(**cluster_params)

        print(f"Clusters found: {int(np.max(labels)) + 1}")
        return labels


    def sort_clusters(self, clusters):
        """ Updates cluster labels 0,1,...,k so that each cluster is of descending size.

        Args:
            clusters (:py:class`~numpy.ndarray`, list)

        Returns:
            An array with the new labels
        """

        # number of clusters (also the same as the label name in the agglomerated cluster dict)
        n_clusters = np.max(clusters)+1
        #  create a histogram of the different clusters
        hist,_ = np.histogram(clusters, np.arange(n_clusters+1))
        # clusters sorted by size (largest to smallest)
        sorted_clusters= np.argsort(hist)[::-1]
        # assign new labels where labels 0,...,k go in decreasing member size 
        new_labels = np.empty(clusters.shape)
        new_labels.fill(np.nan)
        for new_label, old_label in enumerate(sorted_clusters):
            new_labels[clusters == old_label] = new_label

        return new_labels
        
    def save_outputs(self, path):
        """ Save embedding + clusters to a single .npz (keys: embedding, clusters). """
        np.savez(path, embedding=self.embedding, clusters=self.clusters)

    def save(self, filename):
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)

    def save_embedding(self, filename):
        """ Save the embedding to a file

        Args:
            filename (str): Filename to save embedding
        """
        np.save(filename, self.embedding)

    def plot(self, to_plot=None, **kwargs):
        if to_plot.lower() == 'embedding':
            self._plot_embedding(**kwargs)
        elif to_plot.lower() == 'clusters':
            self._plot_clusters(**kwargs)

    def _plot_embedding(self, s=1, subsample=10, alpha=0.4):

        data = self.embedding

        fig = plt.figure()
        if data.shape[1] == 2:
            ax = plt.gca()
        elif data.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
        else:
            raise RuntimeError('Embedding not consistent with plotting function')

        ax.scatter(*data[::subsample].T, s=s, alpha=alpha, zorder=4)

    def _plot_clusters(self, n=None, s=1, subsample=10, alpha=0.4):

        self._plot_embedding(s=s, subsample=subsample, alpha=alpha)

        data = self.embedding
        ax = plt.gca()
        labels = self.clusters
        unique_labels = np.sort(np.unique(labels))
        colors = [plt.cm.tab20(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            class_member_mask = (labels == k)
            xy = data[class_member_mask, :]
            ax.scatter(*xy[::subsample].T, c=np.array(col).reshape((1,-1)), s=s, alpha=1, zorder=4)      


    def __embedding_algo(self, device, method="umap", **kwargs):
        if method == "umap":
            if device == "gpu":
                from cuml.manifold import UMAP as cuUMAP
                return cuUMAP(**kwargs).fit_transform
            return umap.UMAP(**kwargs).fit_transform
        elif method == "tsne":
            if device == "gpu":
                from cuml.manifold import TSNE as cuTSNE
                if kwargs.get('n_components', 2) != 2:
                    raise ValueError(
                        "GPU t-SNE supports n_components=2 only (cuML limitation); "
                        "use n_components=2 or device='cpu'")
                if 'max_iter' in kwargs:
                    kwargs['n_iter'] = kwargs.pop('max_iter')
                return cuTSNE(**kwargs).fit_transform
            return TSNE(**kwargs).fit_transform
        raise ValueError(f"unknown embedding method '{method}'")

    def __cluster_cpu(self, method="agglomerative", **kwargs):
        if method == "agglomerative":
            # Create a graph capturing local connectivity. Larger number of
            # neighbors will give more homogeneous clusters to the cost of
            # computation time. A very large number of neighbors gives more
            # evenly distributed cluster sizes, but may not impose the local
            # manifold structure of the data
            knn_graph = kneighbors_graph(self.embedding, kwargs['n_neighbors'],
                                         include_self=False)
            model = AgglomerativeClustering(linkage=kwargs['linkage'],
                                            connectivity=knn_graph,
                                            n_clusters=kwargs['n_clusters'])
            return model.fit_predict(self.embedding)
        elif method == "dbscan":
            model = DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_samples'])
        elif method == "hdbscan":
            model = HDBSCAN(min_cluster_size=kwargs['min_cluster_size'],
                            min_samples=kwargs['min_samples'])
        elif method == "kmeans":
            model = KMeans(n_clusters=kwargs['n_clusters'],
                           n_init=kwargs.get('n_init', 10),
                           random_state=kwargs.get('random_state', None))
        else:
            raise ValueError(f"unknown clustering method '{method}'")
        return model.fit_predict(self.embedding)

    def __cluster_gpu(self, method="agglomerative", **kwargs):
        import cupy as cp
        from cuml import cluster as cucluster

        X_gpu = cp.asarray(self.embedding)
        if method == "agglomerative":
            model = cucluster.AgglomerativeClustering(
                n_clusters=kwargs['n_clusters'], connectivity='knn',
                linkage='single', n_neighbors=kwargs['n_neighbors'])
        elif method == "dbscan":
            model = cucluster.DBSCAN(eps=kwargs['eps'],
                                     min_samples=kwargs['min_samples'])
        elif method == "hdbscan":
            model = cucluster.HDBSCAN(min_cluster_size=kwargs['min_cluster_size'],
                                      min_samples=kwargs['min_samples'])
        elif method == "kmeans":
            model = cucluster.KMeans(n_clusters=kwargs['n_clusters'],
                                     n_init=kwargs.get('n_init', 10),
                                     random_state=kwargs.get('random_state', 0))
        else:
            raise ValueError(f"unknown clustering method '{method}'")
        model.fit(X_gpu)
        return cp.asnumpy(model.labels_)


class NEMI(SingleNemi):
    """ Main NEMI workflow

    Args:
        params (dict, optional): clustering and enbedding algorithm parameters.
    """

    def __init__(self, params=None):
        # pipeline parameters
        self.params = copy.deepcopy(default_params)
        self.params.update(params if params is not None else {})
        self.base_id = None

    def run(self, X=None, n=1, assess_overlap=True, output=None, mode='full',
            embeddings=None):
        """ Run the NEMI pipeline

        The pipeline consists of steps:

        - fitting the embedding
        - predicting the clusters,
        - sorting the clusters by descending size

        Args:
            X (:py:class:`~numpy.ndarray`, optional): The data contained in a sparse
                matrix of shape (``n_samples``, ``n_features``). Required unless
                ``mode='cluster'``.
            n (int, optional): Number of iterations to run. Defaults to 1.
            assess_overlap (bool, optional): after building the ensemble, run the
                cross-member co-location + majority vote to set ``self.clusters``.
                Set False to keep the raw ``self.nemi_pack`` (per-member
                embeddings and clusters) for downstream analysis — e.g.
                geographic overlap/entropy in another repo. Ignored when
                ``n == 1``. Defaults to True.
            output (str, optional): if given, write ensemble results to this
                ``.npz`` path: per-member ``embeddings`` (n, N, d) and
                ``member_clusters`` (n, N), plus consensus ``clusters`` (N,) and
                ``embedding`` (N, d) when assess_overlap ran.
            mode (str, optional): ``'full'`` embeds then clusters; ``'embed'``
                stops once the embeddings are written; ``'cluster'`` skips the
                embedding and clusters a saved embeddings file. Defaults to
                ``'full'``.
            embeddings (str, optional): path of the ensemble embeddings ``.npz``,
                written by ``'full'``/``'embed'`` and read by ``'cluster'``.
                Defaults to ``DEFAULT_EMBEDDINGS_PATH`` in the working directory,
                which is overwritten on every run.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
        if embeddings is None:
            embeddings = DEFAULT_EMBEDDINGS_PATH

        if mode == 'cluster':
            self._load_pack(embeddings)
        else:
            if X is None:
                raise ValueError(f"mode='{mode}' requires X")
            self._fit_pack(X, n)
            self._save_embeddings(embeddings)
            if mode == 'embed':
                return

        self._cluster_pack()

        if len(self.nemi_pack) == 1:
            self.embedding = self.nemi_pack[0].embedding
            self.clusters = self.nemi_pack[0].clusters
            if output is not None:
                self.save_outputs(output)
            return

        if assess_overlap:
            self.assess_overlap()
            self.entropy_map()

        if output is not None:
            self._save_ensemble(output)

    def _fit_pack(self, X, n):
        """ Fit the embedding for each of the ``n`` ensemble members. """
        self.nemi_pack = []
        for member in tqdm(np.arange(n)):
            nemi = SingleNemi(params=self.params)
            nemi.fit_embedding(X)
            self.nemi_pack.append(nemi)

    def _cluster_pack(self):
        """ Cluster and size-sort every ensemble member's embedding. """
        for nemi in tqdm(self.nemi_pack):
            nemi.clusters = nemi.sort_clusters(nemi.predict_clusters())

    def _save_embeddings(self, path):
        """ Write the ensemble embeddings (n, N, d) and the params that made them. """
        np.savez(path,
                 embeddings=np.stack([m.embedding for m in self.nemi_pack]),
                 params=json.dumps(self.params))

    def _load_pack(self, path):
        """ Rebuild the ensemble member-by-member from a saved embeddings ``.npz``. """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no saved embeddings at '{path}' — run mode='embed' first, or "
                f"pass the path of an existing embeddings file")
        saved = np.load(path)
        self.nemi_pack = []
        for embedding in saved['embeddings']:
            nemi = SingleNemi(params=self.params)
            nemi.embedding = embedding
            self.nemi_pack.append(nemi)

        embed_params = json.loads(saved['params'].item())
        print(f"Loaded {len(self.nemi_pack)} embedding(s) from {path}")
        print(f"Embedding | device={embed_params['device']} | "
              f"{embed_params['embedding_dict']}")

    def _save_ensemble(self, path):
        """ Save ensemble outputs to a single .npz (for downstream entropy).

        Always: per-member ``embeddings`` (n, N, d) and ``member_clusters``
        (n, N).  Plus consensus ``clusters`` (N,) and ``embedding`` (N, d) when
        the co-location vote has been computed (assess_overlap).
        """
        data = {
            "embeddings": np.stack([m.embedding for m in self.nemi_pack]),
            "member_clusters": np.stack([m.clusters for m in self.nemi_pack]),
        }
        if getattr(self, "clusters", None) is not None:
            data["clusters"] = self.clusters
            data["embedding"] = self.embedding
        if getattr(self, "entropy", None) is not None:
            data["entropy"] = self.entropy
        if getattr(self, "unassigned_frac", None) is not None:
            data["unassigned_frac"] = self.unassigned_frac
        np.savez(path, **data)

    def plot(self, to_plot=None, plot_ensemble=False, **kwargs):

        if plot_ensemble:
            for nemi in self.nemi_pack:
                nemi.plot(to_plot, **kwargs)

        if to_plot == 'clusters':
            super().plot('clusters')

    def assess_overlap(self, base_id=None, max_clusters=None, **kwargs):
        """ Assess the overlap between the clusters.

        Args:
            base_id (int, optional): ensemble member used as the base. Defaults
                to the member with the most clusters, so its cluster count
                covers every other member (required for variable-k methods like
                HDBSCAN).
        """
        if base_id is None:
            base_id = int(np.argmax([_num_clusters(nemi.clusters)
                                     for nemi in self.nemi_pack]))

        self.base_id = base_id
        self.embedding = self.nemi_pack[base_id].embedding

        # list of ensemble members we are comparing to the base
        compare_ids = [i for i in range(len(self.nemi_pack))]
        compare_ids.pop(base_id)

        # identify clusters from the base ensemble member
        base_labels = self.nemi_pack[base_id].clusters

        # (NaN-safe: HDBSCAN/DBSCAN emit -1 noise -> NaN)
        num_clusters = _num_clusters(base_labels)

        if max_clusters is None:
            max_clusters = num_clusters

        sortedOverlap=np.zeros((len(compare_ids)+1, max_clusters, base_labels.shape[0]))*np.nan

        print(num_clusters, max_clusters)
        summaryStats=np.zeros((num_clusters, max_clusters))

        # compile sorted cluster data
        # TODO: add assert statement to make sure that the clusters have been sorted?
        dataVector=[nemi.clusters for id, nemi in enumerate(self.nemi_pack) if id != base_id]

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

        # majority vote, with an "unassigned" bin competing so samples the
        # ensemble mostly left as noise get -1 rather than a spurious cluster 0
        aggOverlaps = np.nansum(sortedOverlap, axis=0)          # (K, N)
        n_members = len(self.nemi_pack)
        unassigned = n_members - aggOverlaps.sum(axis=0)        # (N,) members that abstained (noise)
        augmented = np.vstack([aggOverlaps, unassigned])        # (K+1, N)
        voteOverlaps = np.argmax(augmented, axis=0)
        voteOverlaps[voteOverlaps == aggOverlaps.shape[0]] = -1  # noise bin won -> -1

        # save clusters estimated from the ensemble
        self.clusters = voteOverlaps
        self.overlap_votes = aggOverlaps      # (K, N) aligned per-sample vote counts
        self.n_members = n_members
        self.unassigned_frac = unassigned / n_members           # (N,) graded noise support

    def entropy_map(self):
        """ Per-sample normalized Shannon entropy of the aligned ensemble
        cluster-assignment distribution (from assess_overlap's overlap_votes).

        Members that gave a sample no matched vote (noise labels) are pooled into
        an extra "unassigned" bin that still contributes to the entropy, so
        low-support samples are not treated as confident; the per-sample
        distribution always totals n_members.  Normalized by log(min(K, n_members)),
        the most a sample's votes can spread across real clusters.

        0 = every member agrees (on one cluster, or all agree it is noise);
        higher = members disagree and/or many labeled it noise.  Requires
        assess_overlap to have run.
        """
        n = self.n_members
        votes = self.overlap_votes.astype(float)               # (K, N)
        unassigned = n - votes.sum(axis=0, keepdims=True)       # (1, N) members with no matched vote
        p = np.vstack([votes, unassigned]) / n                 # (K+1, N) distribution, totals 1
        H = -(p * np.where(p > 0, np.log(p), 0.0)).sum(axis=0) # (N,) nats
        k = min(votes.shape[0], n)                             # max spread: K real clusters, capped by members
        self.entropy = H / np.log(k) if k > 1 else H
        return self.entropy
