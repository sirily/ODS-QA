import numpy as np
from tqdm import tqdm
from sklearn import cluster, metrics
from sklearn.preprocessing import LabelEncoder
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Turbo256, Category20
from bokeh.models import ColumnDataSource, Jitter

from modules.evaluation import calculate_embeddings

def compute_clusters(vectors, clusters,
                     algorithm='kmeans'):
    # select clustering algorithm
    if algorithm == 'kmeans':
        algorithm = cluster.MiniBatchKMeans(n_clusters=len(set(clusters)))
    elif algorithm == 'dbscan':
        algorithm   = cluster.DBSCAN(eps=1.25, n_jobs=-1)
    elif algorithm == 'optics':
        algorithm = cluster.OPTICS(min_samples=10, eps=10, cluster_method='dbscan', n_jobs=-1)
    elif algorithm == 'birch':
        algorithm = cluster.Birch(n_clusters=len(set(clusters)))
    elif algorithm == 'spectral':
        algorithm = cluster.SpectralClustering(n_clusters=len(set(clusters)), eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1)
    elif algorithm == 'affinity':
        algorithm = cluster.AffinityPropagation(damping=.9, preference=-200)
    else:
        raise NotImplementedError(f"Not implemented for algorithm {algorithm}")

    # predict cluster memberships
    algorithm.fit(vectors)
    if hasattr(algorithm, 'labels_'):
        labels = algorithm.labels_.astype(np.int)
    else:
        labels = algorithm.predict(vectors)

    #transform categorical labels to digits
    if isinstance(clusters[0], str):
        labels_true = LabelEncoder().fit_transform(clusters)
    elif isinstance(clusters[0], (int, np.int)):
        labels_true = clusters

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
        % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(vectors, labels))

    return labels, algorithm

def plot_clusters(vectors, clusters, texts, labels,
                  algorithm_name='Clusterisation',
                  plot_path='',
                  plot_name='clustering.html',
                  plot_size=1000,
                  plot_title="Embeddings clusters"):
    """
    Plots clusters of embeddings .
    Parameters
    ----------
    vectors : list
              A list of embeddings
    clusters : list
               A list of true clusters
    texts : list
            A list of texts corresponding to the embeddings
    clusters : list
               A list of predicted clusters
    algorithm : string, optional
                A name of clustering algorithm
    plot_path : int, optional
                A path to output plot
    plot_name : int, optional
                A name of output plot
    plot_size : int, optional
                A size of output plot
    plot_title : string, optional
                 A title of output plot
    """

    colors = []
    unique_labels = set(labels)
    if len(unique_labels) > 100:
        pal = Turbo256
    else:
        pal = Category20[20]
    if len(unique_labels) <= len(pal):
        palette = pal
    else:
        palette = [pal[each] for each in np.linspace(0, 19, num=len(unique_labels), dtype=np.int)]

    #assign colors from palette
    for cl in labels:
            colors.append(palette[cl])

    source = ColumnDataSource(data=dict(
        x=vectors[:, 0],
        y=vectors[:, 1],
        colors=colors,
        texts=texts,
        clusters=clusters
    ))

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("cluster", "@clusters"),
        ("text", "@texts"),
    ]

    p = figure(output_backend="webgl", title=algorithm_name,
                plot_width=plot_size, plot_height=plot_size, tooltips=TOOLTIPS)

    p.scatter(x={'field': 'x', 'transform': Jitter(width=0.4)}, y={'field': 'y', 'transform': Jitter(width=0.4)}, 
                 color='colors', alpha=0.5, source=source)

    output_file(plot_path+plot_name, title=plot_title)

    show(p)

def make_clusters(model, texts, labels, 
            algorithm='kmeans',
            dims=2,
            model_type='use',
            plot_path='',
            plot_name='clustering.html',
            plot_size=1000,
            plot_title="Embeddings clusters"):
    """
    Computes embeddings and plot clusters of them.
    Parameters
    ----------
    model : the embedder model
    texts : list
            A list of texts corresponding to the embeddings
    clusters : list
               A list of true clusters
    algorithm : string, optional
                A name of clustering algorithm
    plot_path : int, optional
                A path to output plot
    plot_name : int, optional
                A name of output plot
    plot_size : int, optional
                A size of output plot
    plot_title : string, optional
                 A title of output plot
    """
    embs = calculate_embeddings(model, texts, dims, model_type=model_type)
    print('Computing clusters')
    pred_labels, alg = compute_clusters(embs, labels, algorithm)
    print('Plotting')
    plot_clusters(embs, labels, texts, pred_labels, 
                  algorithm_name=alg.__class__.__name__,
                  plot_path=plot_path, plot_name=plot_name,
                  plot_size=plot_size, plot_title=plot_title)
    print(f'Look for {plot_title} at {plot_path}')