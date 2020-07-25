import pickle
import faiss
import numpy as np

from tqdm import tqdm
from sklearn.manifold import TSNE

def get_emb(model, text, model_type=None):
    """Calculates embedding via chosen model

    Parameters:
    model: embedder model
    text: text to encode
    model_type (str): 'use' for Universal Sentence Encoder
                      'sbert' for SentenceBERT
    """
    if model_type == "use":
        return model(text).numpy()
    else:
        if isinstance(text, str):
            return model.encode([text])
        else:
            return model.encode(text)

def batch_encode(model, texts, batch_size=64, model_type=None):
    embs = []
    iters = len(texts) // batch_size
    it = 0
    for _ in tqdm(range(iters), desc='Calculating embeddings', total=iters):
        embs.extend(get_emb(model, texts[it:it+batch_size], model_type))
        it += batch_size

    #add last not full batch
    if len(texts) % batch_size != 0:
        embs.extend(get_emb(model, texts[it:], model_type))

    return embs


def calculate_embeddings(model, texts, dims=2, model_type=None, batch_size=64, save=False, savepath=None):
    """
    Computes embeddings of texts.
    Parameters
    ----------
    model : the embedder model
    texts : list
            A list of texts corresponding to the embeddings
    dims: int, optional
          A number of embeddings dimensions
    Returns
    -------
    embs : np.array
            The embeddings of input text
    """
    embs = batch_encode(model, texts, batch_size, model_type)
    embs = np.vstack(embs)

    #reduse dims if needed
    if embs.shape[1] > dims:
        print(f'Redusing dimensionality from {embs.shape[1]} to {dims}')
        embs = TSNE(n_components=2, n_jobs=-1).fit_transform(embs)
    
    if save:
        print(f'Saving embeddings to {savepath}')
        with open(savepath, 'wb') as f:
            pickle.dump(embs, f)
    return embs

def apk(actual, predicted, k=10):
    """
    from ml_metrics package
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    from ml_metrics package
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

class Index():
    """Vector-space index with FAISS.

    Builds a vector-space with specified model and performs a search within it.

    Example:
        ``model = hub.KerasLayer(path_to_model)
        data = [
            'What color is chameleon?',
            'When is the festival of colors?',
            'When is the next music festival?',
            'How far is the moon?',
            'How far is the sun?',
            'What happens when the sun goes down?',
            'What we do in the shadows?',
            'What is the meaning of all this?',
            'What is the meaning of Russel\'s paradox?',
            'How are you doing?'
        ]
        index = Index(model, data)
        found_text, found_id = index.search(model("как оно?").numpy(), k=1)``


    Attributes:
        search_space (list): List of texts searched.
        index (): Description of `attr2`.

    """
    def __init__(self, model, texts, embedding_dim=512, model_type=None, batch_size=64):
        """
        Parameters:
        model (obj): Embedder model
        texts (list): List of texts searched
        embedding_dim (int): Output shape of model
                             Default is 512 for USE and SBERT
        model_type (str): name of model
                         'use' for Universal Sentence Encoder
                         'sbert' for SentenceBERT
        """
        self.index = self.build_index(model, texts, embedding_dim, model_type, batch_size=batch_size)
        self.search_space = texts

    def build_index(self, model, texts, embedding_dim, model_type, batch_size, from_saved=None):
        if from_saved is not None:
            with open(from_saved, 'rb') as f:
                vectors = pickle.load(f)
        else:
            vectors = calculate_embeddings(model, texts, embedding_dim, model_type=model_type, batch_size=batch_size)

        faiss.normalize_L2(vectors)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
        index.add_with_ids(vectors, np.array(range(0, vectors.shape[0])))

        return index

    def search(self, query_vector, k=10):
        """Performs a search within the search space

        Returns:
            list of found texts
            list of their ids in the search space
        """
        #normalize to get cosine similarity
        faiss.normalize_L2(query_vector.reshape((1, -1))) 
        #top_k contains:
        #- np.array of cosine similarity between query and top_k responses
        #- np.array of indexes in search space of top_k responses
        top_k = self.index.search(query_vector, k+1)
        return [self.search_space[_id] for _id in top_k[1].tolist()[0][1:]], top_k[1].tolist()[0][1:]

def evaluate(model, input_texts, search_texts, relevant, embedding_dim=512, model_type='use', batch_size=64, search_k=10, metric_k=10):
    """Computes MAP@K

    Parameters:
    model (obj): Embedder model
    input_texts (list): List of test queries
    search_texts (list): List of candidates
    relevant (list): List of indexes of relevant candidates
    embedding_dim (int, optional): Model's output shape. Default is 512 for USE.
    search_k (int, optional): Number of texts to seach
    metric_k (int, optional): K in MAP@K
    """
    embs = calculate_embeddings(model, input_texts, embedding_dim, model_type=model_type, batch_size=batch_size)
    search_index = Index(model, search_texts, embedding_dim, model_type=model_type, batch_size=batch_size)

    true_ids, pred_ids = [], []
    for i, _ in tqdm(enumerate(input_texts), desc='Searching for top k texts for all inputs', total=len(input_texts)):
        _, ids = search_index.search(embs[i].reshape((1, -1))) #search wants 2 dims
        true_ids.append(relevant[i])
        pred_ids.append(ids)

    return mapk(true_ids, pred_ids)