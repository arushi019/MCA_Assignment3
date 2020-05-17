from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_gt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    return [(int(l.split()[0]), int(l.split()[2])) for l in lines]

def generate_ground_truth(vec_docs,n=20):
    docs = vec_docs.toarray()
    docs_t = docs.T
    sim = np.dot(docs,docs_t)
    num_docs = sim.shape[0]
    ground = []
    for i in range(num_docs):
        similarity = sim[i]
        temp = []
        for j in range(num_docs):
            temp.append((j,similarity[j]))
        temp = sorted(temp,key=lambda k:k[1])
        selected = temp[len(temp)-n:]
        for j in selected:
            ground.append((i,j[0]))
    print(ground)
    return ground
            

def get_top_terms(ground,vec_docs,num_docs,n=100):
    terms = {}
    docs = vec_docs.toarray()
    for i in range(num_docs):
        if i not in ground:
            continue
        doc = docs[i]
        for j in range(len(doc)):
            if j not in terms.keys():
                terms[j] = doc[j]
            else:
                if terms[j]<doc[j]:
                    terms[j] = doc[j]
    terms = sorted(terms.items(),key = lambda k:k[1])
    top_terms = terms[len(terms)-n:]
    return top_terms
def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    ground = load_gt('data/med/MED.REL')
    #ground has tuples of (query, doc_number)
    num_queries = len(sim[0])
    num_docs = len(sim)
    alpha = 0.9
    beta = 0.8
    #sim will have dimensions num_docs*num_queries
    for i in range(num_queries):
        perceived_relevant = []
        doc_query_sim = []
        for j in range(num_docs):
            doc_query_sim.append((j,sim[j][i]))
        doc_query_sim = sorted(doc_query_sim,key = lambda k:k[1])
        for j in doc_query_sim[num_docs-n:]:
            perceived_relevant.append(j[0])
        ground_relevant = []
        for j in ground:
            if j[0] == i+1:
                ground_relevant.append(j[1]-1)
        query_vector = vec_queries[i].toarray()[0]
        relevant = []
        non_relevant = []
        for j in perceived_relevant:
            if j in ground_relevant:
                relevant.append(j)
            else:
                non_relevant.append(j)
        sum_relevant = np.zeros(query_vector.shape)
        sum_non_relevant = np.zeros(query_vector.shape)
        for j in relevant:
            doc_vec = vec_docs[j].toarray()[0]
            sum_relevant += doc_vec
        if len(relevant)>0:
            sum_relevant = sum_relevant*alpha/len(relevant)
        for j in non_relevant:
            doc_vec = vec_docs[j].toarray()[0]
            sum_non_relevant += doc_vec
        if len(non_relevant)>0:
            sum_non_relevant = sum_non_relevant*beta/len(non_relevant)
        query_vector += sum_relevant - sum_non_relevant
        vec_queries[i] = sparse.csr_matrix(query_vector)
    rf_sim = cosine_similarity(vec_docs,vec_queries)
    #rf_sim = sim
    #print(rf_sim,sim)
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    ground = load_gt('data/med/MED.REL')
    num_queries = len(sim[0])
    num_docs = len(sim)
    for k in range(3):
        for i in range(num_queries):
            ground_relevant = []
            for j in ground:
                if j[0] == i+1:
                    ground_relevant.append(j[1]-1)
            top_terms = get_top_terms(ground_relevant,vec_docs,num_docs)
            query = vec_queries[i].toarray()[0]
            for j in top_terms:
                query[j[0]] = j[1]
            vec_queries[i] = sparse.csr_matrix(query)
    rf_sim = cosine_similarity(vec_docs,vec_queries)
    #print(rf_sim,sim)
    return rf_sim
