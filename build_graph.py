import networkx
from scipy import stats
import mallet

def build_interaction_graph(mallet_model, threshold):
    g = networkx.Graph()
    topic_matrix = model.theta
    for i in xrange(topic_matrix.shape[1]):
        print i
        for j in xrange(i+1, topic_matrix.shape[1]):
            divergence_ij = stats.entropy(topic_matrix[:,i], topic_matrix[:,j])
            divergence_ji = stats.entropy(topic_matrix[:,j], topic_matrix[:,i])
            inverse_divergence_sym = float(1/(divergence_ij + divergence_ji))
            if inverse_divergence_sym >= threshold:
                g.add_node(j, label=', '.join(mallet_model.list_topic(j, 3)))
                g.add_edge(i, j, weight=inverse_divergence_sym)
            else:
                g.add_node(i)

    for i in xrange(topic_matrix.shape[1]):
        if len(g.edge[i]) == 0:
            g.remove_node(i)
    for i in xrange(topic_matrix.shape[1]):
        if i in g.node and len(g.node[i]) == 0 and len(g.edge[i]) != 0:
            print i
            g.add_node(i, label=', '.join(mallet_model.list_topic(i, 3)))
    return g

if __name__ == "__main__":
    model = mallet.MalletLDA('./Data/mallet_files/doc_topics.tsv',
                             './Data/mallet_files/topic_counts.tsv')
    g = build_interaction_graph(model, .33)
    networkx.write_graphml(g, "./data/mallet_files/interaction_graph.graphml")
