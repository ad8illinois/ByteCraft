from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
import tabulate

# K-means clustering traditionally requires a euclidean or cosine distance between vectors and not a similarity
# matrix. Popular python libraries provide k-means implementations that expect standard distance metrics conforming
# to euclidean/cosine distances.

# Agglomerative Clustering is much better suited for using a similarity function out-of-the-box
# Although the implementation denotes the use of a distance matrix instead of a similarity matrix,
# they are simply the inverse of each other.
# So we can precompute a distance matrix as 1-(similairty matrix) before passing it into the clustering function
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

def agglomerative_clustering(distance_matrix, n_clusters):
    # precompute a distance matrix as inverse of similarity_matrix before passing it into the clustering function
    # We can switch back to the default linkage "ward" based on performance
    clustering_model = AgglomerativeClustering(metric="precomputed", linkage="average", n_clusters=n_clusters).fit(
        distance_matrix)
    # Cluster labels [0,1]
    return clustering_model.labels_


def naive_bayes_classification(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    model = GaussianNB()
    y_pred = model.fit(x_train, y_train).predict(x_test)
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y_pred, y_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Metrics:")
    data = [['Precision', 'Recall', "F1", "Accuracy"], [precision, recall, f1, accuracy]]
    print(tabulate(data, headers='keys'))

def knn_classification(n_neighbors, X, y):
    # Use the scikit.learn KNN classifier to classify with n_neighbors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    return X_test, pred
