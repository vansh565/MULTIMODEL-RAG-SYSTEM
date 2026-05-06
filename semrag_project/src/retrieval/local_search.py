from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def local_search(query, graph, chunks, top_k=5, threshold=0.5):
    query_emb = model.encode([query])

    results = []

    for node in graph.nodes:
        emb = model.encode([node])
        sim = cosine_similarity(query_emb, emb)[0][0]

        if sim > threshold:
            for neighbor in graph[node]:
                results.append(neighbor)

    return list(set(results))[:top_k]