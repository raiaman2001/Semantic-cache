"""
Evaluation script for Semantic Cache:
1. Builds a labeled paraphrase dataset (100+ pairs including hard negatives).
2. Sweeps thresholds from 0.50 to 0.98.
3. Calculates Precision, Recall, and False Positive Rate (FPR) with and without cross-encoder reranker.
4. Generates a Markdown table and saves a PNG precision-recall curve.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.embedding import EmbeddingService
from core.rerank import RerankService, verify_lexical_consistency

# 100+ labeled test pairs (a, b, is_paraphrase)
PAIRS = [
    # Positive Paraphrases (General, Tech, FAQ)
    ("How do I reset my password?", "I forgot my password, how can I change it?", 1),
    ("What is RAG?", "Explain retrieval augmented generation", 1),
    ("How to install Python on Windows", "Steps to set up Python on a Windows PC", 1),
    ("What is the capital of France?", "Which city is France's capital?", 1),
    ("How do I enable dark mode?", "How can I turn on the dark theme?", 1),
    ("Best way to learn SQL fast", "Quickest method to study SQL", 1),
    ("How to clear browser cache?", "Steps to remove cached files in chrome", 1),
    ("What causes high CPU usage in Docker?", "Why is my Docker container consuming 100% CPU?", 1),
    ("How do I write unit tests in Python?", "What is the best way to write pytest tests?", 1),
    ("Explain Kubernetes pods", "What is a pod in Kubernetes architecture?", 1),
    ("How to reverse a string in Python?", "Python code to flip a string backwards", 1),
    ("What is gradient descent?", "Explain how gradient descent optimization works", 1),
    ("How to handle merge conflicts in Git?", "Resolving Git merge conflicts step by step", 1),
    ("How to connect to PostgreSQL with asyncpg?", "Using asyncpg to query PostgreSQL in Python", 1),
    ("Difference between SQL and NoSQL databases", "Compare relational databases with NoSQL systems", 1),
    ("How to create a virtual environment in Python?", "Commands to create a venv in python3", 1),
    ("What is an API gateway?", "Explain the role of an API gateway in microservices", 1),
    ("How to optimize SQL query execution plan?", "Tips for speeding up slow relational queries", 1),
    ("How to deploy FastAPI on AWS Lambda?", "Running FastAPI serverless with AWS Lambda", 1),
    ("What is semantic search?", "Explain vector similarity search and semantic retrieval", 1),
    ("How to check disk space in Linux?", "Linux command to view available hard drive storage", 1),
    ("What is an index in SQLite?", "How do indexes work in SQLite database?", 1),
    ("How to authenticate with OAuth2 in FastAPI?", "Implementing OAuth2 token login in FastAPI", 1),
    ("What is rate limiting?", "Explain API rate limiting algorithms like token bucket", 1),
    ("How to exit Vim?", "Command to quit Vim editor", 1),
    ("How to sort a dictionary by value in Python?", "Sorting Python dict items by their values", 1),
    ("What is Redis pub/sub?", "Explain publish subscribe messaging in Redis", 1),
    ("How to generate SSH keys?", "Steps to create a new SSH keypair with ssh-keygen", 1),
    ("Difference between process and thread", "How does a thread differ from an OS process?", 1),
    ("What is cross-entropy loss?", "Explain categorical cross-entropy loss function", 1),
    ("How do I update pip?", "Command to upgrade python pip to latest version", 1),
    ("What is transfer learning?", "Explain pre-trained models and fine-tuning", 1),
    ("How to read a CSV file with Pandas?", "Loading a csv into dataframe using pandas.read_csv", 1),
    ("What is JWT token?", "Explain JSON Web Token structure and verification", 1),
    ("How to configure CORS in FastAPI?", "Enabling cross-origin resource sharing in FastAPI", 1),
    ("What is container orchestration?", "Explain managing container clusters with Kubernetes", 1),
    ("How to set environment variables in Linux?", "Exporting bash environment variables permanently", 1),
    ("What is semantic caching?", "Caching LLM responses using embedding similarity", 1),
    ("How to kill a process on port 8000?", "Finding and terminating process listening on port 8000", 1),
    ("What is a reverse proxy?", "Explain how Nginx acts as a reverse proxy", 1),
    ("How to parse JSON in Python?", "Decoding a JSON string into a Python dict", 1),
    ("What is zero-shot learning?", "Explain classification without fine-tuned training examples", 1),
    ("How to zip a folder in Linux?", "Creating a zip archive from a directory in bash", 1),
    ("What is database sharding?", "Horizontal partitioning of database records", 1),
    ("How to write a Dockerfile?", "Best practices for writing multi-stage Dockerfiles", 1),
    ("What is vector quantization?", "Compressing high-dimensional embeddings for fast ANN search", 1),
    ("How to check git commit history?", "Viewing git log with one-line graph formatting", 1),
    ("What is blue-green deployment?", "Zero-downtime deployment strategy using two environments", 1),
    ("How to find memory leaks in Python?", "Profiling memory consumption with tracemalloc", 1),
    ("What is Cosine Similarity?", "Calculating angle between two normalized vectors", 1),

    # Hard Negatives: Antonyms & Opposite Operations (Crucial test for lexical guard & reranker)
    ("How do I enable notifications?", "How do I disable notifications?", 0),
    ("Add a user to the group", "Remove a user from the group", 0),
    ("How to install Docker", "How to uninstall Docker", 0),
    ("Open a file in Python", "Close a file in Python", 0),
    ("How to start the Redis service", "How to stop the Redis service", 0),
    ("Allow traffic on port 443", "Block traffic on port 443", 0),
    ("Increase thread pool size", "Decrease thread pool size", 0),
    ("Encrypt sensitive database field", "Decrypt sensitive database field", 0),
    ("Enable 2-factor authentication", "Disable 2-factor authentication", 0),
    ("Mount EBS volume to EC2", "Unmount EBS volume from EC2", 0),
    ("Turn on airplane mode", "Turn off airplane mode", 0),
    ("Lock SQLite database during write", "Unlock SQLite database after transaction", 0),
    ("Connect to the remote VPN", "Disconnect from the remote VPN", 0),
    ("Grant admin permissions to role", "Revoke admin permissions from role", 0),
    ("Show hidden files in directory", "Hide dotfiles in directory", 0),
    ("Activate python virtualenv", "Deactivate python virtualenv", 0),
    ("Import module dynamically", "Export module symbols", 0),
    ("Subscribe to Kafka topic", "Unsubscribe from Kafka topic", 0),
    ("Attach network interface", "Detach network interface", 0),
    ("Create AWS IAM policy", "Delete AWS IAM policy", 0),

    # Unrelated Negatives & Topical Distractors
    ("What is machine learning?", "What is the weather in Delhi today?", 0),
    ("Delete my account", "Deactivate my account", 0),
    ("Explain Python lists", "Explain Java interfaces", 0),
    ("How to make coffee?", "How to make pasta carbonara?", 0),
    ("Who won the 2022 World Cup?", "When was the Eiffel Tower built?", 0),
    ("How to train a transformer?", "How to train for a marathon?", 0),
    ("How to repair a flat tire?", "How to install a tire on a bicycle wheel?", 0),
    ("What is Newton's second law?", "Explain Einstein's theory of relativity", 0),
    ("How to create a React component?", "How to write an Angular service?", 0),
    ("Best budget smartphones 2026", "Top luxury cars review", 0),
    ("How to write a resume?", "How to negotiate a salary offer?", 0),
    ("What is photosynthesis?", "Explain cellular respiration in humans", 0),
    ("How to bake sourdough bread?", "How to ferment kimchi at home?", 0),
    ("What is dark matter?", "What is quantum entanglement?", 0),
    ("How to play chess opening?", "How to solve a Rubik's cube?", 0),
    ("Symptoms of acute flu", "Treatment for broken collarbone", 0),
    ("History of the Roman Empire", "History of the Ming Dynasty", 0),
    ("How to invest in index funds?", "How to file income tax return?", 0),
    ("What is WebSockets?", "How does HTTP/2 server push work?", 0),
    ("Best tourist places in Japan", "Best hotels to stay in Paris", 0),
    ("How to plant tomato seeds?", "How to prune apple trees?", 0),
    ("Explain backpropagation algorithm", "Explain Monte Carlo tree search", 0),
    ("How to fix leaky kitchen faucet?", "How to unclog bathroom sink drain?", 0),
    ("What is dark chocolate made of?", "How is milk chocolate manufactured?", 0),
    ("How to setup Prometheus alerts?", "How to configure Grafana dashboards?", 0),
    ("What is a neural network perceptron?", "Explain support vector machine hyperplane", 0),
    ("How to tune hyperparameter with Optuna?", "How to monitor ML training in MLflow?", 0),
    ("How to write regex for email?", "How to validate phone number with regex?", 0),
    ("What is garbage collection in Java?", "How does ARC memory management work in Swift?", 0),
    ("How to learn Spanish fast?", "Tips for learning German vocabulary", 0),
    ("What causes volcanic eruptions?", "How do tsunamis form in oceans?", 0),
]


def run_evaluation():
    print(f"Loaded {len(PAIRS)} test pairs ({sum(p[2] for p in PAIRS)} positives, {sum(1 for p in PAIRS if not p[2])} negatives).")
    embedding_svc = EmbeddingService()
    rerank_svc = RerankService(enabled=True)

    texts_a = [p[0] for p in PAIRS]
    texts_b = [p[1] for p in PAIRS]

    print("Computing embeddings...")
    vecs_a = embedding_svc.encode_sync(texts_a)
    vecs_b = embedding_svc.encode_sync(texts_b)

    sims = [float(np.dot(a, b)) for a, b in zip(vecs_a, vecs_b)]

    print("Evaluating reranker & lexical verification...")
    reranker = rerank_svc.get_model()
    if reranker:
        rr_scores = [float(x) for x in reranker.predict([(p[0], p[1]) for p in PAIRS])]
    else:
        rr_scores = [1.0] * len(PAIRS)

    accepted_gate = [
        verify_lexical_consistency(p[0], p[1]) and (rr_scores[i] >= rerank_svc.min_score)
        for i, p in enumerate(PAIRS)
    ]

    total_pos = sum(p[2] for p in PAIRS)
    total_neg = len(PAIRS) - total_pos

    thresholds = np.arange(0.50, 0.99, 0.04)
    rows = []

    print("\n| Threshold | Precision (raw) | Recall (raw) | FPR (raw) | Precision (+RR) | Recall (+RR) | FPR (+RR) |")
    print("|---|---|---|---|---|---|---|")

    plot_data = {"t": [], "p_raw": [], "r_raw": [], "p_rr": [], "r_rr": []}

    for t in thresholds:
        t_val = round(float(t), 2)
        # Raw cosine
        tp_raw = sum(1 for i, p in enumerate(PAIRS) if sims[i] >= t_val and p[2])
        fp_raw = sum(1 for i, p in enumerate(PAIRS) if sims[i] >= t_val and not p[2])

        # With Reranker + Lexical Guard
        tp_rr = sum(1 for i, p in enumerate(PAIRS) if sims[i] >= t_val and p[2] and accepted_gate[i])
        fp_rr = sum(1 for i, p in enumerate(PAIRS) if sims[i] >= t_val and not p[2] and accepted_gate[i])

        prec_raw = tp_raw / (tp_raw + fp_raw) if (tp_raw + fp_raw) else 1.0
        rec_raw = tp_raw / total_pos if total_pos else 0.0
        fpr_raw = fp_raw / total_neg if total_neg else 0.0

        prec_rr = tp_rr / (tp_rr + fp_rr) if (tp_rr + fp_rr) else 1.0
        rec_rr = tp_rr / total_pos if total_pos else 0.0
        fpr_rr = fp_rr / total_neg if total_neg else 0.0

        print(
            f"| {t_val:.2f} | {prec_raw*100:5.1f}% | {rec_raw*100:5.1f}% | {fpr_raw*100:5.1f}% | "
            f"{prec_rr*100:5.1f}% | {rec_rr*100:5.1f}% | {fpr_rr*100:5.1f}% |"
        )

        plot_data["t"].append(t_val)
        plot_data["p_raw"].append(prec_raw)
        plot_data["r_raw"].append(rec_raw)
        plot_data["p_rr"].append(prec_rr)
        plot_data["r_rr"].append(rec_rr)

    # Plot evaluation curve
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.plot(plot_data["t"], plot_data["p_raw"], "b--", label="Precision (Raw Cosine)")
    plt.plot(plot_data["t"], plot_data["r_raw"], "g--", label="Recall (Raw Cosine)")
    plt.plot(plot_data["t"], plot_data["p_rr"], "b-", linewidth=2.5, label="Precision (+ Reranker Guard)")
    plt.plot(plot_data["t"], plot_data["r_rr"], "g-", linewidth=2.5, label="Recall (+ Reranker Guard)")

    plt.axvline(x=0.88, color="r", linestyle=":", label="Default General Thr (0.88)")
    plt.title("Semantic Cache Threshold vs Precision & Recall", fontsize=13, fontweight="bold")
    plt.xlabel("Cosine Similarity Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    out_png = "eval_curve.png"
    plt.savefig(out_png)
    print(f"\n[OK] Precision/Recall curve saved to {out_png}")


if __name__ == "__main__":
    run_evaluation()
