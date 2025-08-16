### PrEvelOp Dashboard
### Interactive visualization and clustering analysis
### Author: Kaspar Bunne

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

from prevelop.data import generate_toy_dataset
from prevelop.preparation import preprocessing
from prevelop.clustering import (
    gower_distance,
    agglomerative_clustering,
    kmedoids_clustering,
    hdbscan_clustering,
)
from prevelop.evaluation import evaluate_clustering


### page config
st.set_page_config(page_title="PrEvelOp", layout="wide")


### sidebar
st.sidebar.title("PrEvelOp")
st.sidebar.caption("Clustering Framework for Mixed-Type Manufacturing Data")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Exploration", "Clustering", "Evaluation"],
)

st.sidebar.divider()
st.sidebar.markdown("**Dataset**")
n_samples = st.sidebar.slider("Samples", 100, 500, 300, step=50)


### cached data loading and preprocessing
@st.cache_data
def load_data(n):
    return generate_toy_dataset(n_samples=n)


@st.cache_data
def preprocess_data(_data, num_cols, cat_cols):
    return preprocessing(_data, num_cols, cat_cols)


@st.cache_data
def compute_distances(_data_prep):
    return gower_distance(_data_prep)


@st.cache_data
def compute_tsne(_data_prep, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(_data_prep)


data, num_columns, cat_columns = load_data(n_samples)
data_preprocessed = preprocess_data(data, num_columns, cat_columns)


### ── Overview ──────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", len(data))
    col2.metric("Features", len(data.columns))
    col3.metric("Numerical", len(num_columns))
    col4.metric("Categorical", len(cat_columns))

    st.subheader("Data Preview")
    st.dataframe(data, use_container_width=True, height=400)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Numerical Summary")
        st.dataframe(
            data[num_columns].describe().round(2),
            use_container_width=True,
        )
    with c2:
        st.subheader("Categorical Summary")
        for col in cat_columns:
            with st.expander(col):
                counts = data[col].value_counts().reset_index()
                counts.columns = [col, 'count']
                st.dataframe(counts, use_container_width=True, hide_index=True)


### ── Exploration ───────────────────────────────────────────────────────────────
elif page == "Exploration":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distributions", "Box Plots", "Correlation", "Dimensionality Reduction"]
    )

    with tab1:
        col_sel = st.selectbox("Feature", num_columns, key="dist_feature")
        fig = px.histogram(
            data, x=col_sel, nbins=30, marginal="violin",
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        ### categorical distribution
        cat_sel = st.selectbox("Categorical Feature", cat_columns, key="cat_dist")
        counts = data[cat_sel].value_counts().reset_index()
        counts.columns = [cat_sel, 'count']
        fig = px.bar(counts, x=cat_sel, y='count', color=cat_sel)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=num_columns,
        )
        colors = px.colors.qualitative.Set2
        for i, col in enumerate(num_columns):
            r, c = i // 3 + 1, i % 3 + 1
            fig.add_trace(
                go.Box(y=data[col], name=col, marker_color=colors[i % len(colors)]),
                row=r, col=c,
            )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
        corr = data[num_columns].corr(method=method)
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto", zmin=-1, zmax=1,
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        perplexity = c1.slider("Perplexity", 5, 50, 30)
        color_by = c2.selectbox("Color by", cat_columns)

        embedding = compute_tsne(data_preprocessed, perplexity)
        fig = px.scatter(
            x=embedding[:, 0], y=embedding[:, 1],
            color=data[color_by].values,
            labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": color_by},
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)


### ── Clustering ────────────────────────────────────────────────────────────────
elif page == "Clustering":
    st.title("Clustering Analysis")

    col_settings, col_results = st.columns([1, 3])

    with col_settings:
        st.subheader("Settings")
        algorithm = st.selectbox(
            "Algorithm",
            ["Agglomerative", "K-Medoids", "HDBSCAN"],
        )

        if algorithm in ("Agglomerative", "K-Medoids"):
            n_clusters = st.slider("Clusters", 2, 15, 5)
        else:
            min_cluster_size = st.slider("Min cluster size", 3, 50, 5)
            min_samples_val = st.slider("Min samples", 1, 30, 3)

        run_btn = st.button("Run Clustering", type="primary", use_container_width=True)

    with col_results:
        if run_btn:
            with st.spinner("Computing Gower distance matrix..."):
                dist = compute_distances(data_preprocessed)

            with st.spinner(f"Running {algorithm}..."):
                if algorithm == "Agglomerative":
                    labels = agglomerative_clustering(dist, n_clusters)
                elif algorithm == "K-Medoids":
                    labels = kmedoids_clustering(dist, n_clusters)
                else:
                    labels, _ = hdbscan_clustering(dist, min_cluster_size, min_samples_val)

            st.session_state["labels"] = labels

            ### t-SNE cluster visualization
            embedding = compute_tsne(data_preprocessed)
            fig = px.scatter(
                x=embedding[:, 0], y=embedding[:, 1],
                color=labels.astype(str),
                labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": "Cluster"},
                title=f"{algorithm} Clustering Results",
            )
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

            ### cluster size distribution
            unique, counts = np.unique(labels, return_counts=True)
            cluster_df = pd.DataFrame({"Cluster": unique.astype(str), "Size": counts})
            fig = px.bar(
                cluster_df, x="Cluster", y="Size", color="Cluster",
                title="Cluster Sizes",
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif "labels" in st.session_state:
            st.info("Previous results available. Click 'Run Clustering' to update.")
        else:
            st.info("Select algorithm and parameters, then click 'Run Clustering'.")


### ── Evaluation ────────────────────────────────────────────────────────────────
elif page == "Evaluation":
    st.title("Clustering Evaluation")

    if "labels" not in st.session_state:
        st.info("Run clustering first on the Clustering page.")
    else:
        labels = st.session_state["labels"]

        ### filter noise points for evaluation
        valid = labels >= 0
        n_noise = (~valid).sum()
        if n_noise > 0:
            st.warning(f"{n_noise} noise points (label=-1) excluded from metrics.")

        data_eval = data_preprocessed.values[valid]
        labels_eval = labels[valid]

        if len(set(labels_eval)) < 2:
            st.error("Need at least 2 clusters for evaluation metrics.")
        else:
            ### compute metrics
            scores = evaluate_clustering(data_eval, labels_eval)

            st.subheader("Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Silhouette Score", f"{scores['silhouette-score']:.3f}")
            c2.metric("Dunn Index", f"{scores['dunn-index']:.3f}")
            c3.metric("Davies-Bouldin", f"{scores['dbi']:.3f}")
            c4.metric("Calinski-Harabasz", f"{scores['ch-score']:.1f}")

            st.divider()

            ### per-cluster statistics
            st.subheader("Cluster Statistics")
            cluster_stats = []
            for label in sorted(set(labels_eval)):
                mask = labels_eval == label
                cluster_data = data[valid].iloc[mask]
                stats = {"Cluster": int(label), "Size": int(mask.sum())}
                for col in num_columns:
                    stats[f"{col} (mean)"] = round(cluster_data[col].mean(), 2)
                cluster_stats.append(stats)

            stats_df = pd.DataFrame(cluster_stats).set_index("Cluster")
            st.dataframe(stats_df, use_container_width=True)

            ### cluster composition for categorical features
            st.subheader("Cluster Composition")
            cat_sel = st.selectbox("Categorical Feature", cat_columns)
            composition = []
            for label in sorted(set(labels_eval)):
                mask = labels_eval == label
                cluster_data = data[valid].iloc[mask]
                for value, count in cluster_data[cat_sel].value_counts().items():
                    composition.append({
                        "Cluster": str(label),
                        cat_sel: value,
                        "Count": count,
                    })
            comp_df = pd.DataFrame(composition)
            fig = px.bar(
                comp_df, x="Cluster", y="Count", color=cat_sel,
                title=f"{cat_sel} Distribution per Cluster",
                barmode="stack",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
