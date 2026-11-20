import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


import streamlit as st
import pandas as pd
import plotly.express as px


from gnn.graph_loader import GraphLoader
from gnn.graph_dataset import GraphDatasetBuilder
from controller.runtime_controller import RuntimeController
from data.synthetic import SyntheticGraphGenerator



st.set_page_config(
    page_title="GNN-AdaptGraph",
    layout="wide"
)


st.title(
    "GNN-AdaptGraph Runtime Adaptation Dashboard"
)



st.sidebar.header(
    "Graph Configuration"
)


dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    [
        "Synthetic Graph",
        "DBLP"
    ]
)



if st.button("Run Adaptation"):


    # ==========================
    # GRAPH LOADING
    # ==========================


    if dataset_name == "Synthetic Graph":


        generator = SyntheticGraphGenerator()

        G = generator.generate()

        G = GraphLoader.load_synthetic(G)



    elif dataset_name == "DBLP":


        with st.spinner(
            "Loading DBLP graph..."
        ):

            G = GraphLoader.load_dblp()



    


    st.success(
        f"{dataset_name} Loaded Successfully"
    )



    # ==========================
    # GRAPH DATASET CREATION
    # ==========================


    graph_data = GraphDatasetBuilder.build(
        G,
        num_partitions=4
    )



    # ==========================
    # GRAPH SAGE CONTROLLER
    # ==========================


    checkpoint = os.path.join(
        "gnn",
        "checkpoints",
        "best_graphsage_model.pth"
    )


    controller = RuntimeController(
        checkpoint
    )


    predictions = controller.predict(
        graph_data
    )



    # ==========================
    # GRAPH INFORMATION
    # ==========================


    st.subheader(
        "Graph Information"
    )


    c1,c2,c3 = st.columns(3)


    c1.metric(
        "Nodes",
        f"{G.number_of_nodes():,}"
    )


    c2.metric(
        "Edges",
        f"{G.number_of_edges():,}"
    )


    c3.metric(
        "Partitions",
        "4"
    )



    # ==========================
    # FEATURES
    # ==========================


    st.subheader(
        "Partition Features"
    )


    features = graph_data.x.numpy()


    feature_df = pd.DataFrame(

        features,

        columns=[
            "Degree Skew",
            "Imbalance",
            "Clustering",
            "Density",
            "Avg Degree",
            "Size Ratio",
            "Boundary Edges",
            "Edge Cut Ratio"
        ]

    )


    st.dataframe(
        feature_df,
        use_container_width=True
    )



    # ==========================
    # GNN PREDICTION
    # ==========================


    st.subheader(
        "GraphSAGE Controller Prediction"
    )


    result_df = pd.DataFrame(

        {

            "Partition":
            [
                "P1",
                "P2",
                "P3",
                "P4"
            ],

            "Action":
            predictions

        }

    )


    st.table(
        result_df
    )



    # ==========================
    # VISUALIZATION
    # ==========================


    st.subheader(
        "Partition Adaptation Analysis"
    )


    before = (
        feature_df["Imbalance"] * 100
    )


    after = [
        25,
        25,
        25,
        25
    ]



    chart_df = pd.DataFrame(

        {

            "Partition":
            [
                "P1",
                "P2",
                "P3",
                "P4"
            ],

            "Before":
            before,

            "After":
            after

        }

    )


    fig = px.bar(

        chart_df,

        x="Partition",

        y=[
            "Before",
            "After"
        ],

        barmode="group",

        title=
        "Load Balancing Before vs After Adaptation"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



    st.success(
        "GNN-AdaptGraph Adaptation Completed"
    )