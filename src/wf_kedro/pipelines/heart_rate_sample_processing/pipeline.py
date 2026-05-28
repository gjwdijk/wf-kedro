"""
This is a boilerplate pipeline 'heart_rate_sample_processing'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import process_parquet_partitions, create_keras_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=process_parquet_partitions,
            inputs="raw_heart_rate_data",
            outputs="interim_heart_rate_dataframe",
            name="clean_parquet_node",
        ),
        node(
            func=create_keras_dataset,
            inputs="interim_heart_rate_dataframe",
            # We voegen hier 'copy_mode="assign"' toe om TensorFlow-fouten te voorkomen:
            outputs=dict(value="keras_heart_rate_dataset", copy_mode="assign"),
            name="convert_to_keras_dataset_node",
        ),
    ])
