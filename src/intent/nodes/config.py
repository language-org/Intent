# author: steeve laquitaine

import yaml
import mlflow


def load_parameters(proj_path):
    """load parameters from conf/parameters.yml

    Args:
        proj_path (str): [description]

    Returns:
        dict: [description]
    """
    with open(
        proj_path + "/conf/base/parameters.yml"
    ) as file:
        prms = yaml.load(file)
    return prms


def config_mflow(prms: dict):
    """Setup mlflow experiment and run tracking

    Args:
        prms (dict): [description]
    """
    mlflow.set_tracking_uri(prms["mlflow"]["tracking_uri"])
    client = mlflow.tracking.MlflowClient(
        tracking_uri=prms["mlflow"]["tracking_uri"]
    )

    # create or get an existing experiment
    if (
        client.get_experiment_by_name(
            prms["mlflow"]["experiment_name"]
        )
        is None
    ):
        exp_id = client.create_experiment(
            prms["mlflow"]["experiment_name"]
        )
        exp = client.get_experiment(exp_id)
    else:
        exp = client.get_experiment_by_name(
            prms["mlflow"]["experiment_name"]
        )
    client.set_experiment_tag(
        exp.experiment_id, "topic", "intent"
    )

    # create run
    run = client.create_run(exp.experiment_id)
    client.set_tag(
        run.info.run_id, "model", "hierarchical clustering"
    )

    # print experiment info stored on server
    print(f"Experiment:")
    print(f"- name: {exp.name}")
    print(f"- tags: {exp.tags}")
    print(f"- id: {exp.experiment_id}")
    print(f"- tracking uri: {mlflow.get_tracking_uri()}")
    print(f"- artifact location: {exp.artifact_location}")
    print(f"- lifecycle stage: {exp.lifecycle_stage}")

    print(f"Run:")
    print(f"- id: {run.info.run_id}")
    print(f"- artifact uri: {run.info.artifact_uri}")
    print(f"- status: {run.info.status}")
