import pandas as pd
import wandb
from pathlib import Path


# sweep is specified by <entity/project-name>/<sweep-id>
def extract_sweep_data(identifier: str) -> tuple[str, pd.DataFrame]:
    """get the summary and config data from a wandb sweep

    Parameters
    ----------
    identifier : str
        sweep is specified by <entity/project-name/sweep-id>

    Returns
    -------
    str, pd.DataFrame
        sweep name and data
    """
    api = wandb.Api()
    sweep = api.sweep(identifier)
    sweep_name = sweep.name.removesuffix("_sweep")
    print(sweep_name)
    runs = sweep.runs

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We ommit all keys starting with _ and all values that are dictionaries.
        summary_list.append(
            {
                k: v
                for k, v in run.summary._json_dict.items()
                if not ((k.startswith("_")) or isinstance(v, dict))
            }
        )

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not ((k.startswith("_")))}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    summary_df = pd.DataFrame(summary_list)
    config_df = pd.DataFrame(config_list)
    name_df = pd.DataFrame(data={"name": name_list})

    data_df = pd.concat([name_df, summary_df, config_df], axis=1)

    return sweep_name, data_df


# sweep is specified by <entity/project-name>/<sweep-id>
def extract_run_data(identifier: str) -> tuple[str, pd.DataFrame]:
    """get the summary and config data from a wandb run

    Parameters
    ----------
    identifier : str
        sweep is specified by <entity/project-name/run-id>

    Returns
    -------
    str, pd.DataFrame
        sweep name and data
    """
    api = wandb.Api()
    run = api.run(identifier)
    # .summary contains the output keys/values for metrics like accuracy.
    #  We ommit all keys starting with _ and all values that are dictionaries.
    summary = {
        k: v
        for k, v in run.summary._json_dict.items()
        if not ((k.startswith("_")) or isinstance(v, dict))
    }

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config = {k: v for k, v in run.config.items() if not ((k.startswith("_")))}

    summary_df = pd.DataFrame([summary])
    config_df = pd.DataFrame([config])
    name_df = pd.DataFrame(data={"name": [run.name]})

    data_df = pd.concat([name_df, summary_df, config_df], axis=1)

    return run.name, data_df


def extract_sweep_artifacts(identifier: str, art_names: list[str]) -> pd.DataFrame:
    """get the specified artifacts from a wandb sweep

    Parameters
    ----------
    identifier : str
        sweep is specified by <entity/project-name/sweep-id>

    art_names : list[str]
        list of artifact names to extract (`nus`, `ar_params`, `epsilon`)

    Returns
    -------
    str, pd.DataFrame
        sweep name and artifact data
    """
    api = wandb.Api()
    sweep = api.sweep(identifier)
    sweep_name = sweep.name.removesuffix("_sweep")
    print(sweep_name)
    runs = sweep.runs

    arts = []
    for run in runs:
        _arts = {"name": run.name}
        for art_name in art_names:
            downloaded_artifact = api.artifact(
                f"{sweep.entity}/{sweep.project}/run-{run.id}-{art_name}:v0"
            ).get(art_name)
            _arts[art_name] = downloaded_artifact.get_dataframe()
        arts.append(_arts)

    return pd.DataFrame(arts)


def extract_sweep_savepoints(identifier: str, flag: str = "latest") -> dict[str, Path]:
    """get the savepoints from a wandb sweep

    Parameters
    ----------
    identifier : str
        sweep is specified by <entity/project-name/sweep-id>
    flag : str, optional
        flag of the model to get (`v1`,`best`,`latest`), by default "latest"

    Returns
    -------
    dict[str, Path]
        run names and model checkpoint paths
    """
    api = wandb.Api()
    sweep = api.sweep(identifier)
    sweep_name = sweep.name.removesuffix("_sweep")
    print(sweep_name)
    runs = sweep.runs
    model_ckpt_paths = {}
    for run in runs:
        base_path = identifier.rsplit("/", 1)[0]
        model_ckpt = api.artifact(f"{base_path}/model-{run.id}:{flag}")
        model_ckpt_path = Path(model_ckpt.download() + r"\model.ckpt")
        model_ckpt_paths.update({run.name: model_ckpt_path})
    return model_ckpt_paths


def extract_run_savepoint(identifier: str, flag: str = "latest") -> tuple[str, Path]:
    """get the savepoint from a wandb run

    Parameters
    ----------
    identifier : str
        run is specified by <entity/project-name/run-id>
    flag : str, optional
        flag of the model to get (`v1`,`best`,`latest`), by default "latest"

    Returns
    -------
    dict[str, Path]
        run name and checkpoint path
    """
    api = wandb.Api()
    run = api.run(identifier)
    base_path, run_id = identifier.rsplit("/", 1)
    model_ckpt = api.artifact(f"{base_path}/model-{run_id}:{flag}")

    return run.name, Path(model_ckpt.download() + r"\model.ckpt")
