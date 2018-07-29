from keras.callbacks import Callback
import mlflow


def find_or_create_experiment(name, experiments):
    '''

    :param name: name of the looking experiment
    :param experiments: list of experiment as return by list_experiments()
    :return: Returnan experiment if one is found, a list of experiments if multiple are found
    '''
    found = find_experiment(name, experiments)
    if found is None:
        mlflow.tracking.create_experiment(name)
        return find_experiment(name, mlflow.tracking.list_experiments())
    else:
        return found


def find_experiment(name, experiments):
    '''

    :param name: name of the looking experiment
    :param experiments: list of experiment as return by list_experiments()
    :return: None if nothing is found, an experiment if one is found or a list of experiments if multiple are found.
    '''
    findings = list()
    for experiment in experiments:
        if experiment.name == name:
            findings.append(experiment)

    if len(findings) == 0:
        return None
    elif len(findings) == 1:
        return findings[0]
    else:  # Not sure it's possible tor reach
        return findings


def log_metrics_from_dict(metrics):
    '''

    :param metrics: Dict of metrics from keras History callback
    :return: Nothing
    '''
    for metric in metrics:
        for i in metrics[metric]:
            mlflow.log_metric(metric, i)

