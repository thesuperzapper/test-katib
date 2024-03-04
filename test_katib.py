# reference:
#  - https://github.com/kubeflow/katib/blob/master/examples/v1beta1/sdk/cmaes-and-resume-policies.ipynb

from kubeflow.katib import KatibClient
from kubeflow.katib import V1beta1AlgorithmSpec
from kubeflow.katib import V1beta1Experiment
from kubeflow.katib import V1beta1ExperimentSpec
from kubeflow.katib import V1beta1FeasibleSpace
from kubeflow.katib import V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1ParameterSpec
from kubeflow.katib import V1beta1TrialParameterSpec
from kubeflow.katib import V1beta1TrialTemplate
from kubernetes.client import V1ObjectMeta

###############################################################
# CONFIGS
###############################################################
CONFIG_NAMESPACE = "team-1"
CONFIG_EXPERIMENT_NAME = "cmaes-example"


###############################################################
# Define the Experiment
###############################################################
metadata = V1ObjectMeta(name=CONFIG_EXPERIMENT_NAME, namespace=CONFIG_NAMESPACE)

# Algorithm specification.
algorithm_spec = V1beta1AlgorithmSpec(algorithm_name="cmaes")

# Objective specification.
objective_spec = V1beta1ObjectiveSpec(
    type="minimize",
    goal=0.001,
    objective_metric_name="loss",
)

# Experiment search space. In this example we tune learning rate, number of layer and optimizer.
parameters = [
    V1beta1ParameterSpec(
        name="lr",
        parameter_type="double",
        feasible_space=V1beta1FeasibleSpace(min="0.01", max="0.06"),
    ),
    V1beta1ParameterSpec(
        name="momentum",
        parameter_type="double",
        feasible_space=V1beta1FeasibleSpace(min="0.5", max="0.9"),
    ),
]

# JSON template specification for the Trial's Worker Kubernetes Job.
trial_spec = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "spec": {
        "template": {
            "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
            "spec": {
                "containers": [
                    {
                        "name": "training-container",
                        "image": "docker.io/kubeflowkatib/pytorch-mnist-cpu:v0.14.0",
                        "command": [
                            "python3",
                            "/opt/pytorch-mnist/mnist.py",
                            "--epochs=1",
                            "--batch-size=64",
                            "--lr=${trialParameters.learningRate}",
                            "--momentum=${trialParameters.momentum}",
                        ],
                    }
                ],
                "restartPolicy": "Never",
            },
        }
    },
}

# Configure parameters for the Trial template.
trial_template = V1beta1TrialTemplate(
    primary_container_name="training-container",
    trial_parameters=[
        V1beta1TrialParameterSpec(
            name="learningRate",
            description="Learning rate for the training model",
            reference="lr",
        ),
        V1beta1TrialParameterSpec(
            name="momentum",
            description="Momentum for the training model",
            reference="momentum",
        ),
    ],
    trial_spec=trial_spec,
)

# Experiment object.
experiment = V1beta1Experiment(
    api_version="kubeflow.org/v1beta1",
    kind="Experiment",
    metadata=metadata,
    spec=V1beta1ExperimentSpec(
        max_trial_count=3,
        parallel_trial_count=2,
        max_failed_trial_count=1,
        algorithm=algorithm_spec,
        objective=objective_spec,
        parameters=parameters,
        trial_template=trial_template,
    ),
)

###############################################################
# Create the Experiment
###############################################################

# Initialize the Katib client
katib_client = KatibClient()

# Create the Experiment
# katib_client.create_experiment(experiment, namespace=CONFIG_NAMESPACE)

# Get the Experiment
experiment_return = katib_client.get_experiment(
    name=CONFIG_EXPERIMENT_NAME, namespace=CONFIG_NAMESPACE
)
print(experiment_return)
