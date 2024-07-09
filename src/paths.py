import os

# Path to the root directory which contains the src directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_INPUTS_OUTPUTS = os.environ.get(
    "MODEL_INPUTS_OUTPUTS_PATH", os.path.join(ROOT_DIR, "model_inputs_outputs/")
)

# Path to inputs
INPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "inputs")

# Path to training directory inside inputs directory
TRAIN_DIR = os.path.join(INPUT_DIR, "training")

# Path to testing directory inside inputs directory
TEST_DIR = os.path.join(INPUT_DIR, "testing")

# Path to the file containing model weights
MODEL_PATH = os.path.join(MODEL_INPUTS_OUTPUTS, "model", "artifacts", "model.pth")

# Path to the file containing model parameters
MODEL_PARAMS_PATH = os.path.join(
    MODEL_INPUTS_OUTPUTS, "model", "artifacts", "params.json"
)

# Path to the file containing model predictions
PREDICTIONS_PATH = os.path.join(
    MODEL_INPUTS_OUTPUTS, "outputs", "predictions", "predictions.csv"
)
