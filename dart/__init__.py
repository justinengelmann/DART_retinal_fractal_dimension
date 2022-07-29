import logging

from .helpers import load_model
from .inference import get_inference_pipeline, get_model_and_processing

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
