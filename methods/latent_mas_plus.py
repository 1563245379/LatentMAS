from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import torch
import argparse

try:
    from vllm import SamplingParams
except:
    print ("vLLM not installed, may be fine unless vLLM use required.")
    
import pdb
from tqdm import tqdm

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASPlus:
    def __init__(
        self,
        model: ModelWrapper,
        agents: Optional[List[str]] = None,
        agent_kwargs: Optional[Dict] = None,
        max_iterations: int = 5,
        max_retries: int = 3,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: Optional[int] = None,
        use_vllm: bool = False,
    ):
        self.model = model
        self.agents = agents if agents is not None else default_agents()
        self.agent_kwargs = agent_kwargs if agent_kwargs is not None else {}
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.use_vllm = use_vllm

        