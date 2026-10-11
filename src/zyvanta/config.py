"""Single source of truth for Zyvanta model and training configuration.

Every hyperparameter lives here so the architecture, training run, and
generation defaults are auditable from one file. The model defaults must stay
in sync with ``src/zyvanta/model.py``.
"""

# --- Model architecture (must match ZyvantaLM defaults in model.py) ---
VOCAB_SIZE = 16000
D_MODEL = 320
N_LAYERS = 6
N_HEADS = 5
D_FF = 1280
MAX_SEQ_LEN = 512
NORM_EPS = 1e-5

# --- Special token ids (must match ByteBPETokenizer in tokenizer.py) ---
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258

# --- Training hyperparameters (conservative defaults for one Tesla T4) ---
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
MAX_STEPS = 1000
WARMUP_STEPS = 100
GRADIENT_CLIP = 1.0
SEED = 42
LOG_INTERVAL = 10
SAVE_INTERVAL = 500
CHECKPOINT_DIR = "checkpoints"

# --- Data ---
# Directory of plain .txt training corpus files. Empty/missing -> clear error.
DATA_DIR = "data"

# Max UTF-8 bytes consumed when fitting the tokenizer. Keeps full-corpus BPE
# tractable on large corpora; pass None to use the whole corpus.
TOKENIZER_SAMPLE_LIMIT = 4_000_000

# --- Generation defaults ---
GENERATE_MAX_NEW_TOKEN_TOKENS = 100
GENERATE_TEMPERATURE = 0.8