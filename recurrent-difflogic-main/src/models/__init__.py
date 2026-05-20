from models.synced_rnn import SyncedRNN
from models.synced_recurrent_difflogic import SyncedRecurrentDiffLogicModel
from models.synced_lstm import SyncedLSTM
from models.synced_gru import SyncedGRU
from models.transformer import Transformer
from models.synced_feedfoward_difflogic import SyncedFeedForwardDiffLogicModel
from models.unsynced_rnn import UnsyncedRNN
from models.unsynced_lstm import UnsyncedLSTM
from models.bidirectional_unsynced_lstm import BidirectionalUnsyncedLSTM
from models.unsynced_gru import UnsyncedGRU
from models.bidirectional_unsynced_gru import BidirectionalUnsyncedGRU
from models.bidirectional_unsynced_rnn import BidirectionalUnsyncedRNN
from models.bidirectional_recurrent_difflogic import BidirectionalRecurrentDifflogic 
from models.unsynced_recurrent_difflogic import UnsyncedRecurrentDifflogic 

MODEL_REGISTRY = {
    "SyncedRNN": SyncedRNN,
    "SyncedLSTM": SyncedLSTM,
    "SyncedGRU": SyncedGRU,
    "Transformer": Transformer,
    "SyncedFeedForwardDiffLogicModel": SyncedFeedForwardDiffLogicModel,
    "SyncedRecurrentDiffLogicModel": SyncedRecurrentDiffLogicModel,
    "UnsyncedRNN": UnsyncedRNN,
    "UnsyncedLSTM": UnsyncedLSTM,
    "BidirectionalUnsyncedLSTM": BidirectionalUnsyncedLSTM,
    "UnsyncedGRU": UnsyncedGRU,
    "BidirectionalUnsyncedGRU": BidirectionalUnsyncedGRU,
    "BidirectionalUnsyncedRNN": BidirectionalUnsyncedRNN,
    "BidirectionalRecurrentDifflogic":BidirectionalRecurrentDifflogic,
    "UnsyncedRecurrentDifflogic":UnsyncedRecurrentDifflogic
}

# Model synchronization information
# True = synced (aligned source/target sequences), False = unsynced (different lengths/structures)
MODEL_SYNC_INFO = {
    "SyncedRNN": True,
    "SyncedLSTM": True,
    "SyncedGRU": True,
    "Transformer": False,  # Encoder-decoder architecture for translation tasks
    "SyncedFeedForwardDiffLogicModel": True,
    "SyncedRecurrentDiffLogicModel": True,
    "UnsyncedRNN": False,  # Encoder-decoder architecture
    "UnsyncedLSTM": False,  # Encoder-decoder architecture
    "BidirectionalUnsyncedLSTM": False,  # Bidirectional encoder-decoder architecture
    "UnsyncedGRU": False,  # Encoder-decoder architecture
    "BidirectionalUnsyncedGRU": False,  # Bidirectional encoder-decoder architecture
    "BidirectionalUnsyncedRNN": False,  # Bidirectional encoder-decoder architecture
    "BidirectionalRecurrentDifflogic":False,
    "UnsyncedRecurrentDifflogic":False
}