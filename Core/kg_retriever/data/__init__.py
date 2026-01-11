# Dataset module
try:
    from .dataset import KGRetrieverDataset, KGRetrieverDataModule
except ImportError:
    KGRetrieverDataset = None
    KGRetrieverDataModule = None
