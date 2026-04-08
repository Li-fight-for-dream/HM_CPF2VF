from src.data_dealer.dataset_builder_for_fundus import make_dataloaders


def build_dataset(cfg):
    return make_dataloaders(cfg)
