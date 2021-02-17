from omegaconf import OmegaConf


class Config:
    def __init__(self):
        self._config = None

    def __getattr__(self, name):
        return getattr(self._config, name)

    def set_config(self, cfg):
        self._config = cfg

    def to_dict(self):
        return OmegaConf.to_container(self._config)


config = Config()
