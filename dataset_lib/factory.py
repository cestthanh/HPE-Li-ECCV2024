from . import mmfi


def _dataset_name(config=None, dataset_name=None):
    if dataset_name is not None:
        return dataset_name.lower()
    if isinstance(config, dict):
        return str(config.get("dataset_name", "mmfi")).lower()
    return "mmfi"


def make_dataset(dataset_root, config, dataset_name=None):
    name = _dataset_name(config, dataset_name)
    if name == "mmfi":
        return mmfi.make_dataset(dataset_root, config)
    if name == "wipose":
        from . import wipose

        return wipose.make_dataset(dataset_root, config)
    raise ValueError(f"Unsupported dataset_name={name!r}")


def make_dataloader(dataset, is_training, generator, dataset_name=None, **loader_kwargs):
    name = _dataset_name(None, dataset_name)
    if name == "mmfi":
        return mmfi.make_dataloader(
            dataset,
            is_training=is_training,
            generator=generator,
            **loader_kwargs,
        )
    if name == "wipose":
        from . import wipose

        return wipose.make_dataloader(
            dataset,
            is_training=is_training,
            generator=generator,
            **loader_kwargs,
        )
    raise ValueError(f"Unsupported dataset_name={name!r}")
