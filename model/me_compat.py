def sparse_tensor(ME, features, coordinates=None, coordinate_map_key=None, coordinate_manager=None, device=None):
    if device is not None:
        features = features.to(device)
        if coordinates is not None:
            coordinates = coordinates.to(device)

    if coordinates is not None:
        try:
            return ME.SparseTensor(features, coordinates=coordinates, device=device)
        except TypeError:
            try:
                return ME.SparseTensor(features, coordinates=coordinates)
            except TypeError:
                return ME.SparseTensor(features, coords=coordinates)

    try:
        return ME.SparseTensor(
            features,
            coordinate_map_key=coordinate_map_key,
            coordinate_manager=coordinate_manager,
            device=device,
        )
    except TypeError:
        try:
            return ME.SparseTensor(
                features,
                coordinate_map_key=coordinate_map_key,
                coordinate_manager=coordinate_manager,
            )
        except TypeError:
            return ME.SparseTensor(
                features,
                coords_key=coordinate_map_key,
                coords_manager=coordinate_manager,
            )


def coordinate_map_key(tensor):
    if hasattr(tensor, "coordinate_map_key"):
        return tensor.coordinate_map_key
    return tensor.coords_key


def coordinate_manager(tensor):
    if hasattr(tensor, "coordinate_manager"):
        return tensor.coordinate_manager
    if hasattr(tensor, "coords_man"):
        return tensor.coords_man
    return tensor.coords_manager
