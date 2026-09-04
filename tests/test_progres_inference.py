import torch

from scripts.progres_inference import ProgresModel, embed_coordinates, featurize


def test_progres_features_and_embedding_are_rigid_motion_invariant() -> None:
    torch.manual_seed(7)
    coordinates = torch.randn(12, 3) * 4
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1
    transformed = coordinates @ rotation + torch.tensor([10.0, -3.0, 4.0])
    features, adjacency = featurize(coordinates)
    transformed_features, transformed_adjacency = featurize(transformed)
    assert features.shape == (12, 68)
    assert torch.equal(adjacency, transformed_adjacency)
    assert torch.allclose(features, transformed_features, atol=1e-5)
    model = ProgresModel().eval()
    with torch.no_grad():
        embedding = model(features, coordinates, adjacency)
        transformed_embedding = model(
            transformed_features, transformed, transformed_adjacency,
        )
    assert embedding.shape == (128,)
    assert torch.linalg.vector_norm(embedding) == torch.tensor(1.0)
    assert torch.allclose(embedding, transformed_embedding, atol=1e-5)
    assert torch.allclose(embed_coordinates(model, coordinates), embedding)


def test_embed_coordinates_rejects_invalid_traces() -> None:
    model = ProgresModel().eval()
    for coordinates in (torch.zeros(3, 3), torch.zeros(4, 2), torch.full((4, 3), float("nan"))):
        try:
            embed_coordinates(model, coordinates)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid coordinates were accepted")
