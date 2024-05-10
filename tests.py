import torch
from gpt import GPT, GPTConfig


def test_combined_mask_effectiveness():
    batch_size = 4
    sequence_length = 10
    pad_token_id = 0  # Assuming '0' is used as the pad token
    config = GPTConfig()
    model = GPT(config)

    # Simulate input data with padding
    dummy_input = torch.tensor(
        [
            [1, 2, 3, 0, 0],  # Padded after three tokens
            [1, 2, 0, 0, 0],  # Padded after two tokens
            [1, 2, 3, 4, 5],  # No padding
            [0, 0, 0, 0, 0],  # All padded
        ]
    )

    # Create padding mask
    padding_mask = (dummy_input != pad_token_id).unsqueeze(1).unsqueeze(2).bool()

    # Generate causal mask
    causal_mask = model._create_causal_mask(dummy_input.size(1))

    # Combine the masks
    combined_mask = causal_mask & padding_mask

    # Assert the shape of the combined mask
    assert combined_mask.shape == (
        batch_size,
        1,
        dummy_input.size(1),
        dummy_input.size(1),
    ), "Mask shape is incorrect"

    # Assert that the combined mask is correctly masking padded tokens
    expected_mask = torch.tensor(
        [
            [[1, 1, 1, 0, 0]],  # Last two are padding
            [[1, 1, 0, 0, 0]],  # Last three are padding
            [[1, 1, 1, 1, 1]],  # No padding
            [[0, 0, 0, 0, 0]],  # All are padding
        ]
    )
    assert torch.all(
        combined_mask[:, :, 0, :] == expected_mask
    ), "Padding mask is not applied correctly"

    # Assert causal properties
    # Check that the matrix for non-padded inputs is lower triangular
    assert torch.all(
        combined_mask[2, 0, :, :] == torch.tril(torch.ones(5, 5)).bool()
    ), "Causal mask is not lower triangular where expected"

    # Run the test with pytest directly or through your command line


# Run the test
if __name__ == "__main__":
    test_combined_mask_effectiveness()
