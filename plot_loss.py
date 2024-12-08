import re
import matplotlib.pyplot as plt


def smooth(data, alpha=0.1):
    smoothed = []
    for i, point in enumerate(data):
        if i == 0:
            smoothed.append(point)
        else:
            smoothed_value = alpha * point + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_value)
    return smoothed


def load_loss_from_file(log_file_path, smoothing_factor=0.1):
    with open(log_file_path, "r") as file:
        log_lines = file.readlines()

    timestamps = []
    losses = []

    for line in log_lines:
        match = re.search(r"loss:\s*([\d\.]+)", line, re.IGNORECASE)
        if match:
            loss = float(match.group(1))
            losses.append(loss)
            timestamps.append(len(timestamps) + 1)

    smoothed_losses = smooth(losses, alpha=smoothing_factor)
    return timestamps, losses, smoothed_losses


def load_and_plot_loss(log_file_path, smoothing_factor=0.1):
    timestamps, losses, smoothed_losses = load_loss_from_file(
        log_file_path, smoothing_factor
    )

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, losses, label="Original Loss", alpha=0.5)
    plt.plot(
        timestamps,
        smoothed_losses,
        label=f"Smoothed Loss (α={smoothing_factor})",
        linewidth=2,
    )
    plt.xlabel("Time (log index)")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()
    plt.show()


def load_and_plot_validation_loss(log_file_path):
    with open(log_file_path, "r") as file:
        log_lines = file.readlines()

    batch_losses = []
    batch_loss_timestamps = []
    estimate_losses = []
    estimate_timestamps = []

    for i, line in enumerate(log_lines):
        batch_match = re.search(r"Validation Batch Loss:\s*([\d\.]+)", line)
        if batch_match:
            batch_losses.append(float(batch_match.group(1)))
            batch_loss_timestamps.append(len(batch_losses))

        estimate_match = re.search(r"Validation Loss Estimate:\s*([\d\.]+)", line)
        if estimate_match:
            estimate_losses.append(float(estimate_match.group(1)))
            estimate_timestamps.append(len(batch_losses))

    plt.figure(figsize=(12, 6))
    plt.plot(
        batch_loss_timestamps,
        batch_losses,
        label="Validation Batch Loss (mini-batches)",
        alpha=0.6,
        linestyle="--",
    )
    plt.scatter(
        estimate_timestamps,
        estimate_losses,
        color="red",
        label="Validation Loss Estimate",
        zorder=3,
    )
    plt.xlabel("Validation Iteration Index")
    plt.ylabel("Loss")
    plt.title("Validation Loss Over Time")
    plt.legend()
    plt.grid()
    plt.show()


def compare_multiple_losses(log_files, smoothing_factor=0.1, max_iterations=None):

    plt.figure(figsize=(12, 6))

    for i, log_file in enumerate(log_files):
        timestamps, losses, smoothed_losses = load_loss_from_file(
            log_file, smoothing_factor
        )

        if max_iterations is not None:
            timestamps, losses, smoothed_losses = [
                x[:max_iterations] for x in (timestamps, losses, smoothed_losses)
            ]

        plt.plot(
            timestamps,
            smoothed_losses,
            label=f"Smoothed Loss (File {i+1}, α={smoothing_factor})",
            linewidth=2,
        )
        plt.plot(
            timestamps,
            losses,
            label=f"Original Loss (File {i+1})",
            alpha=0.5,
        )

    plt.xlabel("Time (log index)")
    plt.ylabel("Loss")
    plt.title(
        f"Comparison of Training Losses (up to iteration {max_iterations if max_iterations else 'all'})"
    )
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    smoothing_factor = 0.1
    log_file_path = "training_log_2024-12-06_08-16-57-no-vlm-mask.log"
    load_and_plot_loss(log_file_path, smoothing_factor=smoothing_factor)
    val_log_file_path = "training_log_2024-12-06_08-16-57-no-vlm-mask.log"
    load_and_plot_validation_loss(val_log_file_path)

    loss_files = [
        "training_log_2024-12-06_08-16-57-no-vlm-mask.log",
        "training_log_2024-12-05_21-23-22.log",
        "training_log_2024-12-07_17-07-07.log",
    ]
    compare_multiple_losses(
        loss_files,
        max_iterations=5000,
        smoothing_factor=smoothing_factor,
    )
