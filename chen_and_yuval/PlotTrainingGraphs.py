import glob

import matplotlib.pyplot as plt


# Function to parse the file and extract epochs and validation losses
def parse_log_file(file_path):
    epochs = []
    val_losses = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' ')
            epoch_part = parts[1].strip()
            val_loss_part = parts[-1].strip()

            epoch = int(epoch_part.split('/')[0])
            val_loss = float(val_loss_part)

            if epoch == 1:
                epochs = []
                val_losses = []

            epochs.append(epoch)
            val_losses.append(val_loss)

    return epochs, val_losses

files = glob.glob('good_logs_and_models/*training_log.txt')
tv_modes = [f.split('/')[-1].split('_')[0] for f in files] # ['none', 'sharp', 'smooth']
markers = {tv_mode: marker for tv_mode, marker in zip(tv_modes, ['o', 's', '^', 'd'])}

plt.figure(figsize=(10, 6))
for log_name, tv_mode in zip(files, tv_modes):
    #log_name = tv_mode + "_training_log.txt"

    # Parse the log file
    epochs, val_losses = parse_log_file(log_name)

    # Plot the validation loss per epoch
    plt.plot(epochs, val_losses, marker=markers[tv_mode], linestyle='-', label="TV mode: "+tv_mode)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()
