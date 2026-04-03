# This file creates a simple text logger for training and evaluation 
# Each experiment is saved as a .log file in the 'logs/' directory
#
# run_id is saved between --train and --evaluate so they 
# can be called separately but write to the same log file.
#
# functions :
#   start_run(config)           : creates the log file, writes hyperparameters
#   log_epoch(epoch, loss, lr)  : appends one line per epoch
#   end_training()              : writes a separator at the end of training
#   log_metrics(...)            : appends evaluation metrics to the log file

import json
from datetime import datetime
from pathlib import Path

# --- Paths ---
ROOT_DIR     = Path(__file__).parent.parent
LOGS_DIR     = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# active log file path between --train and --evaluate
ACTIVE_LOG_FILE = ROOT_DIR / "checkpoints" / "active_log.txt"


# --- utils ---

'''
    Returns the path of the active log
    if the file is not found, returns FileNotFoundError
'''
def _get_log_path():
    
    if not ACTIVE_LOG_FILE.exists():
        raise FileNotFoundError(
            "No active log found.\n"
            "Run --train before --evaluate."
        )
    return Path(ACTIVE_LOG_FILE.read_text().strip())

'''
    Appends a line of text into the log file
'''
def _write(log_path, text):
    with open(log_path, 'a') as f:
        f.write(text + "\n")
    print(text)


# --- Run logs ---

'''
    Creates a log file for the run 
    writes all hyperparameters
    called at the beginning of train()

    the name is in form :
        e.g. 2026-04-01_14-32_pw4.2_lr0.0005_hs256.log
'''
def start_run(config):



    # Build a descriptive filename from the most important hyperparameters
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M")
    pos_weight = config['training'].get('pos_weight', 'N/A')
    lr         = config['training']['learning_rate']
    hs         = config['model']['hidden_size']
    filename   = f"{timestamp}_pw{pos_weight}_lr{lr}_hs{hs}.log"

    log_path = LOGS_DIR / filename

    # Persist the log path so evaluate.py can find it
    ACTIVE_LOG_FILE.parent.mkdir(exist_ok=True)
    ACTIVE_LOG_FILE.write_text(str(log_path))

    
    lines = [
        "=" * 70,
        f"  EXPERIMENT LOG",
        f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  File    : {filename}",
        "=" * 70,
        "",
        "-- HYPERPARAMETERS --------------------------------------------------",
        "",
        "  [ Model ]",
        f"  input_features     : {config['model']['input_features']}",
        f"  hidden_size        : {config['model']['hidden_size']}",
        f"  lstm_layers        : {config['model']['lstm_layers']}",
        f"  dropout            : {config['model']['dropout']}",
        "",
        "  [ Training ]",
        f"  batch_size         : {config['training']['batch_size']}",
        f"  learning_rate      : {config['training']['learning_rate']}",
        f"  epochs             : {config['training']['epochs']}",
        f"  pos_weight         : {config['training'].get('pos_weight', 'N/A')}",
        f"  scheduler_patience : {config['training']['scheduler_patience']}",
        f"  scheduler_factor   : {config['training']['scheduler_factor']}",
        "",
        "  [ Dataset ]",
        f"  chunk_duration     : {config['dataset']['chunk_duration']} s",
        f"  sample_rate        : {config['dataset']['sample_rate']} Hz",
        "",
        "  [ Evaluation ]",
        f"  threshold          : {config['evaluation']['threshold']}",
        "",
        "-- TRAINING ---------------------------------------------------------",
        "",
        f"  {'Epoch':<8} {'Loss':<12} {'LR':<14} {'Time':<10}",
        f"  {'-'*8} {'-'*12} {'-'*14} {'-'*10}",
    ]

    with open(log_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    print(f"Log file created: {log_path}")
    return str(log_path)




'''
    Appends a line of text in the log for the current epoch
    called at the end of each epoch
'''
def log_epoch(epoch, avg_loss, current_lr, epoch_time=None):

    log_path  = _get_log_path()
    time_str  = f"{epoch_time:.1f}s" if epoch_time is not None else "N/A"
    line      = f"  {epoch+1:<8} {avg_loss:<12.6f} {current_lr:<14.6f} {time_str:<10}"
    _write(log_path, line)




'''
    Writes a separator line at the end of training
    this doesn't close the log, it will be resumed by evaluate()
    called after the training loop
'''
def end_training():

    log_path = _get_log_path()
    lines = [
        "",
        "-- END OF TRAINING --------------------------------------------------",
        "",
    ]
    with open(log_path, 'a') as f:
        f.write("\n".join(lines) + "\n")
    print("Training phase logged.")




# --- Evaluation ---



'''
    Appends evaluation metrics to the active log, then closes the run
    called at the end of evaluate()
'''
def log_metrics(accuracy, precision, recall, f1,
                max_prob, mean_prob, active_preds, active_labels, threshold):

    log_path = _get_log_path()

    lines = [
        "-- EVALUATION -------------------------------------------------------",
        "",
        f"  Evaluated at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Threshold      : {threshold}",
        "",
        f"  Accuracy       : {accuracy:.4f}",
        f"  Precision      : {precision:.4f}",
        f"  Recall         : {recall:.4f}",
        f"  F1-Score       : {f1:.4f}",
        "",
        f"  Max prob       : {max_prob:.4f}",
        f"  Mean prob      : {mean_prob:.4f}",
        f"  % active preds : {active_preds:.4f}",
        f"  % active labels: {active_labels:.4f}",
        "",
        "=" * 70,
        "",
    ]

    with open(log_path, 'a') as f:
        f.write("\n".join(lines) + "\n")

    # Clean up the active log pointer — next --train will create a fresh file
    if ACTIVE_LOG_FILE.exists():
        ACTIVE_LOG_FILE.unlink()

    print(f"Evaluation metrics logged.")
    print(f"Log saved at: {log_path}")