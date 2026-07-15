
from datetime import datetime
from pathlib import Path

ROOT_DIR        = Path(__file__).parent.parent
LOGS_DIR        = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
ACTIVE_LOG_FILE = ROOT_DIR / "checkpoints" / "active_log.txt"


def _get_log_path():
    if not ACTIVE_LOG_FILE.exists():
        raise FileNotFoundError("No active log. Run --train before --evaluate.")
    return Path(ACTIVE_LOG_FILE.read_text().strip())


def _write(log_path, text):
    with open(log_path, 'a') as f:
        f.write(text + "\n")
    print(text)


def start_run(config):
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M")
    pos_weight = config['training'].get('pos_weight', 'N/A')
    lr         = config['training']['learning_rate']
    lw_on      = config['training'].get('loss_weight_onset', 'N/A')
    lw_off     = config['training'].get('loss_weight_offset', 'N/A')
    filename   = f"{timestamp}_pw{pos_weight}_lr{lr}_lwo{lw_on}.log"

    log_path = LOGS_DIR / filename
    ACTIVE_LOG_FILE.parent.mkdir(exist_ok=True)
    ACTIVE_LOG_FILE.write_text(str(log_path))

    lines = [
        "=" * 70,
        "  EXPERIMENT LOG - MAESTRO Phase 2",
        f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  File    : {filename}",
        "=" * 70,
        "",
        "-- HYPERPARAMETERS --------------------------------------------------",
        "",
        "  [ Model ]",
        f"  input_features     : {config['model']['input_features']}",
        f"  dropout            : {config['model']['dropout']}",
        "",
        "  [ Training ]",
        f"  batch_size         : {config['training']['batch_size']}",
        f"  learning_rate      : {config['training']['learning_rate']}",
        f"  epochs             : {config['training']['epochs']}",
        f"  pos_weight (pitch) : {config['training'].get('pos_weight', 'N/A')}",
        f"  pos_weight_onset   : {config['training'].get('pos_weight_onset', 'N/A')}",
        f"  pos_weight_offset  : {config['training'].get('pos_weight_offset', 'N/A')}",
        f"  loss_weight_onset  : {config['training'].get('loss_weight_onset', 'N/A')}",
        f"  loss_weight_offset : {config['training'].get('loss_weight_offset', 'N/A')}",
        f"  scheduler_patience : {config['training']['scheduler_patience']}",
        f"  scheduler_factor   : {config['training']['scheduler_factor']}",
        f"  min_lr             : {config['training']['min_lr']}",
        "",
        "  [ Dataset ]",
        f"  chunk_duration     : {config['dataset']['chunk_duration']} s",
        f"  sample_rate        : {config['dataset']['sample_rate']} Hz",
        f"  hop_length         : {config['dataset']['hop_length']}",
        "",
        "  [ Evaluation ]",
        f"  threshold          : {config['evaluation']['threshold']}",
        "",
        "-- TRAINING ---------------------------------------------------------",
        "",
        f"  {'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'LR':<14} {'Time':<10}",
        f"  {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*10}",
    ]

    with open(log_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Log file created: {log_path}")
    return str(log_path)


def log_epoch(epoch, avg_loss, val_loss, current_lr, epoch_time=None):
    log_path = _get_log_path()
    time_str = f"{epoch_time:.1f}s" if epoch_time is not None else "N/A"
    line     = f"  {epoch+1:<8} {avg_loss:<12.6f} {val_loss:<12.6f} {current_lr:<14.6f} {time_str:<10}"
    _write(log_path, line)


def end_training():
    log_path = _get_log_path()
    lines = ["", "-- END OF TRAINING --------------------------------------------------", ""]
    with open(log_path, 'a') as f:
        f.write("\n".join(lines) + "\n")
    print("Training phase logged.")


def log_metrics(accuracy, precision, recall, f1,
                max_prob, mean_prob, active_preds, active_labels, threshold,
                f1_onset=None, f1_offset=None):

    log_path = _get_log_path()

    lines = [
        "-- EVALUATION -------------------------------------------------------",
        "",
        f"  Evaluated at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Threshold      : {threshold}",
        "",
        "  [ Pitch ]",
        f"  Accuracy       : {accuracy:.4f}",
        f"  Precision      : {precision:.4f}",
        f"  Recall         : {recall:.4f}",
        f"  F1-Score       : {f1:.4f}",
        "",
        "  [ Onset / Offset ]",
        f"  F1 Onset       : {f1_onset:.4f}"  if f1_onset  is not None else "  F1 Onset       : N/A",
        f"  F1 Offset      : {f1_offset:.4f}" if f1_offset is not None else "  F1 Offset      : N/A",
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

    if ACTIVE_LOG_FILE.exists():
        ACTIVE_LOG_FILE.unlink()

    print(f"Evaluation metrics logged.")
    print(f"Log saved at: {log_path}")