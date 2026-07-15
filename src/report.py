
from datetime import datetime
from pathlib import Path

ROOT_DIR        = Path(__file__).parent.parent
LOGS_DIR        = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

ACTIVE_LOG_FILE = ROOT_DIR / "checkpoints" / "active_log.txt"



def _get_log_path():
    if not ACTIVE_LOG_FILE.exists():
        raise FileNotFoundError(
            "No active log found.\n"
            "Run --train before --evaluate."
        )
    return Path(ACTIVE_LOG_FILE.read_text().strip())


def _write(log_path, text):
    with open(log_path, 'a') as f:
        f.write(text + "\n")
    print(text)



def start_run(config):
    """
    Crea il file di log per la run e scrive tutti gli iperparametri.
    Chiamato all'inizio di train().
    Nome file: e.g. 2026-05-01_14-32_pwp3.0_lr0.001_do0.4.log
    """
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M")
    pw_pitch   = config['training'].get('pos_weight_pitch', 'N/A')
    lr         = config['training']['learning_rate']
    dropout    = config['model']['dropout']
    filename   = f"{timestamp}_pwp{pw_pitch}_lr{lr}_do{dropout}.log"

    log_path = LOGS_DIR / filename

    ACTIVE_LOG_FILE.parent.mkdir(exist_ok=True)
    ACTIVE_LOG_FILE.write_text(str(log_path))

    thr_pitch  = config['evaluation'].get('threshold_pitch',  'N/A')
    thr_onset  = config['evaluation'].get('threshold_onset',  'N/A')
    thr_offset = config['evaluation'].get('threshold_offset', 'N/A')

    pw_onset   = config['training'].get('pos_weight_onset',  'N/A')
    pw_offset  = config['training'].get('pos_weight_offset', 'N/A')
    lw_pitch   = config['training'].get('loss_weight_pitch',  'N/A')
    lw_onset   = config['training'].get('loss_weight_onset',  'N/A')
    lw_offset  = config['training'].get('loss_weight_offset', 'N/A')

    lines = [
        "=" * 70,
        f"  EXPERIMENT LOG  –  Fase 2: Multi-Output CNN",
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
        f"  pos_weight_pitch   : {pw_pitch}",
        f"  pos_weight_onset   : {pw_onset}",
        f"  pos_weight_offset  : {pw_offset}",
        f"  loss_weight_pitch  : {lw_pitch}",
        f"  loss_weight_onset  : {lw_onset}",
        f"  loss_weight_offset : {lw_offset}",
        f"  scheduler_patience : {config['training']['scheduler_patience']}",
        f"  scheduler_factor   : {config['training']['scheduler_factor']}",
        "",
        "  [ Dataset ]",
        f"  chunk_duration     : {config['dataset']['chunk_duration']} s",
        f"  sample_rate        : {config['dataset']['sample_rate']} Hz",
        "",
        "  [ Evaluation ]",
        f"  threshold_pitch    : {thr_pitch}",
        f"  threshold_onset    : {thr_onset}",
        f"  threshold_offset   : {thr_offset}",
        "",
        "-- TRAINING ---------------------------------------------------------",
        "",
        f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'LR':<14} {'Time':<10}",
        f"  {'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*10}",
    ]

    with open(log_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    print(f"Log file created: {log_path}")
    return str(log_path)


def log_epoch(epoch, train_loss, val_loss, current_lr, epoch_time=None):
    """Appende una riga per ogni epoch con train loss e val loss."""
    log_path = _get_log_path()
    time_str = f"{epoch_time:.1f}s" if epoch_time is not None else "N/A"
    line     = (f"  {epoch+1:<8} {train_loss:<14.6f} {val_loss:<14.6f} "
                f"{current_lr:<14.6f} {time_str:<10}")
    _write(log_path, line)


def end_training():
    """Scrive il separatore di fine training."""
    log_path = _get_log_path()
    lines = [
        "",
        "-- END OF TRAINING --------------------------------------------------",
        "",
    ]
    with open(log_path, 'a') as f:
        f.write("\n".join(lines) + "\n")
    print("Training phase logged.")



def _format_metrics_block(name, metrics, threshold):
    """
    metrics = (accuracy, precision, recall, f1, max_prob, mean_prob,
               active_preds, active_labels)
    """
    acc, prec, rec, f1, maxp, meanp, ap, al = metrics
    return [
        f"  [ {name} – threshold={threshold} ]",
        f"  Accuracy       : {acc:.4f}",
        f"  Precision      : {prec:.4f}",
        f"  Recall         : {rec:.4f}",
        f"  F1-Score       : {f1:.4f}",
        f"  Max prob       : {maxp:.4f}",
        f"  Mean prob      : {meanp:.4f}",
        f"  % active preds : {ap:.4f}",
        f"  % active labels: {al:.4f}",
        "",
    ]


def log_metrics(pitch_metrics, onset_metrics, offset_metrics,
                thr_pitch, thr_onset, thr_offset):
    """
    Appende le metriche di valutazione dei tre output al log attivo.

    Parametri
    ---------
    pitch_metrics / onset_metrics / offset_metrics :
        tuple (accuracy, precision, recall, f1,
               max_prob, mean_prob, active_preds, active_labels)
    thr_pitch / thr_onset / thr_offset : float
    """
    log_path = _get_log_path()

    lines = [
        "-- EVALUATION -------------------------------------------------------",
        "",
        f"  Evaluated at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    lines += _format_metrics_block("PITCH",  pitch_metrics,  thr_pitch)
    lines += _format_metrics_block("ONSET",  onset_metrics,  thr_onset)
    lines += _format_metrics_block("OFFSET", offset_metrics, thr_offset)
    lines += [
        "=" * 70,
        "",
    ]

    with open(log_path, 'a') as f:
        f.write("\n".join(lines) + "\n")

    if ACTIVE_LOG_FILE.exists():
        ACTIVE_LOG_FILE.unlink()

    print(f"Evaluation metrics logged.")
    print(f"Log saved at: {log_path}")