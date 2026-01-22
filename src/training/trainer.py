"""Training Framework with W&B Integration"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
import wandb
from typing import Optional, Dict, Any
import time
import subprocess
import itertools

from ..models import GPT2
from ..config import Config
from .evaluator import Evaluator
from .utils import get_learning_rate_scheduler, get_device, set_seed


class Trainer:
    """Trainer for GPT-2 model with advanced features"""
    
    def __init__(
        self,
        model: GPT2,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Config] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: GPT-2 model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            config: Configuration object
            device: Device to train on (auto-detected if None)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device if device is not None else get_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = None
        if config is not None:
            total_steps = len(train_loader) * config.training.max_epochs // config.training.gradient_accumulation_steps
            self.scheduler = get_learning_rate_scheduler(
                self.optimizer,
                warmup_steps=config.training.warmup_steps,
                total_steps=total_steps
            )
        
        # Mixed precision training
        self.use_amp = config.training.use_mixed_precision if config else False
        self.scaler = GradScaler() if self.use_amp else None
        
        # Evaluation
        self.evaluator = Evaluator(model, self.device)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.wandb_run_id = None  # Store W&B run ID for resuming
        
        # Output directory
        if config:
            self.output_dir = Path(config.output_dir)
        else:
            self.output_dir = Path("./outputs")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir.absolute()}")
        
        # W&B initialization (will be done after checkpoint loading if resuming)
        # Don't initialize here if we might be resuming
        
        # Temperature monitoring settings
        self.max_safe_temp = 80.0  # Target: ≤ 80°C (ideally 72-76°C)
        self.critical_temp = 83.0  # Pause if above this
        self.ideal_temp = 76.0     # Resume when below this
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        if self.config is None:
            return AdamW(self.model.parameters(), lr=6e-4)
        
        return AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.beta1, self.config.training.beta2)
        )
    
    def _init_wandb(self, resume_run_id: Optional[str] = None):
        """
        Initialize Weights & Biases
        
        Args:
            resume_run_id: Optional W&B run ID to resume. If provided, will resume
                          the existing run instead of creating a new one.
        """
        if not self.config.wandb.enabled:
            return
        
        try:
            init_kwargs = {
                'project': self.config.wandb.project,
                'entity': self.config.wandb.entity,
                'name': self.config.wandb.name,
                'tags': self.config.wandb.tags,
                'config': self.config.to_dict()
            }
            
            # Resume existing run if run_id is provided
            if resume_run_id:
                init_kwargs['resume'] = 'must'
                init_kwargs['id'] = resume_run_id
                print(f"🔄 Resuming W&B run: {resume_run_id}")
            else:
                print("🆕 Starting new W&B run")
            
            wandb.init(**init_kwargs)
            
            # Store the run ID for saving in checkpoints
            self.wandb_run_id = wandb.run.id if wandb.run else None
            
            # Log model architecture
            wandb.watch(self.model, log="all", log_freq=100)
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
            print("Continuing training without W&B logging...")
            self.config.wandb.enabled = False
    
    def _setup_temperature_monitoring(self):
        """Setup temperature monitoring settings"""
        # Temperature monitoring settings
        self.max_safe_temp = 80.0  # Target: ≤ 80°C (ideally 72-76°C)
        self.critical_temp = 83.0  # Pause if above this
        self.ideal_temp = 76.0     # Resume when below this
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary with loss and other metrics
        """
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Create labels (shifted input_ids)
        labels = input_ids.clone()
        
        # Forward pass with mixed precision
        if self.use_amp:
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
                # Scale loss for gradient accumulation
                loss = loss / (self.config.training.gradient_accumulation_steps if self.config else 1)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
            loss = loss / (self.config.training.gradient_accumulation_steps if self.config else 1)
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        metrics = {
            'loss': loss.item() * (self.config.training.gradient_accumulation_steps if self.config else 1),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Clean up tensors to prevent memory leak
        del input_ids, labels, outputs, loss
        if attention_mask is not None:
            del attention_mask
        
        # Track steps for periodic garbage collection
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        
        # Force garbage collection every 10 steps
        if self._step_count % 10 == 0:
            import gc
            gc.collect()
        
        return metrics
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping"""
        if self.config:
            # Gradient clipping
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.optimizer.step()
        else:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        # Clear cache after optimizer step
        torch.cuda.empty_cache()
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
    
    def train(self):
        """Main training loop"""
        if self.config is None:
            raise ValueError("Config is required for training")
        
        max_epochs = self.config.training.max_epochs
        gradient_accumulation_steps = self.config.training.gradient_accumulation_steps
        eval_interval = self.config.training.eval_interval
        save_interval = self.config.training.save_interval
        log_interval = self.config.training.log_interval
        
        # Resume from checkpoint if specified
        resume_wandb_id = None
        if self.config.training.resume_from:
            resume_wandb_id = self.load_checkpoint(self.config.training.resume_from)
        
        # Initialize W&B after checkpoint loading (so we can resume the run)
        if self.config.wandb.enabled and not wandb.run:
            self._init_wandb(resume_run_id=resume_wandb_id)
        
        # Track if we're resuming in the middle of an epoch
        resume_epoch = self.current_epoch
        resume_step = self.global_step
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            self.model.train()
            
            accumulated_loss = 0.0
            accumulated_steps = 0
            
            # Calculate starting batch index when resuming
            # Each step processes gradient_accumulation_steps batches
            start_batch_idx = 0
            if epoch == resume_epoch and resume_step > 0:
                # We're resuming in the middle of an epoch
                # Calculate which batch we should start from
                start_batch_idx = resume_step * gradient_accumulation_steps
                print(f"📊 Resuming from batch {start_batch_idx} (step {resume_step})")
                # Reset resume flags after first use
                resume_epoch = -1
                resume_step = 0
            
            total_batches = len(self.train_loader)
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch+1}/{max_epochs}",
                initial=start_batch_idx,
                total=total_batches
            )
            
            # Skip batches if resuming in the middle of an epoch
            batch_iter = iter(progress_bar)
            if start_batch_idx > 0:
                print(f"⏩ Skipping {start_batch_idx} batches to resume from step {self.global_step}...")
                # Manually skip batches (this allows tqdm to show progress)
                for _ in range(start_batch_idx):
                    next(batch_iter, None)
                    progress_bar.update(1)
            
            for batch_idx, batch in enumerate(batch_iter):
                # Adjust batch_idx to account for skipped batches
                actual_batch_idx = start_batch_idx + batch_idx
                # Training step
                metrics = self.train_step(batch)
                accumulated_loss += metrics['loss']
                accumulated_steps += 1
                
                # Gradient accumulation
                if (actual_batch_idx + 1) % gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Periodic memory cleanup (every 50 steps - more frequent)
                    if self.global_step % 50 == 0:
                        torch.cuda.empty_cache()
                    
                    # Logging
                    if self.global_step % log_interval == 0:
                        avg_loss = accumulated_loss / accumulated_steps
                        log_dict = {
                            'train/loss': avg_loss,
                            'train/learning_rate': metrics['learning_rate'],
                            'train/epoch': epoch,
                            'train/step': self.global_step
                        }
                        
                        # Add GPU metrics to logging
                        gpu_metrics = self._get_gpu_metrics()
                        if gpu_metrics:
                            # Prefix all GPU metrics with 'gpu/' for W&B organization
                            for key, value in gpu_metrics.items():
                                log_dict[f'gpu/{key}'] = value
                        
                        if self.config.wandb.enabled:
                            wandb.log(log_dict, step=self.global_step)
                        
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{metrics["learning_rate"]:.2e}'
                        })
                        
                        accumulated_loss = 0.0
                        accumulated_steps = 0
                    
                    # Evaluation (with memory cleanup)
                    if self.val_loader and self.global_step % eval_interval == 0:
                        # Clear cache before evaluation
                        torch.cuda.empty_cache()
                        eval_metrics = self.evaluator.evaluate(self.val_loader)
                        eval_log_dict = {
                            f'val/{k}': v for k, v in eval_metrics.items()
                        }
                        eval_log_dict['train/step'] = self.global_step
                        
                        # Add GPU metrics during evaluation
                        gpu_metrics = self._get_gpu_metrics()
                        if gpu_metrics:
                            for key, value in gpu_metrics.items():
                                eval_log_dict[f'gpu/{key}'] = value
                        
                        if self.config.wandb.enabled:
                            wandb.log(eval_log_dict, step=self.global_step)
                        
                        print(f"\nStep {self.global_step} - Validation Metrics:")
                        for k, v in eval_metrics.items():
                            print(f"  {k}: {v:.4f}")
                        
                        # Clear cache after evaluation
                        torch.cuda.empty_cache()
                        del eval_metrics, eval_log_dict
                        
                        # Check GPU temperature after evaluation
                        self._check_and_wait_for_safe_temp()
                    
                    # Save checkpoint
                    if self.global_step % save_interval == 0:
                        success = self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                        if not success:
                            print(f"⚠️  WARNING: Checkpoint save failed at step {self.global_step}, but training continues...")
                        # Clear cache after saving to free up memory
                        torch.cuda.empty_cache()
            
            # Final optimizer step if needed
            if accumulated_steps > 0:
                self.optimizer_step()
                self.global_step += 1
            
            # End of epoch evaluation
            if self.val_loader:
                eval_metrics = self.evaluator.evaluate(self.val_loader)
                eval_log_dict = {
                    f'val/{k}': v for k, v in eval_metrics.items()
                }
                eval_log_dict['train/epoch'] = epoch + 1
                
                if self.config.wandb.enabled:
                    wandb.log(eval_log_dict, step=self.global_step)
                
                print(f"\nEpoch {epoch+1} - Validation Metrics:")
                for k, v in eval_metrics.items():
                    print(f"  {k}: {v:.4f}")
            
            # Save epoch checkpoint
            success = self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            if not success:
                print(f"⚠️  WARNING: Failed to save epoch checkpoint at epoch {epoch+1}")
        
        # Save final model
        success = self.save_checkpoint("final_model.pt")
        if not success:
            print(f"⚠️  WARNING: Failed to save final model checkpoint")
        
        if self.config.wandb.enabled:
            wandb.finish()
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get comprehensive GPU metrics using nvidia-smi
        
        Returns:
            Dictionary with GPU metrics or None if unavailable
        """
        if not torch.cuda.is_available():
            return None
        
        try:
            # Query multiple GPU metrics at once
            result = subprocess.run(
                ['nvidia-smi', 
                 '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse the output (assuming single GPU for now)
                line = result.stdout.strip().split('\n')[0]
                parts = [p.strip() for p in line.split(',')]
                
                if len(parts) >= 8:
                    gpu_id = int(parts[0])
                    gpu_util = float(parts[1])
                    mem_util = float(parts[2])
                    mem_used = float(parts[3])  # MB
                    mem_total = float(parts[4])  # MB
                    temperature = float(parts[5])
                    
                    # Parse power values (may be "[Not Supported]" on some systems)
                    power_draw = None
                    power_limit = None
                    try:
                        if parts[6] and parts[6] != '[Not Supported]':
                            power_draw = float(parts[6])
                    except (ValueError, IndexError):
                        pass
                    try:
                        if parts[7] and parts[7] != '[Not Supported]':
                            power_limit = float(parts[7])
                    except (ValueError, IndexError):
                        pass
                    
                    # Get PyTorch memory stats
                    torch_mem_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**2)  # MB
                    torch_mem_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**2)  # MB
                    torch_mem_max_allocated = torch.cuda.max_memory_allocated(gpu_id) / (1024**2)  # MB
                    torch_mem_max_reserved = torch.cuda.max_memory_reserved(gpu_id) / (1024**2)  # MB
                    
                    metrics = {
                        'gpu_utilization': gpu_util,
                        'memory_utilization': mem_util,
                        'memory_used_mb': mem_used,
                        'memory_total_mb': mem_total,
                        'memory_used_percent': (mem_used / mem_total * 100) if mem_total > 0 else 0,
                        'temperature': temperature,
                        'torch_memory_allocated_mb': torch_mem_allocated,
                        'torch_memory_reserved_mb': torch_mem_reserved,
                        'torch_memory_max_allocated_mb': torch_mem_max_allocated,
                        'torch_memory_max_reserved_mb': torch_mem_max_reserved,
                    }
                    
                    if power_draw is not None:
                        metrics['power_draw_watts'] = power_draw
                    if power_limit is not None:
                        metrics['power_limit_watts'] = power_limit
                        if power_draw is not None:
                            metrics['power_utilization_percent'] = (power_draw / power_limit * 100)
                    
                    return metrics
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError, IndexError) as e:
            pass
        return None
    
    def _get_gpu_temperature(self) -> Optional[float]:
        """Get current GPU temperature in Celsius"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return None
    
    def _get_average_gpu_temperature(self, duration_seconds: float = 5.0, sample_interval: float = 0.5) -> Optional[float]:
        """
        Sample GPU temperature over specified duration and return average
        
        Args:
            duration_seconds: Total duration to sample (default: 5 seconds)
            sample_interval: Interval between samples in seconds (default: 0.5 seconds)
            
        Returns:
            Average temperature in Celsius, or None if unable to read
        """
        temperatures = []
        num_samples = int(duration_seconds / sample_interval)
        
        for _ in range(num_samples):
            temp = self._get_gpu_temperature()
            if temp is not None:
                temperatures.append(temp)
            time.sleep(sample_interval)
        
        if len(temperatures) > 0:
            return sum(temperatures) / len(temperatures)
        return None
    
    def _check_and_wait_for_safe_temp(self):
        """
        Check GPU temperature and pause training if too hot.
        Wait until temperature drops to safe operating range.
        """
        if not torch.cuda.is_available():
            return
        
        print("\n🌡️  Checking GPU temperature...")
        avg_temp = self._get_average_gpu_temperature(duration_seconds=5.0, sample_interval=0.5)
        
        if avg_temp is None:
            print("⚠️  Could not read GPU temperature. Continuing training...")
            return
        
        print(f"📊 Average GPU temperature: {avg_temp:.1f}°C")
        
        # Log comprehensive GPU metrics to W&B
        if self.config and self.config.wandb.enabled:
            gpu_metrics = self._get_gpu_metrics()
            if gpu_metrics:
                # Add temperature to the metrics
                gpu_metrics['temperature'] = avg_temp
                log_dict = {f'gpu/{k}': v for k, v in gpu_metrics.items()}
                wandb.log(log_dict, step=self.global_step)
            else:
                # Fallback to just temperature if full metrics unavailable
                wandb.log({'gpu/temperature': avg_temp}, step=self.global_step)
        
        # Check if temperature is critical
        if avg_temp >= self.critical_temp:
            print(f"🚨 GPU temperature is CRITICAL: {avg_temp:.1f}°C (≥ {self.critical_temp}°C)")
            print(f"⏸️  Pausing training until temperature drops below {self.ideal_temp}°C...")
            
            pause_start = time.time()
            check_count = 0
            
            while True:
                time.sleep(10)  # Wait 10 seconds between checks
                check_count += 1
                
                current_temp = self._get_average_gpu_temperature(duration_seconds=3.0, sample_interval=0.5)
                
                if current_temp is None:
                    print("⚠️  Could not read temperature. Waiting...")
                    continue
                
                elapsed = time.time() - pause_start
                print(f"   Check #{check_count}: {current_temp:.1f}°C (elapsed: {elapsed/60:.1f} min)")
                
                if current_temp < self.ideal_temp:
                    print(f"✅ GPU temperature safe: {current_temp:.1f}°C (< {self.ideal_temp}°C)")
                    print(f"▶️  Resuming training...")
                    break
                
                if current_temp >= self.critical_temp:
                    print(f"   ⚠️  Still too hot. Waiting...")
        
        elif avg_temp >= self.max_safe_temp:
            print(f"⚠️  GPU temperature is warm: {avg_temp:.1f}°C (≥ {self.max_safe_temp}°C)")
            print(f"   Consider monitoring. Training will continue but may throttle.")
        
        else:
            print(f"✅ GPU temperature is healthy: {avg_temp:.1f}°C (< {self.max_safe_temp}°C)")
    
    def save_checkpoint(self, filename: str) -> bool:
        """
        Save model checkpoint
        
        Args:
            filename: Name of the checkpoint file
            
        Returns:
            True if checkpoint was saved successfully, False otherwise
        """
        checkpoint_path = self.output_dir / filename
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build checkpoint dictionary
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'config': self.config.to_dict() if self.config else None,
                'wandb_run_id': self.wandb_run_id  # Save W&B run ID for resuming
            }
            
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            # Save checkpoint
            print(f"💾 Saving checkpoint to {checkpoint_path}...")
            torch.save(checkpoint, checkpoint_path)
            
            # Verify file was created and has reasonable size
            if not checkpoint_path.exists():
                print(f"❌ ERROR: Checkpoint file was not created: {checkpoint_path}")
                return False
            
            file_size = checkpoint_path.stat().st_size
            if file_size == 0:
                print(f"❌ ERROR: Checkpoint file is empty (0 bytes): {checkpoint_path}")
                checkpoint_path.unlink()  # Remove empty file
                return False
            
            # Verify we can load it back (basic integrity check)
            try:
                test_load = torch.load(checkpoint_path, map_location='cpu')
                required_keys = ['model_state_dict', 'optimizer_state_dict', 'global_step']
                missing_keys = [key for key in required_keys if key not in test_load]
                if missing_keys:
                    print(f"❌ ERROR: Checkpoint missing required keys: {missing_keys}")
                    return False
                del test_load  # Free memory
            except Exception as load_error:
                print(f"❌ ERROR: Checkpoint file is corrupted and cannot be loaded: {load_error}")
                checkpoint_path.unlink()  # Remove corrupted file
                return False
            
            # Success
            file_size_mb = file_size / (1024 * 1024)
            print(f"✅ Checkpoint saved successfully: {checkpoint_path}")
            print(f"   Size: {file_size_mb:.2f} MB | Step: {self.global_step} | Epoch: {self.current_epoch}")
            return True
                
        except (AttributeError, TypeError) as e:
            # AttributeError/TypeError for serialization issues (missing attributes, incompatible types)
            print(f"❌ ERROR: Failed to serialize checkpoint (serialization error): {e}")
            print(f"   Path: {checkpoint_path}")
            if checkpoint_path.exists():
                checkpoint_path.unlink()  # Remove partial file
            return False
        except OSError as e:
            print(f"❌ ERROR: Failed to save checkpoint (OS error): {e}")
            print(f"   Path: {checkpoint_path}")
            print(f"   Check disk space and permissions")
            return False
        except RuntimeError as e:
            print(f"❌ ERROR: Failed to save checkpoint (runtime error): {e}")
            print(f"   Path: {checkpoint_path}")
            if checkpoint_path.exists():
                checkpoint_path.unlink()  # Remove partial file
            return False
        except Exception as e:
            print(f"❌ ERROR: Unexpected error saving checkpoint: {type(e).__name__}: {e}")
            print(f"   Path: {checkpoint_path}")
            import traceback
            print(f"   Traceback:\n{traceback.format_exc()}")
            if checkpoint_path.exists():
                checkpoint_path.unlink()  # Remove partial file
            return False
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[str]:
        """
        Load model checkpoint
        
        Returns:
            W&B run ID if present in checkpoint, None otherwise
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Verify scheduler step matches global_step
            scheduler_step = self.scheduler.state_dict().get('last_epoch', -1)
            print(f"Scheduler restored to step {scheduler_step}")
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        
        # Get W&B run ID if present
        wandb_run_id = checkpoint.get('wandb_run_id', None)
        if wandb_run_id:
            self.wandb_run_id = wandb_run_id
        
        # Get current learning rate after loading
        current_lr = self.optimizer.param_groups[0]['lr']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from step {self.global_step}, epoch {self.current_epoch}")
        print(f"Current learning rate: {current_lr:.2e}")
        if wandb_run_id:
            print(f"W&B run ID found: {wandb_run_id}")
        
        return wandb_run_id
