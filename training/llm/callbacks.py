from transformers import TrainerCallback

class AdjustContextLengthCallback(TrainerCallback):
    """Dynamically increases max_completion_length during training."""

    def on_step_begin(self, args, state, control, **kwargs):
        """Adjusts max_completion_length based on training progress."""
        step = state.global_step

        if step >= 1000:
            args.max_prompt_length = args.max_prompt_length  # Allow longer completions
        elif step >= 500:
            args.max_completion_length = 256  # Gradually increase

        # Log changes
        if step in [500, 1000]:
            print(f"Adjusted max_completion_length to {args.max_completion_length} at step {step}")