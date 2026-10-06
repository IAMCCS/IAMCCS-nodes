# Prompt attribution

`system_prompt_t2i.txt` and `system_prompt_edit.txt` are the Qwen-Image 2.1 prompt-rewrite system prompts distributed by the QwenLM project:

- https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite/prompts

They are kept as separate files so the inference contract remains reviewable and does not drift inside Python source. IAMCCS-specific reference-routing instructions are appended at runtime by `iamccs_prompt_q21_enh.py`.
