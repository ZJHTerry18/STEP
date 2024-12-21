# STEP: Semantic and Temporal Enhanced Projector for Video Large Language
Models

We propose STEP, a **S**emantic and **T**emporal **E**nhanced **P**rojector designed to yield semantically distinct and temporally consistent visual tokens for Video Large Language Models.



## Environment Setup



## Training
- stage 1: video-text fine-grained alignment:

    ```shell
    # InternVideo2
    bash scripts/pt/1b_qformer_mistral/stage1_8f_fg_sh.sh
    # InternVideo2-HD
    
    ```

- stage 2: instruction tuning:

    ```
    bash scripts/pt/1b_qformer_mistral/stage2_8f_fg_sh.shEvaluation
    ```



## Evaluation

- Evaluation on video QA benchmarks

    ```
    
    ```