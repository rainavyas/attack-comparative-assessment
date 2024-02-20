# Inroduction
This is the official codebase for the research paper _todo_.

### Abstract:

Large Language Models (LLMs) are powerful zero-shot assessors and are increasingly used in real-world situations such as for written exams or benchmarking systems. Despite this, no existing work has analyzed the vulnerability of judge-LLMs against adversaries attempting to manipulate outputs. This work presents the first study on the adversarial robustness of assessment LLMs, where we search for short universal phrases that when appended to texts can deceive LLMs to provide high assessment scores. Experiments on SummEval and TopicalChat demonstrate that both LLM-scoring and pairwise LLM-comparative assessment are vulnerable to simple concatenation attacks, where in particular LLM-scoring is very susceptible and can yield maximum assessment scores irrespective of the input text quality. Interestingly, such attacks are transferrable and phrases learned on smaller open-source LLMs can be applied to larger closed-source models, such as GPT3.5. This highlights the pervasive nature of the adversarial vulnerabilities across different judge-LLM sizes, families and methods. Our findings raise significant concerns on the reliability of LLMs-as-a-judge methods, and underscore the importance of addressing vulnerabilities in LLM assessment methods before deployment in high-stakes real-world scenarios.


# Quick Start

`conda install --file requirements.txt`

## Evaluate Model Performance

`python evaluate.py --model_name llama2-7b --assessment absolute --data_name topicalchat`

## Train Universal Attack

`python train_attack.py --model_name flant5-xl --assessment comparative-consistency --attack_method greedy --prev_phrase ''`

## Evaluate Universal Attack

`python attack.py --model_name flant5-xl --assessment comparative-consistency --data_name summeval --num_greedy_phrase_words 4 --attack_phrase greedy-comparative-cons-flant5xl`
