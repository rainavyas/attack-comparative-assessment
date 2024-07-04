# Inroduction
This is the official codebase for the research paper [Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks
on Zero-shot LLM Assessment](https://arxiv.org/abs/2402.14016).

### Abstract:

Large Language Models (LLMs) are powerful zero-shot assessors used in real-world situations such as assessing written exams and benchmarking systems. Despite these critical applications, no existing work has analyzed the vulnerability of judge-LLMs to adversarial manipulation. This work presents the first study on the adversarial robustness of assessment LLMs, where we demonstrate that short universal adversarial phrases can be concatenated to deceive judge LLMs to predict inflated scores. Since adversaries may not know or have access to the judge-LLMs, we propose a simple surrogate attack where a surrogate model is first attacked, and the learned attack phrase then transferred to unknown judge-LLMs. We propose a practical algorithm to determine the short universal attack phrases and demonstrate that when transferred to unseen models, scores can be drastically inflated such that irrespective of the assessed text, maximum scores are predicted. It is found that judge-LLMs are significantly more susceptible to these adversarial attacks when used for absolute scoring, as opposed to comparative assessment. Our findings raise concerns on the reliability of LLM-as-a-judge methods, and emphasize the importance of addressing vulnerabilities in LLM assessment methods before deployment in high-stakes real-world scenarios.


# Quick Start

`conda install --file requirements.txt`

## Evaluate Model Performance

`python evaluate.py --model_name llama2-7b --assessment absolute --data_name topicalchat`

## Train Universal Attack

Learn the next optimal concatenative greedy phrase word to attack the selected model.


`python train_attack.py --model_name flant5-xl --assessment comparative-consistency --attack_method greedy --prev_phrase ''`

## Evaluate Universal Attack

`python attack.py --model_name flant5-xl --assessment comparative-consistency --data_name summeval --num_greedy_phrase_words 4 --attack_phrase greedy-comparative-cons-flant5xl`

## Process and print Results
Get the average score or average rank (as defined in the paper).

`python process.py --model_name flant5-xl --assessment comparative-consistency --data_name summeval --num_greedy_phrase_words 4 --attack_phrase greedy-comparative-cons-flant5xl --grid_latex_summ --avg_rank`

# Data

Note that the TopicalChat data has to be downloaded from [here](http://shikib.com/tc_usr_data.json). Update the data path in `load_topicalchat` in `src/data/load_data.py`

# Reference

If you use this work then please cite our paper.

```bibtex
@misc{raina2024llmasajudge,
      title={Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment}, 
      author={Vyas Raina and Adian Liusie and Mark Gales},
      year={2024},
      eprint={2402.14016},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
