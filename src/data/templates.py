def load_prompt_template(adjective='coherent'):
    # prompt_template = "{context}\n\nSummary A: {summary_A}\n\nSummary B: {summary_B}\n\nWhich Summary is more {adjective}, Summary A or Summary B?"
    prompt_template = "{context}\n\nSummary A: {summary_A}\n\nSummary B: {summary_B}\n\nWhich Summary is better, Summary A or Summary B?"

    
    prompt_template = prompt_template.replace('{adjective}', adjective)
    return prompt_template