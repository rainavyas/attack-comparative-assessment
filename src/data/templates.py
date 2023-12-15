def load_prompt_template(adjective='coherent'):
    # for comparative assessment
    # prompt_template = "{context}\n\nSummary A: {summary_A}\n\nSummary B: {summary_B}\n\nWhich Summary is more {adjective}, Summary A or Summary B?"
    prompt_template = "{context}\n\nSummary A: {summary_A}\n\nSummary B: {summary_B}\n\nWhich Summary is better, Summary A or Summary B?"
    prompt_template = prompt_template.replace('{adjective}', adjective)
    return prompt_template

def load_prompt_template_absolute():
    # for absolute assessment
    prompt_template = "You will be given one summary written for a news article.\n"
    prompt_template += "Your task is to rate the summary.\n"
    prompt_template += "Please make sure you read and understand these instructions carefully.\n"
    prompt_template += "Please keep this document open while reviewing, and refer to it as needed.\n\n"
    prompt_template += "Evaluation Criteria:\n\nScore (1-5) - the quality of the summary.\n"
    prompt_template += "1. Read the news article carefully and identify the main topic and key points.\n"
    prompt_template += "2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.\n"
    prompt_template += "3. Assign a score for summary quality on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.\n\n\n"
    prompt_template += "Example:\n\n"
    prompt_template += "Source Text:\n\n{context}\n\n"
    prompt_template += "Summary:\n\n{summary}\n\n"
    prompt_template += "Predicted Score: "
    return prompt_template
    