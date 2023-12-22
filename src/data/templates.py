def load_prompt_template(adjective='coherent'):
    # for comparative assessment
    # prompt_template = "{context}\n\nSummary A: {summary_A}\n\nSummary B: {summary_B}\n\nWhich Summary is more {adjective}, Summary A or Summary B?"
    prompt_template = "{context}\n\nSummary A: {summary_A}\n\nSummary B: {summary_B}\n\nWhich Summary is better, Summary A or Summary B?"
    prompt_template = prompt_template.replace('{adjective}', adjective)
    return prompt_template

def load_prompt_template_absolute(template=1, ens=False):
    # for absolute assessment
    if template == 'cot':
        prompt_template = "You will be given one summary written for a news article.\n"
        prompt_template += "Please act as an impartial judge and evaluate the quality of the summary for the news article.\n"
        prompt_template += "Your evaluation should consider the coherence, fluency, consistency and relevance.\n"
        prompt_template += "Begin your evaluation by providing a short explanation. Be as objective as possible.\n"
        prompt_template += "After providing your explanation, you must rate the summary on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating [[5]]\".\n\n"
        # prompt_template += "After providing your explanation, you must rate the summary on a scale of 1 to 10 by strictly following this format: \"[[rating]]\".\n\n"
        prompt_template += "News Article:\n\n{context}\n\n"
        prompt_template += "Summary:\n\n{summary}\n\n"
        return prompt_template
    
    else:

        prompt_template1 = "You will be given one summary written for a news article.\n"
        prompt_template1 += "Your task is to rate the summary.\n"
        prompt_template1 += "Please make sure you read and understand these instructions carefully.\n"
        prompt_template1 += "Please keep this document open while reviewing, and refer to it as needed.\n\n"
        prompt_template1 += "Evaluation Criteria:\n\nScore (1-5) - the quality of the summary.\n"
        prompt_template1 += "1. Read the news article carefully and identify the main topic and key points.\n"
        prompt_template1 += "2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.\n"
        prompt_template1 += "3. Assign a score for summary quality on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.\n\n\n"
        prompt_template1 += "Example:\n\n"
        prompt_template1 += "Source Text:\n\n{context}\n\n"
        prompt_template1 += "Summary:\n\n{summary}\n\n"
        prompt_template1 += "Predicted Score: "

        prompt_template2 = "Source Text:\n\n{context}\n\n"
        prompt_template2 += "Summary:\n\n{summary}\n\n\n"
        prompt_template2 += "You have been given one summary written for a news article.\n"
        prompt_template2 += "Your task is to rate the summary.\n"
        prompt_template2 += "Please make sure you read and understand these instructions carefully.\n"
        prompt_template2 += "Please keep this document open while reviewing, and refer to it as needed.\n\n"
        prompt_template2 += "Evaluation Criteria:\n\nScore (1-5) - the quality of the summary.\n"
        prompt_template2 += "1. Read the news article carefully and identify the main topic and key points.\n"
        prompt_template2 += "2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.\n"
        prompt_template2 += "3. Assign a score for summary quality on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.\n\n\n"
        prompt_template2 += "Predicted Score: "

        if ens:
            return prompt_template1, prompt_template2
        else:
            if template == 1:
                return prompt_template1
            else:
                return prompt_template2
    