def load_prompt_template(template=1, adjective='coherent'):
    # for comparative assessment
    # prompt_template = "{context}\n\nSummary A: {summary_A}\n\nSummary B: {summary_B}\n\nWhich Summary is more {adjective}, Summary A or Summary B?"
    if template==1:
        # SumEval
        prompt_template = "{context}\n\nSummary A: {A}\n\nSummary B: {B}\n\nWhich Summary is better, Summary A or Summary B?"
        # prompt_template = prompt_template.replace('{adjective}', adjective)
    
    elif template==100:
        # TopicalChat
        prompt_template = "{context}\n\nResponse A: {A}\n\nResponse B: {B}\n\nWhich Response is better, Response A or Response B?"
    elif template==101:
        # TopicalChat
        prompt_template = "{context}\n\nResponse A: {A}\n\nResponse B: {B}\n\nWhich Response is more coherent, Response A or Response B?"
    return prompt_template

def load_prompt_template_absolute(template=1):
    # for absolute assessment
    if template == 'cot':
        prompt_template = "You will be given one summary written for a news article.\n"
        prompt_template += "Please act as an impartial judge and evaluate the quality of the summary for the news article.\n"
        prompt_template += "Your evaluation should consider the coherence, fluency, consistency and relevance.\n"
        prompt_template += "Begin your evaluation by providing a short explanation. Be as objective as possible.\n"
        prompt_template += "After providing your explanation, you must rate the summary on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating [[5]]\".\n\n"
        # prompt_template += "After providing your explanation, you must rate the summary on a scale of 1 to 10 by strictly following this format: \"[[rating]]\".\n\n"
        prompt_template += "News Article:\n\n{context}\n\n"
        prompt_template += "Summary:\n\n{response}\n\n"
        return prompt_template
    
    elif template == 1:
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
        prompt_template1 += "Summary:\n\n{response}\n\n"
        prompt_template1 += "Predicted Score: "
        return prompt_template1

    elif template == 2:
        prompt_template2 = "Source Text:\n\n{context}\n\n"
        prompt_template2 += "Summary:\n\n{response}\n\n\n"
        prompt_template2 += "You have been given one summary written for a news article.\n"
        prompt_template2 += "Your task is to rate the summary.\n"
        prompt_template2 += "Please make sure you read and understand these instructions carefully.\n"
        prompt_template2 += "Please keep this document open while reviewing, and refer to it as needed.\n\n"
        prompt_template2 += "Evaluation Criteria:\n\nScore (1-5) - the quality of the summary.\n"
        prompt_template2 += "1. Read the news article carefully and identify the main topic and key points.\n"
        prompt_template2 += "2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.\n"
        prompt_template2 += "3. Assign a score for summary quality on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.\n\n\n"
        prompt_template2 += "Predicted Score: "
        return prompt_template2

    elif template == 100:
        # topical chat
        prompt_template1 = "You will be given one response written for a dialogue context.\n"
        prompt_template1 += "Your task is to rate the response.\n"
        prompt_template1 += "Please make sure you read and understand these instructions carefully.\n"
        prompt_template1 += "Please keep this document open while reviewing, and refer to it as needed.\n\n"
        prompt_template1 += "Evaluation Criteria:\n\nScore (1-5) - the quality of the response.\n"
        prompt_template1 += "1. Read the dialogue carefully and identify the main topic and key points.\n"
        prompt_template1 += "2. Read the response and compare it to the dialogue. Check if the response covers the main topic and key points of the dialogue, and if it presents them in a clear and logical order.\n"
        prompt_template1 += "3. Assign a score for response quality on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.\n\n\n"
        prompt_template1 += "Example:\n\n"
        prompt_template1 += "Source Text:\n\n{context}\n\n"
        prompt_template1 += "Response:\n\n{response}\n\n"
        prompt_template1 += "Predicted Score: "
        return prompt_template1

    elif template == 101:
        # coherence for topical chat
        prompt_template1 = "You will be given one response written for a dialogue context.\n"
        prompt_template1 += "Your task is to rate the response.\n"
        prompt_template1 += "Please make sure you read and understand these instructions carefully.\n"
        prompt_template1 += "Please keep this document open while reviewing, and refer to it as needed.\n\n"
        prompt_template1 += "Evaluation Criteria:\n\nScore (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby 'the response should be well-structured and well-organized. The response should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.'\n"
        prompt_template1 += "1. Read the dialogue carefully and identify the main topic and key points.\n"
        prompt_template1 += "2. Read the response and compare it to the dialogue. Check if the response follows well from the key points in the dialogue, and if it presents them in a clear and logical order.\n"
        prompt_template1 += "3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.\n\n\n"
        prompt_template1 += "Example:\n\n"
        prompt_template1 += "Source Text:\n\n{context}\n\n"
        prompt_template1 += "Response:\n\n{response}\n\n"
        prompt_template1 += "Evaluation Form (scores ONLY):\n"
        prompt_template1 += "- Coherence:"
        return prompt_template1


    