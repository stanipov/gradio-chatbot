def isClose(a,b, eps=1e-4):
    """ Compares 2 float numbers if they are close """
    return abs(a-b) <=eps


def merge2ChatML(message, history, sys_prompt):
    """ Merges user message, history, and system prompt into ChatML format """
    messages = [{'role': 'system', 'content': sys_prompt}]
    for usr, ai in history:
        messages.append({'role': 'user', 'content': usr})
        messages.append({'role': 'assistant', 'content': ai})

    messages.append({'role': 'assistant', 'content': message})
    return messages