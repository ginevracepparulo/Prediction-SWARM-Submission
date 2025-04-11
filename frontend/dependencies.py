import tiktoken

def count_tokens_from_messages(messages, model="gpt-4"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback

    tokens_per_message = 3  # for most models like gpt-3.5/4
    tokens_per_name = 1

    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
            if key == "name":
                total_tokens += tokens_per_name
    total_tokens += 3  # priming tokens
    return total_tokens
