import requests

url = "https://twitter-v24.p.rapidapi.com/user/tweets"

def tweet_return(username):
    querystring = {"username": username, "limit": "50"}  # Construct the querystring
    headers = {
        "X-RapidAPI-Key": "fdba32988cmsh32050ad5d11e0cep1385fcjsncea5a1d350b6",
        "X-RapidAPI-Host": "twitter-v24.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    response_data = response.json()

    all_full_text = ""
    instructions = response_data['data']['user']['result']['timeline_v2']['timeline']['instructions']
    last_instruction_index = len(instructions) - 1
    
    for entry in instructions[last_instruction_index]['entries']:
        if ('itemContent' in entry['content']) and ('retweeted_status_result' not in entry['content']['itemContent']['tweet_results']['result']['legacy']):
            full_text = entry['content']['itemContent']['tweet_results']['result']['legacy']['full_text']
            all_full_text += full_text + " \n" # Append full_text to the result string

    return all_full_text


    