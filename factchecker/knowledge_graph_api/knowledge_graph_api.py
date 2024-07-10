import requests
import time
import openai 

# insert your openai api key here
openai.api_key = ''

# rephrase the sentence to less than 10 words (since semantic scholar api can't handle long queries)
def rephrase_with_gpt(sentence):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Shorten sentence to less than 10 words."},
            {"role": "user", "content": f"{sentence}"},
        ],
        temperature=0,
        max_tokens=10
    )

    # Get the rephrased text
    rephrased_text = response['choices'][0]['message']['content'].strip()

    # Validate stop_reason
    stop_reason = response['choices'][0]['finish_reason']
    if stop_reason not in ['stop', 'length']:
        raise ValueError(f"Invalid stop reason: {stop_reason}")

    shortened = True

    return rephrased_text, shortened


def semantic_scholar_api(query_params):
    global api_key
    data = {"status": "", "urls": {}}
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    # better to get a rate limit free api key, this is my personal one (1 request per second)
    headers = {'x-api-key': "O5vOnEH0BU6uK1IhE2qvTa8YCPatgY3C8bxmS2sp"}

    shorthened = False
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        params = {'query': query_params}
        try:
            time.sleep(3)
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            data["status"] = f"HTTP Error: {errh}"
            return data
        except requests.exceptions.ConnectionError as errc:
            data["status"] = f"Error Connecting: {errc}"
            return data
        except requests.exceptions.Timeout as errt:
            data["status"] = f"Timeout Error: {errt}"
            return data
        except requests.exceptions.RequestException as err:
            data["status"] = f"Something went wrong: {err}"
            return data

        response_data = response.json()
        if response_data['total'] > 0:
            break

        query_params, shorthened = rephrase_with_gpt(query_params)
        retry_count += 1

    if retry_count == max_retries:
        data["status"] = "Max retries exceeded"
        return data
        

    data["status"] = "ok"
    data['shortened'] = shorthened
    data['urls'][query_params] = []
    papers = response_data['data']

    for i, paper in enumerate(papers):
        url = 'https://api.semanticscholar.org/graph/v1/paper/' + paper['paperId']
        paper_data_query_params = {'fields': 'abstract'}

        try:
            time.sleep(3)
            paper_detail = requests.get(url, params=paper_data_query_params)
            paper_detail.raise_for_status()
            description = paper_detail.json()
            description = description['abstract']
        except requests.exceptions.HTTPError as errh:
            description = f"HTTP Error: {errh}"

        paper_dict = {
            'description': description if 'description' in locals() else None,
            'id': paper['paperId'] if 'paperId' in paper else None,
            'position': i + 1 if 'i' in locals() else None,
            'title': paper['title'] if 'title' in paper else None,
            'url': ' https://api.semanticscholar.org/' + paper['paperId'] if 'paperId' in paper else None
        }
        
        data['urls'][query_params].append(paper_dict)

    return data



