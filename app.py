from flask import Flask, render_template, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import time
import transformers
import torch

app = Flask(__name__)

# Load the model once at the start
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def websearch(query):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run without UI
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")  # Disable images
    chrome_options.add_argument("--disable-extensions")
    
    service = Service('')  # Adjust the path for your ChromeDriver

    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get('https://www.google.com')
        search_box = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.NAME, 'q'))
        )
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)  # Press Enter to search

        first_result = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.tF2Cxc .yuRUbf a'))
        )
        first_url = first_result.get_attribute('href')
        print(f"\nDEBUG: The websearch url: {first_url}")

    finally:
        driver.quit()

    try:
        response = requests.get(first_url, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        })

        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.body.get_text(separator=' ', strip=True)
        content = ' '.join(content.split())

    except Exception as e:
        content = f"An error occurred while scraping the page: {e}"

    return content

def query_analysis(user_query):
    print(f'DEBUG: Analyzing the query for websearch.')
    messages = [
        {"role": "system", "content": "You are an AI assistant. For any real-time or up-to-date data query, such as current time, exchange rates, live weather, or current events, respond with **only** 'FALSE'. For general knowledge questions, respond with **only** 'TRUE'."},
        {"role": "user", "content": "What's the current time in Pakistan?"},
        {"role": "assistant", "content": "FALSE"},
        {"role": "user", "content": "What is the current exchange rate of USD to EUR?"},
        {"role": "assistant", "content": "FALSE"},
        {"role": "user", "content": "What is the temperature in Tokyo right now?"},
        {"role": "assistant", "content": "FALSE"},
        {"role": "user", "content": "Who won the latest football World Cup?"},
        {"role": "assistant", "content": "FALSE"},
        {"role": "user", "content": "Who was the first president of the United States?"},
        {"role": "assistant", "content": "TRUE"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "TRUE"},
        {"role": "user", "content": "What is the speed of light in a vacuum?"},
        {"role": "assistant", "content": "TRUE"},
        {"role": "user", "content": "Who developed the theory of relativity?"},
        {"role": "assistant", "content": "TRUE"},
        {"role": "user", "content": f"{user_query}"},
    ]

    outputs = pipeline(messages, max_new_tokens=5)
    response = outputs[0]["generated_text"]
    return response[-1]['content']

def infer_llm(user_query, scraped_data=''):
    messages = [
        {"role": "system", "content": f"You are an AI assistant. Answer the user's question in a clear and helpful way only from the following knowledge base: `{scraped_data}`"},
        {"role": "user", "content": f"{user_query}"},
    ]

    if scraped_data == '':
        messages = [
            {"role": "system", "content": "You are an AI assistant. Answer the user's question in a clear and helpful way."},
            {"role": "user", "content": f"{user_query}"},
        ]

    outputs = pipeline(messages, max_new_tokens=120)
    response = outputs[0]["generated_text"]
    return response[-1]["content"]

def invoke_llm(user_query):
    print(f'DEBUG: Invoking the LLM with query: {user_query}')
    scraped_data = ''
    query_report = query_analysis(user_query)

    if query_report.upper() == 'FALSE':
        scraped_data = websearch(user_query)
    
    torch.cuda.empty_cache()

    return infer_llm(user_query, scraped_data)


@app.route('/send_message', methods=['POST'])
def send_message():
    # Expecting JSON input
    user_message = request.json.get('message')  # This will correctly extract message from JSON body
    if not user_message:  # Check for missing message
        return jsonify({"status": "error", "message": "No message provided"}), 400
    
    llm_response = invoke_llm(user_message)
    print(f'DEBUG: LLM response to user query: {llm_response}')
    
    # Return the messages including the user's message and the LLM's response
    messages = [{"sender": "User", "text": user_message},
                {"sender": "System", "text": llm_response}]

    return jsonify({"status": "success", "messages": messages})

if __name__ == '__main__':
    app.run(debug=True)
