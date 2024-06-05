from flask import Flask, render_template, request, session, jsonify
import openai
import logging
import os

app = Flask(__name__)
# This is needed to use sessions
app.secret_key = os.urandom(24)
# Load the API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to handle API calls
def openai_api_call(messages):
    try:
        logging.debug(f"Messages sent to OpenAI: {messages}")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1500
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        logging.error(f"API call failed: {e}")
        return None

# Function to generate the summary and the "act as" prompt
def generate_summary_and_prompt(messages):
    summary_instruction = (
        "Please summarize the key requirements provided by the user in bullet points. "
        "Based on these requirements, suggest an appropriate persona to 'act as' to fulfill the user's request."
        "Act as a prompt generator for ChatGPT. Use the suggested persona and summary to engineer a prompt that would yield the best and most desirable response from ChatGPT." 
        "Each prompt should involve asking ChatGPT to 'act as' the persona given"
        "The prompt should be detailed and comprehensive and should build on what was requested to generate the best possible response from ChatGPT." 
        "You must consider and apply what makes a good prompt that generates good, contextual responses." 
        "You must give a summary of the key requirements, a suggested persona, and output the prompt you want to use."
    )
    messages.append({"role": "system", "content": summary_instruction})
    summary_response = openai_api_call(messages)    
    return summary_response


# Function to generate the final response based on the suggested prompt
def generate_final_response(messages, suggested_prompt):
    final_instruction = (
        f"Based on the following summary and suggested persona, provide the final response:\n\n"
        f"{suggested_prompt}"
    )
    messages.append({"role": "system", "content": final_instruction})
    final_response = openai_api_call(messages)
    return final_response




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_message = data.get('prompt')
        logging.info(f"Received prompt: {user_message}")

        if not user_message:
            return jsonify({'error': 'No prompt provided'}), 400

        if 'messages' not in session:
            session['messages'] = []

        # Append the user's message to the conversation history
        session['messages'].append({"role": "user", "content": user_message})

        # Check if the AI has already asked for more details
        if len(session['messages']) > 1 and session['messages'][-2]['role'] == 'assistant':
            # The last AI response should have been a follow-up question, so use the user's response to generate the next message
            response = openai_api_call(session['messages'])

            if response:
                session['messages'].append({"role": "assistant", "content": response})
                session.modified = True

                # Generate summary and "act as" prompt
                summary_response = generate_summary_and_prompt(session['messages'])
                logging.info(f"Summary Reponse: {summary_response}")

                if summary_response:
                    session['messages'].append({"role": "assistant", "content": summary_response})
                    session.modified = True
                    # Generate final response based on the summary and suggested prompt
                    final_response_content = generate_final_response(session['messages'], summary_response)
                    logging.info(f"Final response: {final_response_content}")
                    if final_response_content:
                        return jsonify({'summary': summary_response, 'final_response': final_response_content})
                    else:
                        return jsonify({'error': 'Final response generation failed'}), 500
                else:
                    return jsonify({'error': 'Summary generation failed'}), 500
            else:
                return jsonify({'error': 'API call failed'}), 500
        else:
            # Generate a prompt to ask for more details if needed
            instructions = (
                "Help the user develop a clear set of requirements."
                "Ask for the top 3 most important details to help you develop the best prompt possible to fulfil the user's request."
            )

            # Add instructions to the conversation history
            session['messages'].insert(0, {"role": "system", "content": instructions})

            # Send the conversation history to ChatGPT and get a response
            response = openai_api_call(session['messages'])

            # Remove the system instruction to keep the session clean for future interactions
            session['messages'].pop(0)

            # Add ChatGPT's response to the conversation history
            if response:
                session['messages'].append({"role": "assistant", "content": response})
                session.modified = True
                logging.info(f"API response: {response}")
                return jsonify({'response': response})
            else:
                return jsonify({'error': 'API call failed'}), 500
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)


