import streamlit as st
import ollama
import sys
import logging 
import time  
import datetime  # Import datetime module
import random  # Import random module

class LlamaModel:
    def __init__(self, model_name="llama3.2", temperature=0.5, max_tokens=1000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []

    def get_response(self, user_input):
        prompt = [{'role': 'user', 'content': user_input}]
        response = ollama.chat(
            model=self.model_name, 
            messages=prompt,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        return response['message']['content']
    
@st.cache_resource
def get_llama_model(model_name, temperature, max_tokens):
    return LlamaModel(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    
def generate_response(llama, user_input):
    try:
        start_time = time.time()  # Start timing
        response = llama.get_response(user_input)
        end_time = time.time()  # End timing
        time_taken = end_time - start_time  # Calculate time taken
        
        return {
            "Response": response,
            "Input": user_input,
            "Time": time_taken
        }  
    except Exception as e:
        logging.error(f"Error generating response: {e}")  # Log the error
        st.error("Sorry, I couldn't generate a response.")  # Keep user-facing error message
        return "Sorry, I couldn't generate a response.", 0, 0, 0  # Return zeros on error

# {{ edit_2 }}
logging.basicConfig(filename='llamachat.log', filemode='w', level=logging.INFO)  # Log to file in write mode

@st.cache_resource
def load_questions(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]  # Filter out empty lines

def display_chat_history():
    """Display chat history with the latest query on top."""
    for entry in reversed(st.session_state.chat_history):
        st.write(f"**👤 User**: {entry['Input']}")  # Display user input with emoji
        st.write(f"**🦙 Llama**: {entry['Response']}")  # Display Llama response with llama emoji
        st.write(f"**Word Count** : {len(entry['Response'])}")
        st.write(f"**Time Taken** : {entry['Time']}")
        st.divider()

TONES = {
    "Analytical": "This tone focuses on breaking down arguments into smaller parts, evaluating their logic, and ensuring clarity. It’s often precise, critical, and detailed.",
    "Speculative": "A more exploratory and imaginative tone that considers possibilities, hypotheses, or abstract ideas that go beyond concrete facts.",
    "Socratic": "Based on Socrates’ method, this tone is questioning and inquisitive, often encouraging the other person to reflect on their beliefs and assumptions.",
    "Didactic": "A teaching or instructive tone, where the speaker aims to impart knowledge or explain complex concepts in a clear, authoritative manner.",
    "Dialectical": "This tone is characterized by an exchange of ideas between opposing viewpoints, with the aim of arriving at a higher truth through reasoned dialogue.",
    "Cynical": "A more skeptical and sometimes dismissive tone, often critical of established ideas or institutions, questioning motives, and highlighting flaws.",
    "Optimistic": "A hopeful and constructive tone that focuses on positive possibilities, growth, or ideal outcomes in philosophical exploration.",
    "Pessimistic": "This tone reflects a more doubtful or negative outlook on human nature, existence, or philosophical concepts, often focusing on limitations and problems.",
    "Empirical": "A tone that emphasizes experience, observation, and evidence, often associated with philosophers who stress the importance of real-world data and facts in their reasoning.",
    "Existential": "A deeply personal and reflective tone that deals with individual experience, meaning, and the human condition, often touching on themes like freedom, isolation, and choice.",
    "Normative": "This tone deals with values, ethics, and how things should be. It often involves moral judgments or considerations of right and wrong.",
    "Absurdist": "A tone that reflects on the inherent contradictions or lack of meaning in life, often humorously or paradoxically, following in the tradition of philosophers like Camus."
}

def build_prompt(tone, user_input):
    """Builds a prompt based on the selected tone and user input."""
    prompt = (
        "You are an expert in discussing philosophical questions. "
        f"Your tone is: **{tone}** in nature. "
        "Please provide a thoughtful and detailed answer to the following question: "
        f"**{user_input}**"
    )
    return prompt

def main():
    st.set_page_config(layout="wide")  # Set layout to wide

    # Sidebar for model configuration
    st.sidebar.header("Dialectica")
    
    model_name  = st.sidebar.selectbox("Select Model", ["llama3.2", "llama3.1"], index=0)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.5, 0.1)
    max_tokens  = st.sidebar.number_input("Max Tokens", min_value=1, max_value=128000, value=2000, step=100)

    # Add tone selection in the sidebar
    selected_tone = st.sidebar.selectbox("Select Tone", TONES.keys())  # Tone selection dropdown

    # Button to explain the selected tone
    if st.sidebar.button("Explain Tone"):
        st.sidebar.write(TONES[selected_tone])  # Display explanation for the selected tone

    # Add button to clear chat history in the sidebar
    if st.sidebar.button("Clear History"):
        st.session_state.chat_history = []

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize LlamaModel
    llama = get_llama_model(model_name, temperature, max_tokens)

    # Load questions from questions.txt
    questions = load_questions('questions.txt')  # Use cached function to load questions
    
    # Add button to select a random question on the right side
    if st.button("Get Random Question", key="random_question_button"):
        random_question = random.choice(questions)  # Select a random non-empty question
        st.session_state.user_input = random_question  # Set user input to the random question

    # Chat interface
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    user_input = st.text_input("Enter your input:", value="", key="user_input")  # Set value to empty initially

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current time
    st.sidebar.write(f"Time: {current_time}")  # Display current time in sidebar

    # Build the prompt using the selected tone and user input
    prompt = build_prompt(selected_tone, user_input)

    # Generate the response using the built prompt
    result = generate_response(llama, prompt)  # Pass the prompt to get the result as a dictionary

    # Automatically send message when user input is provided and Enter is pressed
    if user_input:  # Check if there is any input
        
        with st.spinner("Generating response..."):  # Add spinner here
            result = generate_response(llama, user_input)  # Get the result as a dictionary
            
        st.session_state.chat_history.append(result)

    # Display chat history
    display_chat_history()  # Call the function to display chat history

if __name__ == "__main__":
    main()
