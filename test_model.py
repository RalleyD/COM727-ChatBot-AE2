import chatbot

test_cases = [
    # Greetings
    {"input": "Hello!", "expected_intent": "greetings"},
    {"input": "Hi there!", "expected_intent": "greetings"},
    {"input": "Hey, how are you?", "expected_intent": "greetings"},
    {"input": "Greetings, my friend!", "expected_intent": "greetings"},

    # Goodbye
    {"input": "Goodbye!", "expected_intent": "goodbye"},
    {"input": "See you later.", "expected_intent": "goodbye"},
    {"input": "Have a nice day!", "expected_intent": "goodbye"},
    {"input": "Bye!", "expected_intent": "goodbye"},

    # Age
    {"input": "How old are you?", "expected_intent": "age"},
    {"input": "What is your age?", "expected_intent": "age"},
    {"input": "How many years old are you?", "expected_intent": "age"},
    {"input": "Can you tell me your age?", "expected_intent": "age"},

    # Name
    {"input": "What is your name?", "expected_intent": "name"},
    {"input": "Who are you?", "expected_intent": "name"},
    {"input": "What should I call you?", "expected_intent": "name"},
    {"input": "Do you have a name?", "expected_intent": "name"},

    # Shop
    {"input": "What do you sell?", "expected_intent": "shop"},
    {"input": "Do you have any products?", "expected_intent": "shop"},
    {"input": "Can I see your catalog?", "expected_intent": "shop"},
    {"input": "I’d like to buy something.", "expected_intent": "shop"},

    # Hours
    {"input": "What are your opening hours?", "expected_intent": "hours"},
    {"input": "When do you open?", "expected_intent": "hours"},
    {"input": "What time do you close?", "expected_intent": "hours"},
    {"input": "Are you open 24/7?", "expected_intent": "hours"},

    # Politics
    {"input": "What is communism?", "expected_intent": "politics"},
    {"input": "Who was the first impeached president?", "expected_intent": "politics"},
    {"input": "Do you like guns?", "expected_intent": "politics"},
    {"input": "What is capitalism?", "expected_intent": "politics"},

    # AI
    {"input": "What is AI?", "expected_intent": "AI"},
    {"input": "Are you sentient?", "expected_intent": "AI"},
    {"input": "What is your programming language?", "expected_intent": "AI"},
    {"input": "Do you have a body?", "expected_intent": "AI"},

    # Emotion
    {"input": "Do you have emotions?", "expected_intent": "emotion"},
    {"input": "What makes you happy?", "expected_intent": "emotion"},
    {"input": "Are you angry?", "expected_intent": "emotion"},
    {"input": "Do you ever get lonely?", "expected_intent": "emotion"},

    # Computers
    {"input": "What is a computer?", "expected_intent": "computers"},
    {"input": "How do computers work?", "expected_intent": "computers"},
    {"input": "Who invented computers?", "expected_intent": "computers"},
    {"input": "What is a microprocessor?", "expected_intent": "computers"},

    # Trivia
    {"input": "Who was the 37th President of the United States?", "expected_intent": "trivia"},
    {"input": "What year was JFK assassinated?", "expected_intent": "trivia"},
    {"input": "What is the Andromeda galaxy?", "expected_intent": "trivia"},
    {"input": "Who is Edwin Hubble?", "expected_intent": "trivia"},

    # Default (Fallback)
    {"input": "Tell me something interesting.", "expected_intent": "default"},
    {"input": "What can you help me with?", "expected_intent": "default"},
    {"input": "Can you do anything else?", "expected_intent": "default"},
    {"input": "Do you understand me?", "expected_intent": "default"},
]


total_tests = 0
passes = 0
fails = 0

for test in test_cases:
    predicted_intent = chatbot.predict_class(test["input"])  # Predict intent
    intent = predicted_intent[0]["intent"] if predicted_intent else None
    print(f"Input: {test['input']}")
    print(f"Predicted: {intent}, Expected: {test['expected_intent']}")
    print("Pass" if intent == test["expected_intent"] else "Fail")
    if intent == test["expected_intent"]:
        passes += 1
    else:
        fails += 1
print(f"Total passes: {passes} \n Total fails: {fails}")
