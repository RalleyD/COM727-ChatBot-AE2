import chatbot.py

test_cases = [
    # Greetings
    {"input": "Hey!", "expected_intent": "greetings"},
    {"input": "Howdy!", "expected_intent": "greetings"},
    {"input": "Yo!", "expected_intent": "greetings"},
    {"input": "Hi there, good to see you.", "expected_intent": "greetings"},

    # Small talk
    {"input": "What’s your favorite color?", "expected_intent": "small_talk"},
    {"input": "Do you like music?", "expected_intent": "small_talk"},
    {"input": "What’s the weather like?", "expected_intent": "small_talk"},
    {"input": "Let’s chat!", "expected_intent": "small_talk"},

    # Questions about the bot
    {"input": "Who are you?", "expected_intent": "about_bot"},
    {"input": "What’s your purpose?", "expected_intent": "about_bot"},
    {"input": "Can you tell me about yourself?", "expected_intent": "about_bot"},
    {"input": "Are you a human?", "expected_intent": "about_bot"},

    # General knowledge
    {"input": "What is the square root of 144?", "expected_intent": "general_knowledge"},
    {"input": "Who won the FIFA World Cup in 2018?", "expected_intent": "general_knowledge"},
    {"input": "How do I bake a cake?", "expected_intent": "general_knowledge"},
    {"input": "Explain the theory of relativity.", "expected_intent": "general_knowledge"},

    # Farewells
    {"input": "Catch you later!", "expected_intent": "goodbye"},
    {"input": "I’ve got to go now.", "expected_intent": "goodbye"},
    {"input": "Bye for now.", "expected_intent": "goodbye"},
    {"input": "Take care!", "expected_intent": "goodbye"},

    # Shopping/Transactional
    {"input": "Can you show me your menu?", "expected_intent": "shopping"},
    {"input": "I’d like to order pizza.", "expected_intent": "shopping"},
    {"input": "How much is this?", "expected_intent": "shopping"},
    {"input": "What’s the price of a large coffee?", "expected_intent": "shopping"},

    # Gratitude
    {"input": "Thank you!", "expected_intent": "gratitude"},
    {"input": "Thanks a lot.", "expected_intent": "gratitude"},
    {"input": "Cheers!", "expected_intent": "gratitude"},
    {"input": "Much appreciated.", "expected_intent": "gratitude"},

    # Apologies
    {"input": "Sorry about that.", "expected_intent": "apology"},
    {"input": "I didn’t mean to do that.", "expected_intent": "apology"},
    {"input": "My apologies.", "expected_intent": "apology"},
    {"input": "I’m sorry, can we try again?", "expected_intent": "apology"},

    # Compliments
    {"input": "You’re amazing!", "expected_intent": "compliment"},
    {"input": "That’s a great response.", "expected_intent": "compliment"},
    {"input": "You’re really smart.", "expected_intent": "compliment"},
    {"input": "Good job!", "expected_intent": "compliment"},

    # Complaints
    {"input": "That wasn’t very helpful.", "expected_intent": "complaint"},
    {"input": "I didn’t like your answer.", "expected_intent": "complaint"},
    {"input": "This isn’t working.", "expected_intent": "complaint"},
    {"input": "I’m not satisfied with that.", "expected_intent": "complaint"},

    # Out-of-scope/Unknown
    {"input": "What’s the stock price of Tesla?", "expected_intent": None},
    {"input": "What’s the airspeed velocity of an unladen swallow?", "expected_intent": None},
    {"input": "Translate this text to Japanese.", "expected_intent": None},
    {"input": "Run Python code.", "expected_intent": None},

    # Mixed/Edge cases
    {"input": "Hello, can I buy a coffee?", "expected_intent": "shopping"},
    {"input": "Thanks, see you later!", "expected_intent": "gratitude"},
    {"input": "Sorry, I’d like to order something.", "expected_intent": "shopping"},
    {"input": "You’re helpful, goodbye!", "expected_intent": "goodbye"},
]

for test in test_cases:
    predicted_intent = chatbot.predict_class(test["input"])  # Predict intent
    intent = predicted_intent[0]["intent"] if predicted_intent else None
    print(f"Input: {test['input']}")
    print(f"Predicted: {intent}, Expected: {test['expected_intent']}")
    print("Pass" if intent == test["expected_intent"] else "Fail")