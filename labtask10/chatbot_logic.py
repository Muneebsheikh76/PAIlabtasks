import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.chat.util import Chat, reflections

nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

menu = {
    "BBQ Pizza": "Rs. 1800",
    "Margherita Pizza": "Rs. 1200",
    "Zinger Burger": "Rs. 450",
    "Beef Burger": "Rs. 550",
    "Chicken Biryani": "Rs. 650",
    "Vegetable Biryani": "Rs. 450",
    "Pasta Alfredo": "Rs. 650",
    "Garlic Bread": "Rs. 200",
    "French Fries": "Rs. 150",
    "Cold Drink": "Rs. 80"
}

orders_db = {}
reservations_db = {}
_next_order_id = 1000
_next_res_id = 2000

pairs = [
    [r"(?i).*hello.*|.*hi.*|.*hey.*",
     ["Welcome to FoodPoint! How can I help you?",
      "Hello! Ask me about menu, reservations, or order tracking."]],

    [r"(?i).*menu.*|.*show.*menu.*|.*what.*do you have.*",
     ["Here is our menu with prices:\n\n" + "\n".join([f"{item}: {price}" for item, price in menu.items()])]],

    [r"(?i).*price.*(pizza|burger|biryani|pasta|fries|drink).*",
     ["Which item specifically? We have: " + ", ".join(list(menu.keys())[:6])]],

    [r"(?i).*reserve.*|.*reservation.*|.*book.*table.*",
     ["To reserve a table, please provide: name, date (YYYY-MM-DD), time (HH:MM), and party size.",
      "I can help you reserve a table. Please give me name, date, time and party size."]],

    [r"(?i).*book table (?P<name>[A-Za-z ]+), (?P<date>\d{4}-\d{2}-\d{2}), (?P<time>\d{2}:\d{2}), (?P<size>\d+)",
     ["Thanks! Creating reservation..."]],

    [r"(?i).*order.*|.*place order.*",
     ["To place an order, please send: item name and quantity. Example: Order Chicken Biryani, 2"]],

    [r"(?i).*order (?P<item>[A-Za-z ]+),? ?(?P<qty>\d+)",
     ["Placing order..."]],

    [r"(?i).*track.*order.*|.*order status.*|.*where is my order.*",
     ["Please provide your Order ID to check status."]],

    [r"(?i).*rating.*|.*rate.*|.*feedback.*",
     ["Please rate our service from 1 to 5 "]],

    [r"(?i).*1 star.*|.*\b1\b.*",
     [" We're sorry. We'll try to improve."]],
    [r"(?i).*2 star.*|.*\b2\b.*",
     [" Thank you for the feedback. We'll improve."]],
    [r"(?i).*3 star.*|.*\b3\b.*",
     [" Thanks! We appreciate your feedback."]],
    [r"(?i).*4 star.*|.*\b4\b.*",
     [" Great! Happy to serve you."]],
    [r"(?i).*5 star.*|.*\b5\b.*",
     [" Thank you for the 5-star rating!"]],

    [r"(?i).*bye.*|.*goodbye.*|.*see you.*",
     ["Goodbye! Have a tasty day!"]]
]

chatbot = Chat(pairs, reflections)

def analyze_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def _generate_order_id():
    global _next_order_id
    oid = f"ORD{_next_order_id}"
    _next_order_id += 1
    return oid


def _generate_res_id():
    global _next_res_id
    rid = f"RES{_next_res_id}"
    _next_res_id += 1
    return rid



def get_bot_response(user_message):
  
    text = user_message.strip()

   
    import re
    m = re.search(r"(?i)order\s+([A-Za-z ]+),?\s*(\d+)", text)
    if m:
        item = m.group(1).strip().title()
        qty = int(m.group(2))
        if item not in menu:
            return f"Sorry, we don't have '{item}' on the menu. Ask for 'menu' to see available items."
        order_id = _generate_order_id()
        orders_db[order_id] = {"item": item, "qty": qty, "status": "Preparing"}
        return f"Order placed! Order ID: {order_id}. Item: {item}, Qty: {qty}. We'll notify you when it's out for delivery."

    m2 = re.search(r"(?i)book table[:]?\s*([A-Za-z ]+),?\s*(\d{4}-\d{2}-\d{2}),?\s*(\d{2}:\d{2}),?\s*(\d+)", text)
    if m2:
        name = m2.group(1).strip().title()
        date = m2.group(2)
        time = m2.group(3)
        size = int(m2.group(4))
        res_id = _generate_res_id()
        reservations_db[res_id] = {"name": name, "date": date, "time": time, "size": size}
        return f"Reservation confirmed! Reservation ID: {res_id}. Name: {name}, Date: {date}, Time: {time}, Party: {size}."

    m3 = re.search(r"(?i)(ORD\d+)", text)
    if m3:
        oid = m3.group(1)
        if oid in orders_db:
            st = orders_db[oid]["status"]
            return f"Order {oid} status: {st}"
        else:
            return f"Order ID {oid} not found. Check your order number."

    if re.search(r"(?i).*reserve.*|.*reservation.*|.*book.*table.*", text):
        return (
            "To reserve a table, send: Book table: YourName, YYYY-MM-DD, HH:MM, party_size\n"
            "Example: Book table: Ali Khan, 2025-12-01, 19:30, 4"
        )

    if re.search(r"(?i).*menu.*|.*what do you have.*|.*show.*menu.*", text):
        return "Here is our menu:\n\n" + "\n".join([f"{item}: {price}" for item, price in menu.items()])

    if re.search(r"(?i).*track.*order.*|.*where is my order.*|.*order status.*", text):
        return "Please provide your Order ID (like ORD1000) so I can check the status."

    if re.search(r"(?i).*sentiment.*|.*feel.*|.*mood.*", text):
        return "Sentiment: " + analyze_sentiment(text)

    response = chatbot.respond(text)
    if response:
        return response

    return (
        "Sorry, I didn't understand that. Try asking about:\n"
        "- 'menu' to see items and prices\n"
        "- 'Order <item>, <qty>' to place order (example: Order Chicken Biryani, 2)\n"
        "- 'Book table: Name, YYYY-MM-DD, HH:MM, size' to reserve a table\n"
        "- 'Track ORD1000' to check order status\n"
        "- 'rating' to leave feedback"
    )
