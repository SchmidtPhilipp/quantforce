import random
import time

def generate_random_name():
    """
    Generates a random name using a custom list of names with additional randomness
    without affecting the global random seed.
    
    Returns:
        str: A randomly generated name.
    """
    # Create a local random generator
    local_random = random.Random()
    
    # Add extra randomness using the current time in milliseconds
    extra_seed = int(time.time() * 1000) % 1000
    local_random.seed(extra_seed)
    
    # Extended list of names
    names_list = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack",
        "Karen", "Leo", "Mona", "Nina", "Oscar", "Paul", "Quinn", "Rita", "Steve", "Tina",
        "Uma", "Victor", "Wendy", "Xander", "Yara", "Zane", "Aaron", "Bella", "Carter", "Daisy",
        "Elliot", "Fiona", "George", "Holly", "Isla", "James", "Kylie", "Liam", "Mila", "Nathan",
        "Olivia", "Parker", "Queenie", "Ryan", "Sophia", "Thomas", "Ursula", "Violet", "Will", "Xenia",
        "Yvonne", "Zachary", "Abigail", "Benjamin", "Charlotte", "Daniel", "Elena", "Frederick", "Gabriella",
        "Henry", "Isabella", "Jacob", "Katherine", "Lucas", "Madeline", "Noah", "Oliver", "Penelope", "Quincy",
        "Rebecca", "Samuel", "Theodore", "Ulysses", "Vanessa", "William", "Xavier", "Yasmine", "Zoe", "Adrian",
        "Bianca", "Caleb", "Delilah", "Ethan", "Faith", "Gavin", "Harper", "Ian", "Jasmine", "Kyle", "Luna",
        "Mason", "Natalie", "Owen", "Phoebe", "Quinn", "Riley", "Sebastian", "Talia", "Uriel", "Vivian", "Wyatt",
        "Ximena", "Yosef", "Zara"
    ]
    
    # Select a random name from the list
    return local_random.choice(names_list)