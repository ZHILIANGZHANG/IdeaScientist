"""Raw word lists. Per-tokenizer filtering (single-token check) happens in `pools.py`.

Categories are chosen to be mutually exclusive for a lay reader (no 'bird' next to
'animal', no tomato/pepper between fruit and vegetable, no polysemes like 'file',
'lead', 'organ', 'tie').
"""
from __future__ import annotations

CATEGORY_ITEMS: dict[str, list[str]] = {
    "fruit": ["apple", "banana", "cherry", "grape", "lemon", "mango", "orange", "peach", "pear",
              "plum", "melon", "kiwi", "apricot", "coconut", "pineapple", "strawberry", "fig",
              "papaya"],
    "animal": ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "lion", "tiger", "bear",
               "wolf", "fox", "rabbit", "mouse", "deer", "monkey", "zebra", "camel", "elephant",
               "giraffe"],
    "tool": ["hammer", "wrench", "saw", "drill", "shovel", "rake", "axe", "chisel",
             "screwdriver", "pliers", "ladder", "clamp", "spade", "crowbar", "trowel"],
    "vehicle": ["car", "truck", "bus", "bicycle", "train", "boat", "ship", "plane", "tractor",
                "van", "taxi", "scooter", "motorcycle", "helicopter", "jeep", "tram", "canoe"],
    "instrument": ["piano", "guitar", "violin", "drum", "flute", "trumpet", "harp", "cello",
                   "clarinet", "saxophone", "banjo", "trombone", "tuba", "oboe", "accordion",
                   "ukulele"],
    "clothing": ["shirt", "jacket", "coat", "dress", "skirt", "sweater", "scarf", "glove",
                 "sock", "hat", "boot", "shoe", "belt", "vest", "hoodie", "blouse"],
    "furniture": ["chair", "table", "sofa", "bed", "desk", "shelf", "cabinet", "stool",
                  "bench", "couch", "dresser", "wardrobe", "bookcase", "armchair", "cupboard"],
    "vegetable": ["carrot", "potato", "onion", "cabbage", "lettuce", "broccoli", "spinach",
                  "garlic", "celery", "radish", "turnip", "beet", "kale", "leek", "cauliflower",
                  "asparagus"],
    "drink": ["coffee", "tea", "juice", "milk", "wine", "beer", "soda", "lemonade", "whiskey",
              "vodka", "cocoa", "brandy", "cider", "rum"],
    "metal": ["iron", "copper", "gold", "silver", "steel", "zinc", "nickel", "aluminum",
              "bronze", "brass", "platinum", "titanium", "chromium"],
    "flower": ["rose", "tulip", "lily", "daisy", "orchid", "sunflower", "lotus", "jasmine",
               "poppy", "lavender", "dahlia", "magnolia", "carnation"],
}

# Preference order when more than `n_categories` survive tokenizer filtering.
CATEGORY_PREFERENCE = ["fruit", "animal", "tool", "vehicle", "instrument", "clothing",
                       "furniture", "vegetable", "drink", "metal", "flower"]

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack",
         "Kate", "Leo", "Mary", "Nick", "Olivia", "Paul", "Rachel", "Sam", "Tom", "Victor",
         "Wendy", "Zoe", "Anna", "Ben", "Chris", "Dan", "Eva", "Fred", "George", "Helen", "Ian",
         "James", "Laura", "Mark", "Nina", "Oscar", "Peter", "Sarah", "Tony", "Adam", "Bella",
         "Carl", "Diana", "Eric", "Fiona", "Greg", "Hannah", "Isaac", "Julia", "Kevin", "Linda",
         "Mike", "Nora", "Owen", "Paula", "Ruth", "Steve", "Tina", "Uma", "Walter"]

# Two-letter box labels; skip strings that are common English words/abbreviations.
_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I / O (confusable with 1 / 0)
_LABEL_STOP = {"AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS", "IT",
               "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO", "TO", "UP", "US", "WE", "TV",
               "PM", "UK", "EU", "UN", "DC", "LA", "NY", "PC", "ID", "AI", "OS", "HQ", "VS",
               "XX", "AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "JJ", "KK", "LL", "MM",
               "NN", "PP", "QQ", "RR", "SS", "TT", "UU", "VV", "WW", "YY", "ZZ"}
BOX_LABELS = [a + b for a in _LETTERS for b in _LETTERS if a + b not in _LABEL_STOP]
